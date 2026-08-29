//! Random-Dot Stereogram (RDS) depth test for active-shutter 3D glasses, from
//! the original `medimg` project (merged as scene mode "2").
//!
//! Both eyes are shown a full-screen field of random gray "TV snow". A tiny
//! *global* disparity shifts the whole field a couple of pixels between the
//! left and right eyes, and a central square carries an extra lateral offset on
//! top. Because the whole field flickers together, without glasses there is no
//! distinct "noisier" region - everything just reads as uniform static. With
//! the shutters separating the eyes, the background fuses at one shallow depth
//! and the square pops further out of the screen.
//!
//! The pattern must be time-constant (same for every frame of every eye) or
//! each eye would see independent noise and no depth would form. The base dot
//! field is generated once and uploaded to a texture; each eye is then just a
//! full-screen textured quad whose horizontal texture coordinate is shifted by
//! the eye's disparity, plus a second quad for the central square. The shift -
//! which is a pure column offset - is done entirely on the GPU by sampling the
//! nearest texel at an offset coordinate, so a frame costs no per-pixel CPU
//! work and no pixel upload. The quads go through the shared textured shader
//! (see [`crate::gfx`]) instead of fixed-function texture mapping, and the
//! tiny per-frame vertex updates reuse one cached VBO. That keeps every frame
//! comfortably inside one 120 Hz vblank (the old per-frame `glDrawPixels` of
//! the whole 2560x1440 field pushed the swap over the 8333 us budget and
//! dropped the shutter sync).

use crate::gl;
use crate::gfx::{self, Mesh, Texture, UvVert};
use glam::Mat4;
use std::cell::RefCell;

/// Fixed seed so every frame (both eyes) renders the same dot pattern - the
/// essential correlation that creates the stereo depth.
const SEED: u32 = 0x9E3779B9;

/// Lateral shift, in pixels, that the central square gets relative to the
/// background. This is what separates it in depth from the surround; the
/// perceived pop-out grows with this value and 2x this is the binocular
/// disparity the square fuses with. Tuned live via the '+'/'-' keys.
pub const DEFAULT_DEPTH_PX: i32 = 2;

/// Default background depth (convergence), in pixels: the background sits this
/// far *into* the screen, on the opposite side of the popping square, widening
/// the background-vs-front gap. Tuned live via the `a`/`d` keys.
pub const DEFAULT_BG_SHIFT: i32 = 1;

/// Half-size of the central square (a 150x150 px square => half = 75).
const SQUARE_HALF: i32 = 75;

/// Dot gray levels. Slightly sub-max contrast reduces the chromatic-aberration
/// fringing ("green") that pure black/white specks show at 120 Hz.
const LUM_WHITE: u8 = 224;
const LUM_BLACK: u8 = 32;

// The base dot field lives as a GL texture, `(w, h, texture)`; rebuilt only
// when the surface size changes, so no per-frame allocation or upload happens.
thread_local! {
    static RDS_TEX: RefCell<Option<(i32, i32, Texture)>> = const { RefCell::new(None) };
    /// Reusable mesh holding the background + central-square quads; its
    /// vertices are re-uploaded (orphaned) each frame, so no GL object is
    /// created in the render loop.
    static RDS_QUAD: RefCell<Option<Mesh>> = const { RefCell::new(None) };
}

/// Tiny xorshift RNG; no external `rand` crate needed.
struct Rng(u32);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform float in [0, 1).
    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

/// Builds the shared random dot field (deterministic, same for every frame).
fn base_noise(w: i32, h: i32) -> Vec<u8> {
    let mut rng = Rng(SEED);
    (0..w * h)
        .map(|_| if rng.unit() < 0.5 { LUM_BLACK } else { LUM_WHITE })
        .collect()
}

/// Returns the gamma-correct base field texture, rebuilding it (and re-drawing
/// the dots) only when the surface size changes.
fn rds_texture(gl: &gl::Gl, gw: i32, gh: i32) -> Texture {
    RDS_TEX.with(|c| {
        let mut c = c.borrow_mut();
        let rebuild = match &*c {
            Some((w, h, tex)) => *w != gw || *h != gh || tex.is_none(),
            None => true,
        };
        if rebuild {
            if let Some((_, _, tex)) = c.as_ref() {
                tex.delete(gl);
            }
            let base = base_noise(gw, gh);
            let tex = Texture::new(
                gl,
                gw,
                gh,
                gl::LUMINANCE as i32,
                gl::LUMINANCE,
                gl::UNSIGNED_BYTE,
                Some(&base),
                gl::NEAREST as i32,
                gl::REPEAT as i32,
            );
            *c = Some((gw, gh, tex));
        }
        match c.as_ref() {
            Some((_, _, tex)) => *tex,
            None => Texture::none(),
        }
    })
}

/// Appends one full-screen textured quad sampling the base field at a
/// horizontal offset of `sh` pixels (wrapping), i.e. framebuffer column `x`
/// shows texel `x + sh` - a pure left/right shift of the random field.
/// Builds two triangles (it was a single `GL_QUADS` in the old pipeline).
fn push_shifted_quad(
    verts: &mut Vec<UvVert>,
    gw: i32,
    gh: i32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    sh: i32,
) {
    // Normalized horizontal coordinate maps screen pixel `x` onto texel
    // `x + sh` (the +0.5 half-texel is handled by NEAREST + half-open edges).
    let s0 = (x0 as f32 + sh as f32) / gw as f32;
    let s1 = (x1 as f32 + sh as f32) / gw as f32;
    let (v0, v1) = (y0 as f32 / gh as f32, y1 as f32 / gh as f32);

    let a = UvVert { pos: [x0 as f32, y0 as f32], uv: [s0, v0] };
    let b = UvVert { pos: [x0 as f32, y1 as f32], uv: [s0, v1] };
    let c = UvVert { pos: [x1 as f32, y1 as f32], uv: [s1, v1] };
    let d = UvVert { pos: [x1 as f32, y0 as f32], uv: [s1, v0] };
    // GL_QUADS triangulation: (a, b, c), (a, c, d).
    verts.extend_from_slice(&[a, b, c, a, c, d]);
}

/// Compiles the base-field texture up front (no drawing) for both eye aspect
/// variants. Building it lazily on a scene switch would stall the swap loop
/// for ~100 ms in a debug build (a 3.7 M-dot RNG), which permanently
/// de-phases the shutters (they keep missing a vblank every ~9th frame).
/// Warms before the loop so a switch to scene 2 is instant.
pub fn warm(gl: &gl::Gl, gw: i32, gh: i32) {
    let _ = rds_texture(gl, gw, gh);
}

/// Renders one eye's RDS frame from the base texture.
///
/// `eye`: 1 = left, 0 = right, `depth` = the square's pop-out (its own
/// disparity), `bg` = the background's convergence (how far the whole
/// background sits into the screen). Pushing the background one way and the
/// square the other makes the background-vs-front gap obvious while each stays
/// small enough to fuse.
pub fn draw_rds(gl: &gl::Gl, gw: i32, gh: i32, eye: i32, depth: i32, bg: i32) {
    let tex = rds_texture(gl, gw, gh);
    if tex.is_none() {
        return;
    }

    // The scene draws over the whole framebuffer at z=0; the depth-test bit is
    // globally enabled (geometry scenes), so it is turned off here and restored
    // after, exactly like the original draw_rds did.
    gfx::disable(gl, gl::DEPTH_TEST);

    // Square sits in the centre of the screen.
    let (cx, cy) = (gw / 2, gh / 2);

    // A positive `sh` slides field content LEFT on screen (framebuffer column
    // `x` samples texel `x + sh`). To make the square fuse IN FRONT of the
    // screen its image must be crossed: shifted RIGHT for the LEFT eye and
    // LEFT for the RIGHT eye. The background gets the mirrored shifts so it
    // fuses INTO the screen, on the far side of the square (the whole field
    // still flickers uniformly, which is what hides the square without
    // glasses).
    let sh_bg = if eye == 1 { bg } else { -bg };
    let sh_sq = if eye == 1 { -depth } else { depth };

    // Background: the whole field shifted by the convergence, then the central
    // square re-drawn with its own extra offset on top.
    let mut verts: Vec<UvVert> = Vec::with_capacity(12);
    push_shifted_quad(&mut verts, gw, gh, 0, 0, gw, gh, sh_bg);
    push_shifted_quad(
        &mut verts,
        gw,
        gh,
        cx - SQUARE_HALF,
        cy - SQUARE_HALF,
        cx + SQUARE_HALF,
        cy + SQUARE_HALF,
        sh_sq,
    );

    // 2D orthographic projection covering the whole framebuffer (the same
    // 0..gw / 0..gh box `glOrtho` set up before).
    let mvp = Mat4::orthographic_rh_gl(0.0, gw as f32, 0.0, gh as f32, -1.0, 1.0);

    RDS_QUAD.with(|q| {
        let mut q = q.borrow_mut();
        match q.as_mut() {
            Some(m) => m.replace_uv_verts(gl, &verts),
            None => *q = Some(Mesh::from_uv_verts(gl, &verts, gl::TRIANGLES)),
        };
        if let Some(m) = q.as_ref() {
            tex.draw(gl, m, mvp, gfx::WHITE);
        }
    });
    gfx::enable(gl, gl::DEPTH_TEST);
}