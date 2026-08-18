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
//! work and no pixel upload. That keeps every frame comfortably inside one
//! 120 Hz vblank (the old per-frame `glDrawPixels` of the whole 2560x1440 field
//! pushed the swap over the 8333 us budget and dropped the shutter sync).

use crate::gl;
use std::cell::RefCell;
use std::ffi::c_void;

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
    static RDS_TEX: RefCell<Option<(i32, i32, u32)>> = const { RefCell::new(None) };
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
fn rds_texture(gl: &gl::Gl, gw: i32, gh: i32) -> u32 {
    let mut out = 0u32;
    RDS_TEX.with(|c| {
        let mut c = c.borrow_mut();
        let rebuild = match &*c {
            Some((w, h, id)) => *w != gw || *h != gh || *id == 0,
            None => true,
        };
        if rebuild {
            if let Some((_, _, id)) = c.as_ref() {
                if *id != 0 {
                    let mut arr = [*id];
                    gl.delete_textures(1, &mut arr);
                }
            }
            let mut id = 0u32;
            gl.gen_textures(1, std::slice::from_mut(&mut id));
            if id != 0 {
                let base = base_noise(gw, gh);
                gl.bind_texture(gl::TEXTURE_2D, id);
                gl.pixel_storei(gl::UNPACK_ALIGNMENT, 1);
                gl.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
                gl.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
                gl.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
                gl.tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);
                gl.tex_image_2d(
                    gl::TEXTURE_2D,
                    0,
                    gl::LUMINANCE as i32,
                    gw,
                    gh,
                    0,
                    gl::LUMINANCE,
                    gl::UNSIGNED_BYTE,
                    base.as_ptr() as *const c_void,
                );
            }
            *c = Some((gw, gh, id));
        }
        if let Some((_, _, id)) = c.as_ref() {
            out = *id;
        }
    });
    out
}

/// Draws one full-screen textured quad sampling the base field at a horizontal
/// offset of `sh` pixels (wrapping), i.e. framebuffer column `x` shows texel
/// `x + sh` - a pure left/right shift of the random field.
fn draw_shifted_quad(
    gl: &gl::Gl,
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

    gl.begin(gl::QUADS);
    gl.tex_coord2f(s0, v0);
    gl.vertex2f(x0 as f32, y0 as f32);
    gl.tex_coord2f(s0, v1);
    gl.vertex2f(x0 as f32, y1 as f32);
    gl.tex_coord2f(s1, v1);
    gl.vertex2f(x1 as f32, y1 as f32);
    gl.tex_coord2f(s1, v0);
    gl.vertex2f(x1 as f32, y0 as f32);
    gl.end();
}

/// Compiles the base-field texture up front (no drawing). Building it lazily
/// on scene switch would stall the KMS render loop for ~100 ms in a debug
/// build (3.7 M dot RNG), which permanently de-phases the strict VT flip clock
/// (the shutters keep missing a vblank every ~9th frame). Warms before the
/// loop so a switch to scene 3 is instant.
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
    let id = rds_texture(gl, gw, gh);
    if id == 0 {
        return;
    }

    // Square sits in the centre of the screen.
    let (cx, cy) = (gw / 2, gh / 2);

    // Whole field gets the small mirrored background shift (into the screen)
    // so it flickers uniformly - hiding the square without glasses - and the
    // square pops out on the near side.
    let sh_bg = if eye == 1 { -bg } else { bg };
    let sh_sq = if eye == 1 { depth } else { -depth };

    // 2D orthographic projection covering the whole framebuffer.
    gl.matrix_mode(gl::PROJECTION);
    gl.load_identity();
    gl.ortho(0.0, gw as f64, 0.0, gh as f64, -1.0, 1.0);
    gl.matrix_mode(gl::MODELVIEW);
    gl.load_identity();

    gl.disable(gl::LIGHTING);
    gl.disable(gl::DEPTH_TEST);
    gl.tex_envi(gl::TEXTURE_ENV, gl::TEXTURE_ENV_MODE, gl::REPLACE as i32);
    gl.bind_texture(gl::TEXTURE_2D, id);
    gl.enable(gl::TEXTURE_2D);

    // Background: the whole field shifted by the convergence, then the central
    // square re-drawn with its own extra offset on top.
    draw_shifted_quad(gl, gw, gh, 0, 0, gw, gh, sh_bg);
    draw_shifted_quad(
        gl,
        gw,
        gh,
        cx - SQUARE_HALF,
        cy - SQUARE_HALF,
        cx + SQUARE_HALF,
        cy + SQUARE_HALF,
        sh_sq,
    );

    gl.disable(gl::TEXTURE_2D);
    gl.enable(gl::DEPTH_TEST);
}