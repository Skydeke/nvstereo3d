//! Minimal 5x7 bitmap-font text overlay for the demo.
//!
//! Renders a single line of text using a pre-built glyph atlas texture and
//! a single textured quad per `draw_text` call.  This replaces the old
//! per-line `glDrawPixels` path, which uploaded a fresh CPU RGBA buffer on
//! every call and could not sustain 120 Hz on NVIDIA's driver (the
//! `glDrawPixels` + alpha-blend path has no fast implementation in the
//! proprietary driver).  The quad is now drawn through the shared textured
//! shader (see [`crate::gfx`]) and the CPU rasteriser reuses a cached
//! buffer, so a text line still costs one small `glTexSubImage2D` and one
//! tiny VBO orphaning upload per frame.

use crate::gl;
use crate::gfx::{self, Mesh, Texture, UvVert};
use glam::Mat4;
use std::cell::RefCell;

/// One glyph: 7 rows of 5 columns, MSB (0x10) = the leftmost pixel.
fn glyph(c: char) -> Option<[u8; 7]> {
    Some(match c {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        ' ' => [0, 0, 0, 0, 0, 0, 0],
        ':' => [0, 0b00100, 0, 0, 0b00100, 0, 0],
        '-' => [0, 0, 0, 0b01110, 0, 0, 0],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        '.' => [0, 0, 0, 0, 0, 0b00100, 0b00100],
        '>' => [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        '+' => [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],
        '=' => [0, 0, 0b11111, 0, 0b11111, 0, 0],
        _ => return None,
    })
}

/// Unscaled character cell advance (glyph 5 wide + 1 column of spacing).
const CELL_W: i32 = 6;
/// Unscaled glyph height.
const CELL_H: i32 = 7;

/// Atlas texture width in pixels. Must be able to fit the widest `draw_text`
/// line at the largest scale: the longest HUD line is ~27 chars and the
/// longest scene label ("RIGHT: BLUE TRIANGLES") is 22 chars at scale up to
/// 6 -> 22 * 6 * 6 = 792 px. 1024 covers that with headroom.
const ATLAS_W: i32 = 1024;
/// Atlas texture height in pixels (must be >= CELL_H * max_scale).
const ATLAS_H: i32 = 128;

/// Pre-built GL texture atlas (white glyphs, alpha-only).
#[derive(Clone, Copy)]
struct Atlas {
    tex: Texture,
    /// Per-glyph width in atlas pixels (0 = missing glyph).
    widths: [i32; 128],
}

thread_local! {
    static ATLAS: RefCell<Option<Atlas>> = const { RefCell::new(None) };
    /// Reusable CPU-side RGBA buffer for text rasterisation.  Retained across
    /// frames to avoid a per-call allocation; resized on demand.
    static BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// Reusable textured-quad mesh; its vertices are re-uploaded (orphaned)
    /// per `draw_text` call so no GL object is created every frame.
    static QUAD: RefCell<Option<Mesh>> = const { RefCell::new(None) };
}

/// Creates the glyph atlas texture and builds the per-glyph origin/width
/// tables.  Must be called once after a GL context is current.
pub fn init(gl: &gl::Gl) {
    ATLAS.with(|a| {
        let mut a = a.borrow_mut();
        if a.is_some() {
            return;
        }

        // --- Build the atlas image (RGBA, white glyphs with alpha mask) ---
        let mut img = vec![0u8; (ATLAS_W * ATLAS_H * 4) as usize];
        let mut widths = [0i32; 128];
        let mut cur_x = 0i32;

        for ch_i in 0u8..128u8 {
            let ch = ch_i as char;
            if let Some(bits) = glyph(ch) {
                let gw = 5;
                widths[ch_i as usize] = gw;
                for (row, &bits_row) in bits.iter().enumerate() {
                    let rowi = row as i32;
                    for col in 0i32..5 {
                        if bits_row & (0x10u8 >> col as u32) == 0 {
                            continue;
                        }
                        // buffer row 0 = visual bottom (glyph row 6)
                        let buf_row = (6 - rowi) * ATLAS_W * 4;
                        let buf_col = (cur_x + col) * 4;
                        let i = (buf_row + buf_col) as usize;
                        img[i] = 255; // R
                        img[i + 1] = 255; // G
                        img[i + 2] = 255; // B
                        img[i + 3] = 255; // A
                    }
                }
                cur_x += gw;
            }
        }

        // --- Upload to GL once ---
        let tex = Texture::new(
            gl,
            ATLAS_W,
            ATLAS_H,
            gl::RGBA as i32,
            gl::RGBA,
            gl::UNSIGNED_BYTE,
            Some(&img),
            gl::NEAREST as i32,
            gl::CLAMP_TO_EDGE as i32,
        );

        *a = Some(Atlas { tex, widths });
    });
}

/// Draws one line of text blended over the rendered scene using a single
/// textured quad per call.
///
/// `(x, y)` is the bottom-left corner in window pixels. `scale` multiplies
/// the 5×7 font. `r,g,b` set the glyph colour (the atlas stores white glyphs
/// and the textured shader modulates them, exactly like `GL_MODULATE` did).
/// The background behind the text is left untouched.
pub fn draw_text(
    gl: &gl::Gl,
    gw: i32,
    gh: i32,
    text: &str,
    x: i32,
    y: i32,
    scale: i32,
    r: f32,
    g: f32,
    b: f32,
) {
    let n = text.chars().count() as i32;
    if n == 0 || scale <= 0 {
        return;
    }

    let text_w = n * CELL_W * scale;
    let text_h = CELL_H * scale;
    if text_w == 0 || text_h == 0 {
        return;
    }

    let atx = x.max(0);
    let aty = y.max(0);

    // The text is uploaded as a sub-image at atlas origin (0,0); refuse to
    // overflow the atlas (would silently mis-render or fail the upload).
    if text_w > ATLAS_W || text_h > ATLAS_H {
        return;
    }

    ATLAS.with(|atlas_ref| {
        let Some(atlas) = atlas_ref.borrow().as_ref().copied() else {
            return;
        };

        // --- CPU rasterise into a reusable RGBA buffer -------------------------
        BUF.with(|buf_ref| {
            let mut buf = buf_ref.borrow_mut();
            let size = (text_w * text_h * 4) as usize;
            if buf.len() < size {
                buf.resize(size, 0);
            }
            let buf = &mut *buf;
            buf[..size].fill(0);

            let mut gx = 0i32;
            for ch in text.chars() {
                let ci = ch as u32;
                if ci < 128 && atlas.widths[ci as usize] > 0 {
                    let bits = glyph(ch).unwrap();
                    let gw = atlas.widths[ci as usize];
                    for (row, &bits_row) in bits.iter().enumerate() {
                        let rowi = row as i32;
                        for col in 0..gw {
                            if bits_row & (0x10u8 >> col as u32) == 0 {
                                continue;
                            }
                            for sy in 0..scale {
                                let buf_row = (6 - rowi) * scale + sy;
                                for sx in 0..scale {
                                    let screen_x = gx + col * scale + sx;
                                    let i = ((buf_row * text_w + screen_x) * 4) as usize;
                                    buf[i] = 255;
                                    buf[i + 1] = 255;
                                    buf[i + 2] = 255;
                                    buf[i + 3] = 255;
                                }
                            }
                        }
                    }
                }
                gx += CELL_W * scale;
            }

            // --- Upload text to the atlas texture (sub-image) --------------------
            atlas.tex.sub_image(gl, 0, 0, text_w, text_h, gl::RGBA, gl::UNSIGNED_BYTE, &buf[..size]);
        });

        // --- Screen-space projection (same as old draw_text) ---------------------
        let mvp = Mat4::orthographic_rh_gl(0.0, gw as f32, 0.0, gh as f32, -1.0, 1.0);

        // The upload occupies the atlas's top-left `text_w × text_h` corner;
        // sample exactly that sub-rectangle (clamped, so the edges stay clean).
        let s1 = text_w as f32 / ATLAS_W as f32;
        let t1 = text_h as f32 / ATLAS_H as f32;

        // --- GL state for textured + colour-modulated quad -----------------------
        // Depth test is globally enabled (geometry scenes); the text quad sits at
        // z=0 and would be depth-culled by scene geometry, so it is turned off
        // for the overlay like the original draw_text did.
        gfx::disable(gl, gl::DEPTH_TEST);
        gfx::enable(gl, gl::BLEND);
        gfx::blend_func(gl, gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        atlas.tex.bind(gl);

        // --- Draw one textured quad covering the text area -----------------------
        // Memory row 0 (visual bottom of the text) sits at texture t=0; it
        // must land on the BOTTOM of the screen quad so the text reads
        // upright (glTexSubImage2D takes its first row as t=0, like
        // glDrawPixels takes it as the bottom scanline).  Emitted as an
        // explicit GL_QUADS-style 6-vertex triangulation so the whole quad
        // is covered (a 4-vertex strip/fan or a single TRIANGLES batch would
        // only paint part of it).
        let q0 = UvVert { pos: [atx as f32, aty as f32], uv: [0.0, 0.0] };
        let q1 = UvVert { pos: [atx as f32, (aty + text_h) as f32], uv: [0.0, t1] };
        let q2 = UvVert { pos: [(atx + text_w) as f32, (aty + text_h) as f32], uv: [s1, t1] };
        let q3 = UvVert { pos: [(atx + text_w) as f32, aty as f32], uv: [s1, 0.0] };
        let verts = [q0, q1, q2, q0, q2, q3];
        QUAD.with(|q| {
            let mut q = q.borrow_mut();
            match q.as_mut() {
                Some(m) => m.replace_uv_verts(gl, &verts),
                None => *q = Some(Mesh::from_uv_verts(gl, &verts, gl::TRIANGLES)),
            };
            if let Some(m) = q.as_ref() {
                m.draw_textured(gl, mvp, [r, g, b, 1.0]);
            }
        });
        gfx::Texture::unbind(gl);
        gfx::disable(gl, gl::BLEND);
        gfx::enable(gl, gl::DEPTH_TEST);
    });
}