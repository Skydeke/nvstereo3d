//! Minimal 5x7 bitmap-font text overlay for the fixed-function-GL demo.
//!
//! Renders a single line of text into an RGBA buffer and blits it with
//! `glDrawPixels`, using alpha blending so only the glyph pixels touch the
//! framebuffer. Currently used by the hexagon/triangle scene to label which
//! eye should see which pattern.

use std::ffi::c_void;

use crate::gl;

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
        _ => return None,
    })
}

/// Unscaled character cell advance (glyph 5 wide + 1 column of spacing).
const CELL_W: i32 = 6;
/// Unscaled glyph height.
const CELL_H: i32 = 7;

/// Draws one line of text blended over the framebuffer.
///
/// `(x, y)` is the bottom-left corner in window pixels (the origin agrees
/// with `glDrawPixels`). `scale` multiplies the 5x7 font. `r,g,b` set the
/// glyph colour; the background inside the text's box stays untouched
/// because the glyphs are alpha-blended.
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
    let (w, h) = (n * CELL_W * scale, CELL_H * scale);
    if n == 0 || w == 0 || h == 0 {
        return;
    }

    // Build one RGBA image for the whole line. `glDrawPixels` treats the
    // FIRST memory row as the BOTTOM scanline, so glyph row 0 (the visual
    // top of the letter) goes in the LAST memory row.
    let mut buf = vec![0u8; (w * h * 4) as usize];
    let (cr, cg, cb) = (
        (r * 255.0).round().clamp(0.0, 255.0) as u8,
        (g * 255.0).round().clamp(0.0, 255.0) as u8,
        (b * 255.0).round().clamp(0.0, 255.0) as u8,
    );
    let mut gx = 0i32;
    for ch in text.chars() {
        if let Some(glyph) = glyph(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if bits & (0x10 >> col) == 0 {
                        continue;
                    }
                    for sy in 0..scale {
                        let oy = (CELL_H - 1 - row as i32) * scale + sy;
                        for sx in 0..scale {
                            let ox = gx + col * scale + sx;
                            let i = ((oy * w + ox) * 4) as usize;
                            buf[i..i + 4].copy_from_slice(&[cr, cg, cb, 255]);
                        }
                    }
                }
            }
        }
        gx += CELL_W * scale;
    }

    // Screen-space 2D projection mapping window pixels 1:1.
    gl.matrix_mode(gl::PROJECTION);
    gl.load_identity();
    gl.ortho(0.0, f64::from(gw), 0.0, f64::from(gh), -1.0, 1.0);
    gl.matrix_mode(gl::MODELVIEW);
    gl.load_identity();

    // Flat, depth-less, alpha-blended glyphs over the rendered scene.
    gl.disable(gl::LIGHTING);
    gl.disable(gl::DEPTH_TEST);
    gl.enable(gl::BLEND);
    gl.blend_func(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
    gl.pixel_storei(gl::UNPACK_ALIGNMENT, 1);

    gl.raster_pos2i(x, y);
    gl.draw_pixels(
        w,
        h,
        gl::RGBA,
        gl::UNSIGNED_BYTE,
        buf.as_ptr() as *const c_void,
    );

    gl.disable(gl::BLEND);
    gl.enable(gl::DEPTH_TEST);
}