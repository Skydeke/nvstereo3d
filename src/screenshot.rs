//! Frame buffer screenshots in 24-bit uncompressed TGA format.
//!
//! Port of `src/screenshot.cpp` from the original C project, reimplemented on
//! the `glow` API.

use crate::gl::{self, HasContext};
use std::io::Write;

/// Configures OpenGL pixel-store state so screenshots read correctly.
pub fn init(gl: &gl::Gl) {
    unsafe {
        gl.pixel_store_i32(gl::PACK_ALIGNMENT, 1);
        gl.pixel_store_i32(gl::UNPACK_ALIGNMENT, 1);
    }
}

/// Reads the region `(x, y)` to `(x + w, y + h)` from the front buffer and
/// writes it to `filename` as a 24-bit uncompressed TGA file.
pub fn screenshot(gl: &gl::Gl, x: i32, y: i32, w: i32, h: i32, filename: &str) {
    use gl::PixelPackData;

    // Read from the front buffer.
    unsafe { gl.read_buffer(gl::FRONT) };

    // Grab the pixel data.
    let mut buffer = vec![0u8; (w * h * 3) as usize];
    unsafe {
        gl.read_pixels(x, y, w, h, gl::RGB, gl::UNSIGNED_BYTE, PixelPackData::Slice(Some(&mut buffer)));
    }

    let mut file = match std::fs::File::create(filename) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Failed to open screenshot file for writing: {e}");
            std::process::exit(1);
        }
    };

    // 24-bit uncompressed targa header (thanks to Paul Bourke).
    file.write_all(&[
        0, // no color map
        0, // unused
        2, // type: uncompressed RGB
        0,
        0, // color map start
        0,
        0, // color map length
        0, // color map depth
        0,
        0, // x origin
        0,
        0, // y origin
        w as u8,
        (w >> 8) as u8, // width
        h as u8,
        (h >> 8) as u8, // height
        24,             // 24-bit color depth
        0,              // image descriptor
    ])
    .unwrap();

    // Write the image data in BGR order.
    for j in 0..h {
        for i in 0..w {
            let offset = ((i + j * w) * 3) as usize;
            let r = buffer[offset];
            let g = buffer[offset + 1];
            let b = buffer[offset + 2];
            file.write_all(&[b, g, r]).unwrap();
        }
    }
}