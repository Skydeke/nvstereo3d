//! 3dvgl diagnostic pattern: the RIGHT eye sees blue triangles, the LEFT eye
//! green (flat-top) hexagons, drawn on a billboard facing the camera.
//!
//! Port of `src/scene.cpp` from the original C project, built for the shader
//! pipeline (see [`crate::gfx`]): the immediate-mode display lists become two
//! vertex-buffer meshes compiled once from the same world-space positions, so
//! the rendered pattern is identical. The billboard only depends on
//! `eye`/`look`/`up` - all fixed at init in the demo - so the cached meshes
//! stay valid for the whole run.

use crate::gl;
use crate::gfx::{ColorVert, Mesh};
use crate::stereo_helper::Camera;
use glam::Mat4;
use std::cell::RefCell;

thread_local! {
    /// Meshes: `[0]` = right eye (triangles), `[1]` = left eye (hexagons).
    /// Built once per GL context.
    static HEX_TRI_MESHES: RefCell<Option<[Mesh; 2]>> = const { RefCell::new(None) };
}

const COLS: i32 = 5;
const ROWS: i32 = 4;
const SPACING: f32 = 6.5;
const RADIUS: f32 = 2.4;

/// Billboards one flat-top regular hexagon at `(cx, cy)`.
fn push_hexagon(verts: &mut Vec<ColorVert>, color: [f32; 3], to_world: &dyn Fn(f32, f32) -> [f32; 3], cx: f32, cy: f32, r: f32) {
    let center = to_world(cx, cy);
    for k in 0..6 {
        let a0 = k as f64 * std::f64::consts::PI / 3.0;
        let a1 = (k as f64 + 1.0) * std::f64::consts::PI / 3.0;
        let p0 = to_world(cx + r * a0.cos() as f32, cy + r * a0.sin() as f32);
        let p1 = to_world(cx + r * a1.cos() as f32, cy + r * a1.sin() as f32);
        verts.push(ColorVert { pos: center, color });
        verts.push(ColorVert { pos: p0, color });
        verts.push(ColorVert { pos: p1, color });
    }
}

/// Billboards one point-up triangle at `(cx, cy)`.
fn push_triangle(verts: &mut Vec<ColorVert>, color: [f32; 3], to_world: &dyn Fn(f32, f32) -> [f32; 3], cx: f32, cy: f32, r: f32) {
    for k in 0..3 {
        let a = k as f64 * 2.0 * std::f64::consts::PI / 3.0 + std::f64::consts::PI / 2.0;
        let (dx, dy) = (r * a.cos() as f32, r * a.sin() as f32);
        let p = to_world(cx + dx, cy + dy);
        verts.push(ColorVert { pos: p, color });
    }
}

/// Billboard world-space basis for the pattern plane at the origin, facing
/// the camera.
fn billboard_basis(cam: Camera) -> ([f32; 3], [f32; 3]) {
    let dir = (cam.look - cam.eye).normalize();
    let right = dir.cross(cam.up).normalize();
    let up = right.cross(dir);
    ([right.x, right.y, right.z], [up.x, up.y, up.z])
}

/// Builds the two per-eye meshes once. No-op once built; a mesh that failed
/// to allocate is kept as-is (empty) so it isn't retried every frame.
fn build_meshes(gl: &gl::Gl, cam: Camera) {
    HEX_TRI_MESHES.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_some() {
            return;
        }

        let (rx, ry, rz) = {
            let (r, _) = billboard_basis(cam);
            (r[0], r[1], r[2])
        };
        let (ux, uy, uz) = {
            let (_, u) = billboard_basis(cam);
            (u[0], u[1], u[2])
        };
        let to_world = |x: f32, y: f32| -> [f32; 3] {
            [x * rx + y * ux, x * ry + y * uy, x * rz + y * uz]
        };

        // Right eye (show == 0): blue triangles.
        let mut tris = Vec::with_capacity((COLS * ROWS * 3) as usize);
        for row in 0..ROWS {
            for col in 0..COLS {
                let cx = ((col - COLS / 2) as f32) * SPACING;
                let cy = ((row - ROWS / 2) as f32) * SPACING;
                push_triangle(&mut tris, [0.2, 0.4, 1.0], &to_world, cx, cy, RADIUS);
            }
        }
        let right = Mesh::from_color_verts(gl, &tris, gl::TRIANGLES);

        // Left eye (show == 1): green hexagons.
        let mut hexs = Vec::with_capacity((COLS * ROWS * 18) as usize);
        for row in 0..ROWS {
            for col in 0..COLS {
                let cx = ((col - COLS / 2) as f32) * SPACING;
                let cy = ((row - ROWS / 2) as f32) * SPACING;
                push_hexagon(&mut hexs, [0.0, 0.9, 0.1], &to_world, cx, cy, RADIUS);
            }
        }
        let left = Mesh::from_color_verts(gl, &hexs, gl::TRIANGLES);

        *c = Some([right, left]);
    });
}

/// Pre-compiles the meshes while the flip clock is quiet (no-op if already
/// built).  Call once at startup so a scene switch never stalls the 120 Hz
/// swap loop.
pub fn warm(gl: &gl::Gl, cam: Camera) {
    build_meshes(gl, cam);
}

/// Renders the diagnostic pattern for the effective eye (`show`):
/// `show == 1` (left) -> green hexagons, `show == 0` (right) -> blue
/// triangles. `mvp` is the projection·view matrix for that eye.
pub fn make_geometry(gl: &gl::Gl, cam: Camera, show: i32, mvp: Mat4) {
    build_meshes(gl, cam);
    HEX_TRI_MESHES.with(|c| {
        let c = c.borrow();
        if let Some(list) = c.as_ref() {
            let mesh = if show != 0 { &list[1] } else { &list[0] };
            mesh.draw_color(gl, mvp);
        }
    });
}