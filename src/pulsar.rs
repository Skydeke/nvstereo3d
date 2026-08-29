//! Paul Bourke's "pulsar" scene, ported from `3dvgl-c/src/scene.cpp` (merged
//! as scene mode "4"). `glutSolidSphere` (a GLUT helper) is replaced by an
//! equivalent latitude/longitude tessellation so the Rust demo has no GLUT
//! dependency.
//!
//! http://paulbourke.net/miscellaneous/stereographics/stereorender/
//!
//! The geometry is completely static - only the top-level spin angle changes
//! from frame to frame, and that is a matrix rotation applied outside the
//! scene. The quads/cones/field-lines are therefore baked once into vertex
//! buffers (previously a display list) and replayed with a new spin matrix
//! each frame. This removes the ~15 000 sin/cos vertices of per-frame
//! tessellation that pushed a debug build over one 120 Hz vblank (8333 us)
//! and desynced the shutters.
//!
//! Flat colours replace the original fixed-function lighting: the scene's
//! `LIGHT0` is a directional light of zero direction (so diffuse and specular
//! contribute nothing) over a full `LIGHT_MODEL_AMBIENT`, which reduces
//! exactly to the current `glColor`, so the rendered result is unchanged.

use crate::gl;
use crate::gfx::{ColorVert, Mesh};
use glam::{Mat4, Vec3};
use std::cell::RefCell;

const DTOR: f64 = 0.0174532925;

thread_local! {
    /// (solid geometry, field lines) meshes, built once per GL context.
    static PULSAR_MESHES: RefCell<Option<(Mesh, Mesh)>> = const { RefCell::new(None) };
}

/// The fixed 45-degree tilt about the Z axis baked into every static vertex
/// (was `glRotatef(45, 0, 0, 1)` outside the display list).
fn tilt(p: Vec3) -> Vec3 {
    let c = 0.7071067811865476f32;
    let s = 0.7071067811865476f32;
    Vec3::new(c * p.x - s * p.y, s * p.x + c * p.y, p.z)
}

/// Rotation about the Y axis by `deg` degrees (was `glRotatef(deg, 0, 1, 0)`).
fn rot_y(p: Vec3, deg: f64) -> Vec3 {
    let a = deg * DTOR;
    let (c, s) = (a.cos() as f32, a.sin() as f32);
    Vec3::new(c * p.x + s * p.z, p.y, -s * p.x + c * p.z)
}

/// Emits one tilted quad as two triangles (`glBegin(GL_QUADS)` / `GL_POLYGON`
/// in the original).
fn push_quad(verts: &mut Vec<ColorVert>, q: [Vec3; 4], color: [f32; 3]) {
    for p in [q[0], q[1], q[2], q[0], q[2], q[3]] {
        let w = tilt(p);
        verts.push(ColorVert {
            pos: w.to_array(),
            color,
        });
    }
}

/// Emits one tilted triangle.
fn push_tri(verts: &mut Vec<ColorVert>, q: [Vec3; 3], color: [f32; 3]) {
    for p in q {
        let w = tilt(p);
        verts.push(ColorVert {
            pos: w.to_array(),
            color,
        });
    }
}

/// Bakes every static primitive (the sphere, lat/lon "center", cones and field
/// lines) into the two meshes. `make_geometry` applies the per-frame spin
/// matrix around them.
fn build_meshes(gl: &gl::Gl) {
    PULSAR_MESHES.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_some() {
            return;
        }

        let cradius = 5.3; // Final radius of the cone
        let clength = 30.0; // Cone length
        let sradius = 10.0; // Final radius of sphere
        let r1 = 12.0; // Min radius of field lines
        let r2 = 16.0; // Max radius of field lines

        let mut solid: Vec<ColorVert> = Vec::new();
        let mut field: Vec<ColorVert> = Vec::new();
        let mut field_ranges: Vec<(i32, i32)> = Vec::new();

        // Light in center (white sphere).
        let (sl, st) = (16.0f64, 8.0f64);
        let two_pi = 2.0 * std::f64::consts::PI;
        for i in 0..st as usize {
            let u0 = i as f64 * two_pi / st;
            let u1 = (i as f64 + 1.0) * two_pi / st;
            for j in 0..sl as usize {
                let v0 = j as f64 * std::f64::consts::PI / sl;
                let v1 = (j as f64 + 1.0) * std::f64::consts::PI / sl;
                let p0 = [
                    radius5(v0.sin() * u0.cos()),
                    radius5(v0.cos()),
                    radius5(v0.sin() * u0.sin()),
                ];
                let p1 = [
                    radius5(v0.sin() * u1.cos()),
                    radius5(v0.cos()),
                    radius5(v0.sin() * u1.sin()),
                ];
                let p2 = [
                    radius5(v1.sin() * u1.cos()),
                    radius5(v1.cos()),
                    radius5(v1.sin() * u1.sin()),
                ];
                let p3 = [
                    radius5(v1.sin() * u0.cos()),
                    radius5(v1.cos()),
                    radius5(v1.sin() * u0.sin()),
                ];
                push_quad(
                    &mut solid,
                    [
                        vec3_of(p0),
                        vec3_of(p1),
                        vec3_of(p2),
                        vec3_of(p3),
                    ],
                    [1.0, 1.0, 1.0],
                );
            }
        }

        // Spherical center.
        for i in (0..360).step_by(5) {
            let (if_, if5) = (i as f64 * DTOR, (i as f64 + 5.0) * DTOR);
            for j in (-80..80).step_by(5) {
                let (jf, jf5) = (j as f64 * DTOR, (j as f64 + 5.0) * DTOR);

                let p0 = [
                    sradius * jf.cos() * if_.cos(),
                    sradius * jf.sin(),
                    sradius * jf.cos() * if_.sin(),
                ];
                let p1 = [
                    sradius * jf5.cos() * if_.cos(),
                    sradius * jf5.sin(),
                    sradius * jf5.cos() * if_.sin(),
                ];
                let p2 = [
                    sradius * jf5.cos() * if5.cos(),
                    sradius * jf5.sin(),
                    sradius * jf5.cos() * if5.sin(),
                ];
                let p3 = [
                    sradius * jf.cos() * if5.cos(),
                    sradius * jf.sin(),
                    sradius * jf.cos() * if5.sin(),
                ];

                let color = if i % 20 == 0 { [1.0, 0.0, 0.0] } else { [0.5, 0.0, 0.0] };
                push_quad(
                    &mut solid,
                    [
                        vec3_of(p0),
                        vec3_of(p1),
                        vec3_of(p2),
                        vec3_of(p3),
                    ],
                    color,
                );
            }
        }

        // Draw the cones.
        for j in [-1.0f64, 1.0] {
            for i in (0..360).step_by(10) {
                let (if_, if5) = (i as f64 * DTOR, (i as f64 + 10.0) * DTOR);

                let p0 = [0.0f64, 0.0, 0.0];
                let p1 = [cradius * if_.cos(), j * clength, cradius * if_.sin()];
                let p2 = [cradius * if5.cos(), j * clength, cradius * if5.sin()];

                let color = if i % 30 == 0 { [0.0, 0.2, 0.0] } else { [0.0, 0.5, 0.0] };
                push_tri(
                    &mut solid,
                    [vec3_of(p0), vec3_of(p1), vec3_of(p2)],
                    color,
                );
            }
        }

        // Draw the field lines (each strip its own primitive range, sharing
        // one VBO so the whole static scene stays a rebuild-free upload).
        for j in (0..360).step_by(20) {
            let start = field.len() as i32;
            for i in -140..140 {
                let x = r1 + r1 * (i as f64 * DTOR).cos();
                let y = r2 * (i as f64 * DTOR).sin();
                let p = rot_y(Vec3::new(x as f32, y as f32, 0.0), j as f64);
                let w = tilt(p);
                field.push(ColorVert {
                    pos: w.to_array(),
                    color: [0.7, 0.7, 0.7],
                });
            }
            field_ranges.push((start, field.len() as i32 - start));
        }

        let solid_mesh = Mesh::from_color_verts(gl, &solid, gl::TRIANGLES);
        let field_mesh = Mesh::from_color_verts_ranges(gl, &field, &field_ranges, gl::LINE_STRIP);
        *c = Some((solid_mesh, field_mesh));
    });
}

#[inline]
fn vec3_of(p: [f64; 3]) -> Vec3 {
    Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)
}

#[inline]
fn radius5(x: f64) -> f64 {
    // The light sphere radius (5.0) is hardcoded in the original (`sphere(gl,
    // 5.0, 16, 8)`), spelled out here to avoid a magic number in the loop.
    5.0 * x
}

/// Bakes the static meshes up front (no drawing). Building them lazily on a
/// scene switch would stall the swap loop for tens of ms (~15 k sin/cos
/// vertices in a debug build) and permanently de-phase the shutters.
pub fn warm(gl: &gl::Gl) {
    build_meshes(gl);
}

/// Draws the pulsar for the current frame. The static scene is baked into the
/// meshes on first use; each frame applies only the spin matrix.
pub fn make_geometry(gl: &gl::Gl, rotateangle: f32, mvp: Mat4) {
    build_meshes(gl);
    PULSAR_MESHES.with(|c| {
        let c = c.borrow();
        if let Some((solid, field)) = c.as_ref() {
            // Top-level spin - the only per-frame-varying transform.
            let spin = Mat4::from_rotation_y(rotateangle.to_radians());
            let model = mvp * spin;
            solid.draw_color(gl, model);
            field.draw_color(gl, model);
        }
    });
}