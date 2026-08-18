//! Paul Bourke's "pulsar" scene, ported from `3dvgl-c/src/scene.cpp` (merged
//! as scene mode "3"). `glutSolidSphere` (a GLUT helper) is replaced by an
//! equivalent latitude/longitude tessellation so the Rust demo has no GLUT
//! dependency.
//!
//! http://paulbourke.net/miscellaneous/stereographics/stereorender/
//!
//! The geometry is completely static - only the top-level spin angle changes
//! from frame to frame, and that is a matrix rotation applied outside the
//! scene. The quads/cones/field-lines are therefore compiled into a display
//! list once and replayed with a new spin matrix each frame. This removes the
//! ~15 000 sin/cos vertices of per-frame tessellation that pushed a debug build
//! over one 120 Hz vblank (8333 us) and desynced the shutters.

use crate::gl;
use std::cell::RefCell;

const DTOR: f64 = 0.0174532925;

thread_local! {
    // Display list holding the static geometry, built once per GL context.
    static PULSAR_LIST: RefCell<Option<u32>> = const { RefCell::new(None) };
}

/// Draws a solid shaded sphere built from quads with per-vertex normals,
/// standing in for GLUT's `glutSolidSphere`.
fn sphere(gl: &gl::Gl, radius: f64, slices: i32, stacks: i32) {
    let (sl, st) = (slices as f64, stacks as f64);
    let two_pi = 2.0 * std::f64::consts::PI;
    for i in 0..stacks {
        let u0 = i as f64 * two_pi / st;
        let u1 = (i as f64 + 1.0) * two_pi / st;
        gl.begin(gl::QUADS);
        for j in 0..slices {
            let v0 = j as f64 * std::f64::consts::PI / sl;
            let v1 = (j as f64 + 1.0) * std::f64::consts::PI / sl;
            let p0 = [
                radius * v0.sin() * u0.cos(),
                radius * v0.cos(),
                radius * v0.sin() * u0.sin(),
            ];
            let p1 = [
                radius * v0.sin() * u1.cos(),
                radius * v0.cos(),
                radius * v0.sin() * u1.sin(),
            ];
            let p2 = [
                radius * v1.sin() * u1.cos(),
                radius * v1.cos(),
                radius * v1.sin() * u1.sin(),
            ];
            let p3 = [
                radius * v1.sin() * u0.cos(),
                radius * v1.cos(),
                radius * v1.sin() * u0.sin(),
            ];
            for p in [p0, p1, p2, p3] {
                let n = [p[0] / radius, p[1] / radius, p[2] / radius];
                gl.normal3f(n[0] as f32, n[1] as f32, n[2] as f32);
                gl.vertex3f(p[0] as f32, p[1] as f32, p[2] as f32);
            }
        }
        gl.end();
    }
}

/// Emits every static primitive (the sphere, lat/lon "center", cones and field
/// lines, under the fixed 45-degree tilt). Runs exactly once, inside the
/// display list. `make_geometry` applies the per-frame spin matrix around it.
fn draw_static(gl: &gl::Gl) {
    let cradius = 5.3; // Final radius of the cone
    let clength = 30.0; // Cone length
    let sradius = 10.0; // Final radius of sphere
    let r1 = 12.0; // Min radius of field lines
    let r2 = 16.0; // Max radius of field lines

    let specular = [1.0f32, 1.0, 1.0, 1.0];
    let shiny = [5.0f32];

    gl.materialfv(gl::FRONT_AND_BACK, gl::SPECULAR, &specular);
    gl.materialfv(gl::FRONT_AND_BACK, gl::SHININESS, &shiny);

    // Rotation about spin axis (fixed 45-degree tilt; the spin itself is
    // applied by the caller around this display list).
    gl.push_matrix();
    gl.rotatef(45.0, 0.0, 0.0, 1.0);

    // Light in center.
    gl.color3f(1.0, 1.0, 1.0);
    sphere(gl, 5.0, 16, 8);

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

            gl.begin(gl::POLYGON);
            if i % 20 == 0 {
                gl.color3f(1.0, 0.0, 0.0);
            } else {
                gl.color3f(0.5, 0.0, 0.0);
            }
            for p in [p0, p1, p2, p3] {
                gl.normal3f(p[0] as f32, p[1] as f32, p[2] as f32);
                gl.vertex3f(p[0] as f32, p[1] as f32, p[2] as f32);
            }
            gl.end();
        }
    }

    // Draw the cones.
    for j in [-1.0f64, 1.0] {
        for i in (0..360).step_by(10) {
            let (if_, if5) = (i as f64 * DTOR, (i as f64 + 10.0) * DTOR);

            let p0 = [0.0f64, 0.0, 0.0];
            let p1 = [cradius * if_.cos(), j * clength, cradius * if_.sin()];
            let p2 = [cradius * if5.cos(), j * clength, cradius * if5.sin()];

            let n0 = [p0[0], -1.0, p0[2]];
            let n1 = [p1[0], 0.0, p1[2]];
            let n2 = [p2[0], 0.0, p2[2]];

            gl.begin(gl::POLYGON);
            if i % 30 == 0 {
                gl.color3f(0.0, 0.2, 0.0);
            } else {
                gl.color3f(0.0, 0.5, 0.0);
            }
            for k in 0..3 {
                let p = [p0, p1, p2][k];
                let n = [n0, n1, n2][k];
                gl.normal3f(n[0] as f32, n[1] as f32, n[2] as f32);
                gl.vertex3f(p[0] as f32, p[1] as f32, p[2] as f32);
            }
            gl.end();
        }
    }

    // Draw the field lines.
    for j in (0..360).step_by(20) {
        gl.push_matrix();
        gl.rotatef(j as f32, 0.0, 1.0, 0.0);
        gl.begin(gl::LINE_STRIP);
        gl.color3f(0.7, 0.7, 0.7);
        for i in -140..140 {
            let x = r1 + r1 * (i as f64 * DTOR).cos();
            let y = r2 * (i as f64 * DTOR).sin();
            gl.vertex3f(x as f32, y as f32, 0.0);
        }
        gl.end();
        gl.pop_matrix();
    }

    gl.pop_matrix(); // Pulsar axis rotation.
}

/// Returns the (cached) display-list id, compiling the static geometry the
/// first time it is called.
fn build_list(gl: &gl::Gl) -> u32 {
    PULSAR_LIST.with(|c| {
        let mut c = c.borrow_mut();
        match *c {
            Some(id) => id,
            None => {
                let id = gl.gen_lists(1);
                if id != 0 {
                    gl.new_list(id, gl::COMPILE);
                    draw_static(gl);
                    gl.end_list();
                }
                *c = Some(id);
                id
            }
        }
    })
}

/// Compiles the static display list up front (no drawing). Building it lazily
/// on scene switch would stall the KMS flip clock for tens of ms (15 k sin/cos
/// vertices in a debug build) and permanently de-phase the VT shutters.
pub fn warm(gl: &gl::Gl) {
    let _ = build_list(gl);
}

/// Create the geometry for the pulsar. The static scene is baked into a
/// display list on first use; each frame only the spin matrix is applied
/// around a single `glCallList`.
pub fn make_geometry(gl: &gl::Gl, rotateangle: f32) {
    let id = build_list(gl);
    if id == 0 {
        return;
    }

    // Top-level spin - the only per-frame-varying transform.
    gl.push_matrix();
    gl.rotatef(rotateangle, 0.0, 1.0, 0.0);
    gl.call_list(id);
    gl.pop_matrix();
}

/// Set up the lighting environment.
pub fn make_lighting(gl: &gl::Gl) {
    let fullambient = [1.0f32, 1.0, 1.0, 1.0];
    let position = [0.0f32, 0.0, 0.0, 0.0];
    let ambient = [0.2f32, 0.2, 0.2, 1.0];
    let diffuse = [1.0f32, 1.0, 1.0, 1.0];
    let specular = [0.0f32, 0.0, 0.0, 1.0];

    // Turn off all the lights.
    for light in [
        gl::LIGHT0,
        gl::LIGHT1,
        gl::LIGHT2,
        gl::LIGHT3,
        gl::LIGHT4,
        gl::LIGHT5,
        gl::LIGHT6,
        gl::LIGHT7,
    ] {
        gl.disable(light);
    }
    gl.light_modeli(gl::LIGHT_MODEL_LOCAL_VIEWER, gl::TRUE as i32);
    gl.light_modeli(gl::LIGHT_MODEL_TWO_SIDE, gl::FALSE as i32);

    // Turn on the appropriate lights.
    gl.light_modelfv(gl::LIGHT_MODEL_AMBIENT, &fullambient);
    gl.lightfv(gl::LIGHT0, gl::POSITION, &position);
    gl.lightfv(gl::LIGHT0, gl::AMBIENT, &ambient);
    gl.lightfv(gl::LIGHT0, gl::DIFFUSE, &diffuse);
    gl.lightfv(gl::LIGHT0, gl::SPECULAR, &specular);
    gl.enable(gl::LIGHT0);

    // Sort out the shading algorithm.
    gl.shade_model(gl::SMOOTH);

    // Turn lighting on.
    gl.enable(gl::LIGHTING);
}