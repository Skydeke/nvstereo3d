//! Per-eye diagnostic pattern: the LEFT eye sees a grid of green hexagons,
//! the RIGHT eye a grid of blue triangles.  With this it is easy to verify
//! (a) which physical lens is driven by which eye packet and (b) whether
//! each lens sees only its own content (no ghosting / cross-talk).
//!
//! This became the default "1" scene for the merged 3dv3d demo.

use crate::gl;
use crate::stereo_helper::Camera;

/// Draws one regular hexagon (flat-top) at `(cx, cy)` in billboard space.
fn hexagon(gl: &gl::Gl, to_world: impl Fn(f32, f32) -> [f32; 3], cx: f32, cy: f32, r: f32) {
    gl.begin(gl::POLYGON);
    for k in 0..6 {
        let a = k as f64 * std::f64::consts::PI / 3.0;
        let (dx, dy) = (r * a.cos() as f32, r * a.sin() as f32);
        let p = to_world(cx + dx, cy + dy);
        gl.vertex3f(p[0], p[1], p[2]);
    }
    gl.end();
}

/// Draws one equilateral triangle (point-up) at `(cx, cy)` in billboard space.
fn triangle(gl: &gl::Gl, to_world: impl Fn(f32, f32) -> [f32; 3], cx: f32, cy: f32, r: f32) {
    gl.begin(gl::TRIANGLES);
    for k in 0..3 {
        let a = k as f64 * 2.0 * std::f64::consts::PI / 3.0 + std::f64::consts::PI / 2.0;
        let (dx, dy) = (r * a.cos() as f32, r * a.sin() as f32);
        let p = to_world(cx + dx, cy + dy);
        gl.vertex3f(p[0], p[1], p[2]);
    }
    gl.end();
}

/// Renders the diagnostic pattern for the effective eye (`show`):
/// `show == 1` (left) -> green hexagons, `show == 0` (right) -> blue triangles.
/// The pattern is billboarded in the plane facing the camera at the origin.
pub fn make_geometry(gl: &gl::Gl, cam: Camera, show: i32) {
    // Billboard basis at the origin, facing the camera.
    let dir = cam.look.sub(cam.eye).normalize();
    let right = dir.cross(cam.up).normalize();
    let up = right.cross(dir);
    let (rx, ry, rz) = (right.x, right.y, right.z);
    let (ux, uy, uz) = (up.x, up.y, up.z);

    // Flat colors, no lighting modulation.
    gl.disable(gl::LIGHTING);

    let to_world = |x: f32, y: f32| -> [f32; 3] {
        [x * rx + y * ux, x * ry + y * uy, x * rz + y * uz]
    };

    let cols = 5i32;
    let rows = 4i32;
    let spacing = 6.5f32;
    let radius = 2.4f32;

    if show == 1 {
        // Left eye: green hexagons.
        gl.color3f(0.0, 0.9, 0.1);
        for row in 0..rows {
            for col in 0..cols {
                let cx = ((col - cols / 2) as f32) * spacing;
                let cy = ((row - rows / 2) as f32) * spacing;
                hexagon(gl, &to_world, cx, cy, radius);
            }
        }
    } else {
        // Right eye: blue triangles.
        gl.color3f(0.2, 0.4, 1.0);
        for row in 0..rows {
            for col in 0..cols {
                let cx = ((col - cols / 2) as f32) * spacing;
                let cy = ((row - rows / 2) as f32) * spacing;
                triangle(gl, &to_world, cx, cy, radius);
            }
        }
    }

    gl.enable(gl::LIGHTING);
}

/// Sets up the lighting environment.
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