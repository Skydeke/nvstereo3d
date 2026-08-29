//! Stereo camera helpers: vector math, camera description and the
//! projection/eye-position code, plus X11 refresh-rate detection.
//!
//! Port of `src/stereo_helper.h` from the original C project. The vector and
//! matrix math now come from `glam` (replacing the hand-rolled `Vec3` and the
//! GL matrix-stack replication of `gluPerspective`/`gluLookAt`); the matrices
//! are built exactly as the GLU originals did, so the eye projection is
//! bit-for-bit the same as before.

pub use glam::Vec3;

use crate::nvstusb::NvstusbContext;
use glam::Mat4;
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_void};

/// The different ways of projecting a stereo pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraType {
    ToeIn,
    ParallelAxisAsymmetric,
}

/// The camera used to generate stereo pairs. See the original header for
/// documentation of each field.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub camera_type: CameraType,
    pub eye: Vec3,
    pub look: Vec3,
    pub up: Vec3,
    pub focal: f32,
    pub fov: f32,
    pub iod: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            camera_type: CameraType::ToeIn,
            eye: Vec3::default(),
            look: Vec3::default(),
            up: Vec3::new(0.0, 1.0, 0.0),
            focal: 70.0,
            fov: 50.0,
            iod: 70.0 / 30.0,
            near: 1.0,
            far: 200.0,
        }
    }
}

// XF86VidModeModeLine from <X11/extensions/xf86vmode.h>. The pixel clock is
// returned separately by `XF86VidModeGetModeLine`; note the `hskew` field
// between `htotal` and `vdisplay`.
#[repr(C)]
struct XF86VidModeModeLine {
    hdisplay: u16,
    hsyncstart: u16,
    hsyncend: u16,
    htotal: u16,
    hskew: u16,
    vdisplay: u16,
    vsyncstart: u16,
    vsyncend: u16,
    vtotal: u16,
    flags: u32,
    privsize: c_int,
    private: *mut c_int,
}

type XOpenDisplayFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type XCloseDisplayFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type XDefaultScreenFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type XF86VidModeGetModeLineFn =
    unsafe extern "C" fn(*mut c_void, c_int, *mut c_int, *mut XF86VidModeModeLine) -> c_int;

/// Detects the monitor refresh rate and pushes it to the emitter.
///
/// On modern desktops `XF86VidModeGetModeLine` is frequently unavailable
/// (Wayland/Xwayland, composited X, no X server at all), which in the original
/// left `NvstusbContext::rate` at 0 and, worse, meant `set_rate` was never
/// called so the emitter's driver-enable register (0x1B) was never written and
/// no IR was ever emitted.  The emitter must be configured regardless, so this
/// tries, in order:
///
/// 1. `NVSTUSB_RATE` environment override (e.g. `NVSTUSB_RATE=120`);
/// 2. `preferred_mhz`, the wl_output mode rate of the monitor the window is
///    on (passed by the app on Wayland). This - not XF86VidMode - is the rate
///    that must be configured when several displays with different refresh
///    rates are attached: XWayland reports the X screen's global rate, which
///    may be the OTHER monitor's;
/// 3. the X11 `XF86VidMode` mode line;
/// 4. a 120 Hz default (with a warning).
///
/// Every path ends in `ctx.set_rate(...)`, guaranteeing the emitter gets its
/// driver-enable and a sane refresh rate even on headless / Wayland systems.
pub fn config_refresh_rate(ctx: &mut NvstusbContext, preferred_mhz: Option<u32>) {
    // Env override wins outright (handles headless, unknown display, etc.).
    if let Some(raw) = std::env::var_os("NVSTUSB_RATE") {
        if let Ok(s) = raw.into_string() {
            if let Ok(v) = s.parse::<f32>() {
                if v > 60.0 {
                    println!("Using NVSTUSB_RATE override: {v} Hz.");
                    ctx.set_rate(v);
                    return;
                }
                eprintln!("stereo_helper: ignoring invalid NVSTUSB_RATE={v} (must be > 60)");
            }
        }
    }

    // Per-monitor Wayland/winit rate of the output actually showing the demo.
    if let Some(mhz) = preferred_mhz {
        if mhz >= 60_000 {
            let f = mhz as f64 / 1000.0;
            println!("Using monitor refresh rate of {f:.6} Hz.");
            ctx.set_rate(f as f32);
            return;
        }
        eprintln!(
            "stereo_helper: ignoring implausible monitor refresh {} mHz; probing X11",
            mhz
        );
    }

    let x11 = match unsafe { Library::new("libX11.so.6") } {
        Ok(lib) => lib,
        Err(_) => {
            eprintln!("stereo_helper: could not load libX11.so.6; defaulting to 120 Hz");
            ctx.set_rate(120.0);
            return;
        }
    };
    let xf86vm = match unsafe { Library::new("libXxf86vm.so.1") } {
        Ok(lib) => lib,
        Err(_) => {
            eprintln!("stereo_helper: could not load libXxf86vm.so.1; defaulting to 120 Hz");
            ctx.set_rate(120.0);
            return;
        }
    };

    unsafe {
        let x_open_display: Symbol<XOpenDisplayFn> = match x11.get(b"XOpenDisplay\0") {
            Ok(s) => s,
            Err(_) => {
                eprintln!("stereo_helper: could not load XOpenDisplay; defaulting to 120 Hz");
                ctx.set_rate(120.0);
                return;
            }
        };
        let x_close_display: Symbol<XCloseDisplayFn> = match x11.get(b"XCloseDisplay\0") {
            Ok(s) => s,
            Err(_) => {
                eprintln!("stereo_helper: could not load XCloseDisplay; defaulting to 120 Hz");
                ctx.set_rate(120.0);
                return;
            }
        };
        let x_default_screen: Symbol<XDefaultScreenFn> = match x11.get(b"XDefaultScreen\0") {
            Ok(s) => s,
            Err(_) => {
                eprintln!("stereo_helper: could not load XDefaultScreen; defaulting to 120 Hz");
                ctx.set_rate(120.0);
                return;
            }
        };
        let get_mode_line: Symbol<XF86VidModeGetModeLineFn> =
            match xf86vm.get(b"XF86VidModeGetModeLine\0") {
                Ok(s) => s,
                Err(_) => {
                    eprintln!("stereo_helper: could not load XF86VidModeGetModeLine; defaulting to 120 Hz");
                    ctx.set_rate(120.0);
                    return;
                }
            };

        let display = x_open_display(std::ptr::null());
        if display.is_null() {
            eprintln!("stereo_helper: no X display available; defaulting to 120 Hz");
            ctx.set_rate(120.0);
            return;
        }

        let screen = x_default_screen(display);
        let mut pixel_clk: c_int = 0;
        let mut mode_line: XF86VidModeModeLine = std::mem::zeroed();
        if get_mode_line(display, screen, &mut pixel_clk, &mut mode_line) != 0
            && pixel_clk > 0
            && mode_line.htotal > 0
            && mode_line.vtotal > 0
        {
            let frame_rate =
                pixel_clk as f64 * 1000.0 / mode_line.htotal as f64 / mode_line.vtotal as f64;
            println!("Detected refresh rate of {frame_rate:.6} Hz.");
            ctx.set_rate(frame_rate as f32);
        } else {
            eprintln!("stereo_helper: XF86VidModeGetModeLine failed; defaulting to 120 Hz");
            ctx.set_rate(120.0);
        }

        x_close_display(display);
    }
}

/// GLU `gluPerspective` as a matrix, replicating the original's call with the
/// GLU implementation's exact values (column-major).
fn glu_perspective(fovy: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let f = 1.0 / (fovy / 2.0).to_radians().tan();
    let m: [f32; 16] = [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        (z_far + z_near) / (z_near - z_far),
        -1.0,
        0.0,
        0.0,
        (2.0 * z_far * z_near) / (z_near - z_far),
        0.0,
    ];
    Mat4::from_cols_array(&m)
}

/// GLU `gluLookAt` as a matrix, replicating the original's call with the GLU
/// implementation's exact values (column-major).
fn glu_look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let forward = (center - eye).normalize();
    let side = forward.cross(up).normalize();
    let up2 = side.cross(forward);

    let m: [f32; 16] = [
        side.x,
        up2.x,
        -forward.x,
        0.0,
        side.y,
        up2.y,
        -forward.y,
        0.0,
        side.z,
        up2.z,
        -forward.z,
        0.0,
        -(side.x * eye.x + side.y * eye.y + side.z * eye.z),
        -(up2.x * eye.x + up2.y * eye.y + up2.z * eye.z),
        forward.x * eye.x + forward.y * eye.y + forward.z * eye.z,
        1.0,
    ];
    Mat4::from_cols_array(&m)
}

/// The combined projection·view matrix for the given eye. `eye` is `1` for
/// left and `0` for right (matching the original). This replaces the old GL
/// matrix-stack fiddling (`glMatrixMode` + `gluPerspective`/`gluLookAt`/
/// `glFrustum`): the matrices are built identically and multiplied in the
/// same order, so `mvp * vertex` lands on exactly the NDC the old pipeline
/// produced.
pub fn project_mvp(cam: Camera, aspect: f32, eye: i32) -> Mat4 {
    // Camera basis.
    let dir = (cam.look - cam.eye).normalize();
    let right = dir.cross(cam.up).normalize();

    // Ocular shift based on which eye we're showing.
    let shift = if eye != 0 {
        right * (-(cam.iod / 2.0)) // left
    } else {
        right * (cam.iod / 2.0) // right
    };

    // The focal point is the focal distance along the view direction.
    let focus = cam.eye + dir * cam.focal;

    let (projection, view_eye, view_center) = match cam.camera_type {
        CameraType::ToeIn => {
            // Traditional perspective frusta.
            let p = glu_perspective(cam.fov, aspect, cam.near, cam.far);
            (p, cam.eye + shift, focus)
        }
        CameraType::ParallelAxisAsymmetric => {
            // Bounds of the asymmetric frustum.
            let top = cam.near * (cam.fov / 2.0).to_radians().tan();
            let bottom = -top;
            let right = if eye != 0 {
                aspect * top - 0.5 * cam.iod * (cam.near / cam.focal)
            } else {
                aspect * top + 0.5 * cam.iod * (cam.near / cam.focal)
            };
            let left = -right;

            let m: [f32; 16] = [
                2.0 * cam.near / (right - left),
                0.0,
                0.0,
                0.0,
                0.0,
                2.0 * cam.near / (top - bottom),
                0.0,
                0.0,
                (right + left) / (right - left),
                (top + bottom) / (top - bottom),
                -(cam.far + cam.near) / (cam.far - cam.near),
                -1.0,
                0.0,
                0.0,
                -(2.0 * cam.far * cam.near) / (cam.far - cam.near),
                0.0,
            ];
            let p = Mat4::from_cols_array(&m);

            // For the parallel axis camera both the eye and the focus are
            // shifted, keeping the camera direction axis parallel.
            (p, cam.eye + shift, focus + shift)
        }
    };

    // The old code post-multiplied the projection matrix (projection =
    // projection * lookAt), so clip = projection * view * vertex.
    projection * glu_look_at(view_eye, view_center, cam.up)
}