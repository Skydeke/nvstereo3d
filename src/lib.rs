//! NVIDIA 3D Vision OpenGL on Linux demo, rewritten in Rust.
//!
//! Port of `src/main.cpp` from the original C project. Renders Paul Bourke's
//! "pulsar" scene in stereo and keeps the NVIDIA 3D Vision USB IR emitter in
//! sync with the display, with GLUT replaced by winit + glutin.

pub mod gl {
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));

    // Safe, snake_case wrappers around the generated (unsafe, PascalCase)
    // methods for the small subset of fixed-function OpenGL used by the demo.
    impl Gl {
        pub fn clear(&self, mask: types::GLbitfield) {
            unsafe { self.Clear(mask) }
        }
        pub fn clear_color(&self, r: f32, g: f32, b: f32, a: f32) {
            unsafe { self.ClearColor(r, g, b, a) }
        }
        pub fn enable(&self, cap: types::GLenum) {
            unsafe { self.Enable(cap) }
        }
        pub fn disable(&self, cap: types::GLenum) {
            unsafe { self.Disable(cap) }
        }
        pub fn depth_mask(&self, flag: bool) {
            unsafe { self.DepthMask(flag as types::GLboolean) }
        }
        pub fn begin(&self, mode: types::GLenum) {
            unsafe { self.Begin(mode) }
        }
        pub fn end(&self) {
            unsafe { self.End() }
        }
        pub fn color3f(&self, r: f32, g: f32, b: f32) {
            unsafe { self.Color3f(r, g, b) }
        }
        pub fn normal3f(&self, x: f32, y: f32, z: f32) {
            unsafe { self.Normal3f(x, y, z) }
        }
        pub fn vertex3f(&self, x: f32, y: f32, z: f32) {
            unsafe { self.Vertex3f(x, y, z) }
        }
        pub fn vertex3fv(&self, v: &[f32; 3]) {
            unsafe { self.Vertex3fv(v.as_ptr()) }
        }
        pub fn push_matrix(&self) {
            unsafe { self.PushMatrix() }
        }
        pub fn pop_matrix(&self) {
            unsafe { self.PopMatrix() }
        }
        pub fn rotatef(&self, angle: f32, x: f32, y: f32, z: f32) {
            unsafe { self.Rotatef(angle, x, y, z) }
        }
        pub fn materialfv(&self, face: types::GLenum, pname: types::GLenum, params: &[f32]) {
            unsafe { self.Materialfv(face, pname, params.as_ptr()) }
        }
        pub fn shade_model(&self, mode: types::GLenum) {
            unsafe { self.ShadeModel(mode) }
        }
        pub fn light_modeli(&self, pname: types::GLenum, param: i32) {
            unsafe { self.LightModeli(pname, param) }
        }
        pub fn light_modelfv(&self, pname: types::GLenum, params: &[f32]) {
            unsafe { self.LightModelfv(pname, params.as_ptr()) }
        }
        pub fn lightfv(&self, light: types::GLenum, pname: types::GLenum, params: &[f32]) {
            unsafe { self.Lightfv(light, pname, params.as_ptr()) }
        }
        pub fn matrix_mode(&self, mode: types::GLenum) {
            unsafe { self.MatrixMode(mode) }
        }
        pub fn load_identity(&self) {
            unsafe { self.LoadIdentity() }
        }
        pub fn mult_matrixf(&self, m: &[f32]) {
            unsafe { self.MultMatrixf(m.as_ptr()) }
        }
        pub fn frustum(&self, l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) {
            unsafe { self.Frustum(l as f64, r as f64, b as f64, t as f64, n as f64, f as f64) }
        }
        pub fn pixel_storei(&self, pname: types::GLenum, param: i32) {
            unsafe { self.PixelStorei(pname, param) }
        }
        pub fn read_buffer(&self, mode: types::GLenum) {
            unsafe { self.ReadBuffer(mode) }
        }
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        #[allow(clippy::too_many_arguments)]
        pub fn read_pixels(
            &self,
            x: i32,
            y: i32,
            w: i32,
            h: i32,
            format: types::GLenum,
            ty: types::GLenum,
            pixels: *mut std::ffi::c_void,
        ) {
            unsafe { self.ReadPixels(x, y, w, h, format, ty, pixels) }
        }
        pub fn color_material(&self, face: types::GLenum, mode: types::GLenum) {
            unsafe { self.ColorMaterial(face, mode) }
        }
        pub fn viewport(&self, x: i32, y: i32, w: i32, h: i32) {
            unsafe { self.Viewport(x, y, w, h) }
        }
        pub fn ortho(&self, l: f64, r: f64, b: f64, t: f64, n: f64, f: f64) {
            unsafe { self.Ortho(l, r, b, t, n, f) }
        }
        pub fn raster_pos2i(&self, x: i32, y: i32) {
            unsafe { self.RasterPos2i(x, y) }
        }
        pub fn draw_pixels(
            &self,
            w: i32,
            h: i32,
            format: types::GLenum,
            ty: types::GLenum,
            data: *const std::ffi::c_void,
        ) {
            unsafe { self.DrawPixels(w, h, format, ty, data) }
        }
        pub fn gen_textures(&self, n: i32, textures: &mut [u32]) {
            unsafe { self.GenTextures(n, textures.as_mut_ptr()) }
        }
        pub fn delete_textures(&self, n: i32, textures: &mut [u32]) {
            unsafe { self.DeleteTextures(n, textures.as_mut_ptr()) }
        }
        pub fn gen_lists(&self, range: i32) -> u32 {
            unsafe { self.GenLists(range) }
        }
        pub fn new_list(&self, list: u32, mode: types::GLenum) {
            unsafe { self.NewList(list, mode) }
        }
        pub fn end_list(&self) {
            unsafe { self.EndList() }
        }
        pub fn call_list(&self, list: u32) {
            unsafe { self.CallList(list) }
        }
        pub fn bind_texture(&self, target: types::GLenum, texture: u32) {
            unsafe { self.BindTexture(target, texture) }
        }
        #[allow(clippy::too_many_arguments)]
        pub fn tex_image_2d(
            &self,
            target: types::GLenum,
            level: i32,
            internal_format: i32,
            width: i32,
            height: i32,
            border: i32,
            format: types::GLenum,
            ty: types::GLenum,
            pixels: *const std::ffi::c_void,
        ) {
            unsafe {
                self.TexImage2D(
                    target, level, internal_format, width, height, border, format, ty, pixels,
                )
            }
        }
        pub fn tex_parameteri(&self, target: types::GLenum, pname: types::GLenum, param: i32) {
            unsafe { self.TexParameteri(target, pname, param) }
        }
        pub fn tex_coord2f(&self, s: f32, t: f32) {
            unsafe { self.TexCoord2f(s, t) }
        }
        pub fn tex_envi(&self, target: types::GLenum, pname: types::GLenum, param: i32) {
            unsafe { self.TexEnvi(target, pname, param) }
        }
        pub fn vertex2f(&self, x: f32, y: f32) {
            unsafe { self.Vertex2f(x, y) }
        }
    }
}
mod medimg;
pub mod nvstusb;
mod pulsar;
mod scene;
mod screenshot;
mod stereo_helper;

/// Shared-memory ring coupling the `nvstereo3d-host` helper to wiz3D's
/// `Nvidia3DOutput.dll` (see [`host`]).
pub mod shm;

/// The `nvstereo3d-host` helper logic (a thin `main` in `src/bin`).
pub mod host;

use gl::Gl;
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{GlSurface, Surface, SurfaceAttributesBuilder, SwapInterval, WindowSurface};
use glutin_winit::{ApiPreference, DisplayBuilder};
use nvstusb::kms::KmsDisplay;
use nvstusb::Eye;
use raw_window_handle::HasWindowHandle;
use std::ffi::{c_int, c_void};
use std::num::NonZeroU32;
use std::time::Instant;
use stereo_helper::{Camera, CameraType, Vec3};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::Key;
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

pub fn run_demo() {
    println!("Starting up the demo app!");
    let no_emitter = std::env::args().any(|a| a == "--no-emitter");

    // Always try the direct KMS/DRM backend first: it renders straight to the
    // display engine, so the present vblank is exactly the vblank we predicted
    // and page-flipped to (best on a bare VT with the projector at 120 Hz).
    // On a composited desktop DRM master cannot be taken, this returns Err
    // quickly, and we fall through to the windowed path below.
    if let Err(e) = run_kms(no_emitter) {
        eprintln!("nvstusb: KMS/DRM backend unavailable ({e}); falling back to windowed mode");
    } else {
        // `run_kms` returned Ok only once its loop exited normally (user
        // quit) - the whole app is done, so don't fall through to windowed.
        return;
    }

    // Default windowed path: winit/glutin, which runs on the Wayland
    // compositor and handles keyboard through winit and sync through the
    // GLX/swap methods.
    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            eprintln!(
                "nvstusb: windowed fallback unavailable ({e}); run from a bare VT so the \
                 direct KMS backend can take the display"
            );
            return;
        }
    };
    let mut app = App {
        no_emitter,
        ..App::default()
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}

/// Selectable scene. `Default` is the original 3dvgl per-eye diagnostic
/// pattern (hexagons / triangles); key "2" switches to the 3dvgl-c "pulsar";
/// key "3" to the medimg random-dot stereogram; key "4" to the alternating
/// blue/red frame sync checker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SceneMode {
    HexTri,
    Pulsar,
    Rds,
    AltBlink,
}

impl Default for SceneMode {
    fn default() -> Self {
        SceneMode::HexTri
    }
}

struct App {
    window: Option<Window>,
    gl_surface: Option<Surface<WindowSurface>>,
    gl_context: Option<glutin::context::PossiblyCurrentContext>,
    gl: Option<Gl>,
    nv_ctx: Option<nvstusb::NvstusbContext>,
    kms: Option<KmsDisplay>,
    cam: Camera,
    gw: i32,
    gh: i32,
    force_eye: i32,
    current_eye: i32,
    rds_depth: i32,
    rds_bg: i32,
    /// Currently active scene: "1" hexagons/triangles, "2" 3dvgl-c pulsar,
    /// "3" medimg RDS. Defaults to the 3dvgl diagnostic pattern.
    scene: SceneMode,
    /// Pulsar spin angle (accumulated when `pulsar_rotate` is on).
    pulsar_angle: f32,
    /// Whether the pulsar is rotating (toggled via the emitter 3D button).
    pulsar_rotate: bool,
    no_emitter: bool,
    alarm_delay_us: u32,
    swap_phase_us: u32,
    last_frame_time: Option<Instant>,
    frame_stats: FrameStats,
    last_swap_ret: Option<Instant>,
    swap_stats: FrameStats,
    /// Cumulative frame-period accumulator printed every 512 frames in the
    /// style of the original C `example` (mean rate, mean period, std dev).
    frame_accum: FrameAccum,
    /// Set when the 512-frame counter boundary is hit, so the once-per-window
    /// perf report prints together with the `frame:` line instead of on its
    /// own 1-second timer.
    perf_due: bool,
    /// Absolute start time, for mid-run timing reports.
    app_start: Instant,
}

/// Cumulative frame-period statistics (never reset): mean rate, mean period,
/// and the std dev of the frame-to-frame delta, reported every 512 frames so
/// the numbers converge over the run like the C `libnvstusb/example` output.
#[derive(Default)]
struct FrameAccum {
    count: u64,
    sum_us: u128,
    sum_sq_us: u128,
}

impl FrameAccum {
    fn observe(&mut self, us: u64) {
        self.count += 1;
        self.sum_us += us as u128;
        self.sum_sq_us += (us as u128) * (us as u128);
    }

    /// Returns true when the 512-frame report boundary was hit, so the caller
    /// can print the perf block together with this line.
    fn report_if_due(&mut self, started: Instant) -> bool {
        if self.count == 0 || self.count % 512 != 0 {
            return false;
        }
        let mean_us = self.sum_us as f64 / self.count as f64;
        let var = (self.sum_sq_us as f64 / self.count as f64 - mean_us * mean_us).max(0.0);
        let sigma = var.sqrt();
        let hz = self.count as f64 / started.elapsed().as_secs_f64();
        eprintln!(
            "[frame] {} frames ({:.2} s) mean: {:.6} Hz ({:.2} us) sqrt(var): {:.2} us ({:.1} %)",
            self.count,
            started.elapsed().as_secs_f64(),
            hz,
            mean_us,
            sigma,
            sigma / mean_us * 100.0
        );
        true
    }
}

/// Rolling frame-period statistics used for the once-per-second perf report.
#[derive(Default)]
struct FrameStats {
    count: u64,
    sum_us: u128,
    min_us: u64,
    max_us: u64,
    window_start: Option<Instant>,
}

impl FrameStats {
    fn observe(&mut self, us: u64) {
        if self.window_start.is_none() {
            self.window_start = Some(Instant::now());
        }
        self.count += 1;
        self.sum_us += us as u128;
        if self.count == 1 || us < self.min_us {
            self.min_us = us;
        }
        if us > self.max_us {
            self.max_us = us;
        }
    }

    /// Prints one summary line and resets the accumulation window.
    fn report_and_reset(
        &mut self,
        refresh: f32,
        gw: i32,
        gh: i32,
        phase_us: u32,
        write_stats: (u64, u64, u64, u64),
        wait_stats: (u64, u64, u64),
        swap_stats: &FrameStats,
        drm_present: (u64, i64, i64),
        drm_resync: (u64, i64, i64),
        kms_inverted: bool,
    ) {
        let Some(start) = self.window_start else { return };
        let elapsed = start.elapsed();
        self.window_start = None;
        if self.count == 0 {
            return;
        }

        let secs = elapsed.as_secs_f64();
        let fps = self.count as f64 / secs;
        let avg_us = self.sum_us as f64 / self.count as f64;
        let jitter_us = self.max_us.saturating_sub(self.min_us);

        let expected_us = if refresh > 0.0 {
            1_000_000.0 / refresh as f64
        } else {
            0.0
        };
        let vsync = if expected_us > 0.0 {
            let period_ratio = (1_000_000.0 / fps) / expected_us;
            if (period_ratio - 1.0).abs() < 0.02 {
                "OK (1x refresh)"
            } else if (period_ratio - 2.0).abs() < 0.02 {
                "SUSPICIOUS: 2x period (swap not at every frame)"
            } else {
                "BROKEN: period does not match refresh"
            }
        } else {
            "unknown (no refresh rate)"
        };

        let w_avg = if write_stats.0 > 0 {
            write_stats.1 as f64 / write_stats.0 as f64
        } else {
            0.0
        };
        let g_avg = if wait_stats.0 > 0 {
            wait_stats.1 as f64 / wait_stats.0 as f64
        } else {
            0.0
        };

        let s_avg = if swap_stats.count > 0 {
            swap_stats.sum_us as f64 / swap_stats.count as f64
        } else {
            0.0
        };

        eprintln!(
            "[perf] {} frames in {:.3}s: fps={:.2}  period avg={:.0}us min={}us max={}us jitter={}us | expected={:.0}us -> vsync {}",
            self.count, secs, fps, avg_us, self.min_us, self.max_us, jitter_us, expected_us, vsync
        );
        eprintln!(
            "[perf]   vblank wait avg={:.0}us max={}us | swap write avg={:.0}us max={}us slow={} | window {}x{} @ {:.2} Hz | phase {}us",
            g_avg,
            wait_stats.2,
            w_avg,
            write_stats.2,
            write_stats.3,
            gw,
            gh,
            refresh,
            phase_us
        );
        if swap_stats.count > 0 {
            eprintln!(
                "[perf]   swap return->return: avg={:.0}us min={}us max={}us jitter={}us",
                s_avg, swap_stats.min_us, swap_stats.max_us,
                swap_stats.max_us.saturating_sub(swap_stats.min_us)
            );
        }
        if drm_present.0 > 0 || drm_resync.0 > 0 {
            let p_avg = if drm_present.0 > 0 {
                drm_present.1 as f64 / drm_present.0 as f64
            } else {
                0.0
            };
            let r_avg = if drm_resync.0 > 0 {
                drm_resync.1 as f64 / drm_resync.0 as f64
            } else {
                0.0
            };
            eprintln!(
                "[perf]   drm present err: avg={:.0}us max={}us (predicted vblank - swap return) | resync: n={} avg={:.0}us max={}us | eye inversion: {}",
                p_avg, drm_present.2, drm_resync.0, r_avg, drm_resync.2,
                if kms_inverted { "ON (flip lands 1 vblank late)" } else { "OFF (flip lands on prediction)" }
            );
        }

        self.count = 0;
        self.sum_us = 0;
        self.min_us = 0;
        self.max_us = 0;
    }
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            gl_surface: None,
            gl_context: None,
            gl: None,
            nv_ctx: None,
            kms: None,
            cam: Camera::default(),
            gw: 800,
            gh: 600,
            force_eye: 0,
            current_eye: 0,
            rds_depth: medimg::DEFAULT_DEPTH_PX,
            rds_bg: medimg::DEFAULT_BG_SHIFT,
            scene: SceneMode::default(),
            pulsar_angle: 0.0,
            pulsar_rotate: true,
            no_emitter: false,
            alarm_delay_us: 0,
            swap_phase_us: 2080,
            last_frame_time: None,
            frame_stats: FrameStats::default(),
            last_swap_ret: None,
            swap_stats: FrameStats::default(),
            frame_accum: FrameAccum::default(),
            perf_due: false,
            app_start: Instant::now(),
        }
    }
}

impl App {
    /// Renders and presents one frame. Works with both the winit/glutin
    /// surface and the direct-KMS backend: whichever is present wins. Returns
    /// early (doing nothing) before the windowed path is set up.
    fn render_once(&mut self) {
        let Some(gl) = self.gl.as_ref() else {
            return;
        };

        // Which eye are we on? (1/0 for left/right)
        self.current_eye = (self.current_eye + 1) % 2;

        // Eye actually being rendered this frame once `force_eye` is
        // applied. This - not the raw alternating `current_eye` - is what
        // must be sent to the emitter below: previously the emitter was
        // always told the raw alternating eye even while `force_eye` pinned
        // the on-screen content to a single eye, so forcing an eye left the
        // shutters flipping normally against static content (both eyes see
        // the same forced image). Mirrors the `show` computation in `draw`.
        let show = match self.force_eye {
            0 => self.current_eye,
            1 => 1,
            _ => 0,
        };

        // Measure the frame period (time between consecutive render calls).
        let now = Instant::now();
        if let Some(prev) = self.last_frame_time.take() {
            let us = now.duration_since(prev).as_micros().min(u64::MAX as u128) as u64;
            self.frame_stats.observe(us);
            self.frame_accum.observe(us);
            if self.frame_accum.report_if_due(self.app_start) {
                self.perf_due = true;
            }
        }
        self.last_frame_time = Some(now);

        // Advance the pulsar spin (if enabled) before drawing the frame, so
        // either eye of a given frame sees the same angle (mirrors the C
        // `draw()` updating its static angle once per frame).
        if self.pulsar_rotate {
            self.pulsar_angle = (self.pulsar_angle + 1.0) % 360.0;
        }

        // Draw the frame for the current eye. `show` is already resolved,
        // so force_eye=0 here to avoid re-applying it inside `draw`.
        draw(
            gl,
            self.cam,
            self.gw,
            self.gh,
            0,
            show,
            self.scene,
            self.rds_depth,
            self.rds_bg,
            self.pulsar_angle,
        );

        // Present via whichever backend is active. `Surface` is not `Clone`,
        // so the swap closure captures disjoint field borrows directly.
        // Returns the KMS backend's flip hardware timestamp when available
        // (see `nvstusb::swap`); other backends have none to give.
        let mut swap_fn: Box<dyn FnMut() -> Option<u64> + '_> = match (
            self.gl_surface.as_ref(),
            self.gl_context.as_ref(),
            self.kms.as_mut(),
        ) {
            (Some(surface), Some(context), _) => {
                Box::new(move || {
                    let _ = surface.swap_buffers(context);
                    None
                })
            }
            (_, _, Some(kms)) => Box::new(move || match kms.present() {
                Ok(ts) => ts,
                Err(e) => {
                    eprintln!("nvstusb: kms present: {e}");
                    None
                }
            }),
            _ => Box::new(|| None),
        };

        // Let the usb emitter code swap and keep track of things.
        let eye = if show != 0 { Eye::Left } else { Eye::Right };
        match self.nv_ctx.as_mut() {
            Some(nv_ctx) => nv_ctx.swap(eye, gl, || swap_fn()),
            None => {
                swap_fn();
            }
        }

        // Isolate the present anchor: time between consecutive swap returns.
        // This excludes draw time, the phase sleep, and get_keys, so its
        // jitter shows how tightly the IR anchor tracks the vblank.
        let swap_ret = Instant::now();
        if let Some(prev) = self.last_swap_ret.take() {
            let us = swap_ret.duration_since(prev).as_micros().min(u64::MAX as u128) as u64;
            self.swap_stats.observe(us);
        }
        self.last_swap_ret = Some(swap_ret);

        // Feed the paced stream (vblank method 5) its frame-present anchor so
        // its packet phase is measured against the real on-screen frame
        // boundaries (see nvstusb::PresentAnchor).
        if let Some(nv_ctx) = self.nv_ctx.as_mut() {
            nv_ctx.notify_present();
        }

        // Get the status of the button/wheel on the emitter (you MUST do this,
        // otherwise the whole system will stall out after just a couple of
        // frames).
        if let Some(nv_ctx) = self.nv_ctx.as_mut() {
            let keys = nv_ctx.get_keys();

            // The 3D button on the IR emitter toggles the pulsar rotation.
            if keys.toggled_3d {
                self.pulsar_rotate = !self.pulsar_rotate;
                println!("Toggled rotation.");
            }

            // The wheel on the back adjusts the camera focal length (and
            // interoccular distance, keeping IOD = 1/30th of the focal length).
            if keys.delta_wheel != 0 {
                self.cam.focal += keys.delta_wheel as f32;
                self.cam.iod = self.cam.focal / 30.0;
                println!("Set camera focal length to {:.6}.", self.cam.focal);
            }
        }

        // At each 512-frame counter boundary, print the perf report (fps,
        // frame jitter, vsync health, USB write / vblank wait timing) together
        // with the `frame:` line emitted by report_if_due.
        if self.perf_due {
            self.perf_due = false;
            if self.frame_stats.window_start.is_some() {
                let (write, wait) = match self.nv_ctx.as_ref() {
                    Some(ctx) => (ctx.dbg_write_stats(), ctx.dbg_wait_stats()),
                    None => ((0, 0, 0, 0), (0, 0, 0)),
                };
                let refresh = self.nv_ctx.as_ref().map(|c| c.rate()).unwrap_or(0.0);
                let drm_present = match self.nv_ctx.as_ref() {
                    Some(ctx) if ctx.vblank_method() == 4 => ctx.drm_present_stats(),
                    _ => (0, 0, 0),
                };
                let drm_resync = match self.nv_ctx.as_ref() {
                    Some(ctx) if ctx.vblank_method() == 4 => ctx.drm_resync_stats(),
                    _ => (0, 0, 0),
                };
                let kms_inverted = match self.nv_ctx.as_ref() {
                    Some(ctx) if ctx.vblank_method() == 4 => ctx.kms_eye_inverted(),
                    _ => false,
                };
                self.frame_stats.report_and_reset(
                    refresh,
                    self.gw,
                    self.gh,
                    self.swap_phase_us,
                    write,
                    wait,
                    &self.swap_stats,
                    drm_present,
                    drm_resync,
                    kms_inverted,
                );
                self.swap_stats = FrameStats::default();
                // These are cumulative atomics/fields, not tied to
                // frame_stats' own window - reset them explicitly so next
                // window's printed avg/max reflect that window, not a
                // lifetime-since-start average.
                if let Some(nv_ctx) = self.nv_ctx.as_mut() {
                    nv_ctx.reset_drm_present_stats();
                    nv_ctx.reset_drm_resync_stats();
                }
            }
        }
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, event: KeyEvent) {
        match &event.logical_key {
            // Escape quits cleanly (the Character arm below can't see it).
            Key::Named(winit::keyboard::NamedKey::Escape) => {
                if self.process_key('\u{1b}') {
                    event_loop.exit();
                }
            }
            Key::Character(c) => {
                if let Some(ch) = c.chars().next() {
                    if self.process_key(ch) {
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    /// Shared keyboard handling for both backends. Returns true when the app
    /// should quit.
    fn process_key(&mut self, c: char) -> bool {
        match c {
            // 'q' or Escape (0x1b; Escape arrives as Key::Escape in winit and
            // as byte 0x1b on the VT/KMS raw-tty path) quits cleanly.
            'q' | 'Q' | '\u{1b}' => return true,
            'c' | 'C' => {
                if self.cam.camera_type == CameraType::ToeIn {
                    self.cam.camera_type = CameraType::ParallelAxisAsymmetric;
                    println!("Using parallel axis asymmetric frusta camera.");
                } else {
                    self.cam.camera_type = CameraType::ToeIn;
                    println!("Using toe-in stereo camera.");
                }
            }
            'f' | 'F' => {
                self.force_eye = (self.force_eye + 1) % 3;
                match self.force_eye {
                    0 => println!("Swapping eyes normally."),
                    1 => println!("Forcing left eye always."),
                    _ => println!("Forcing right eye always."),
                }
            }
            's' | 'S' => {
                if let Some(gl) = &self.gl {
                    screenshot::screenshot(gl, 0, 0, self.gw, self.gh, "screenshot.tga");
                    println!("Wrote frame buffer to screenshot.tga.");
                }
            }
            ',' | ';' => self.adjust_delay(-100),
            '.' | ':' => self.adjust_delay(100),
            '[' | '{' => self.adjust_delay(-1000),
            ']' | '}' => self.adjust_delay(1000),
            'i' | 'I' => {
                if let Some(ctx) = self.nv_ctx.as_mut() {
                    ctx.invert_eyes();
                    println!(
                        "Manual eye swap: {}",
                        if ctx.is_inverted() {
                            "ON (left/right lenses swapped)"
                        } else {
                            "OFF"
                        }
                    );
                }
            }
            '+' | '=' => {
                self.rds_depth = (self.rds_depth + 2).min(80);
                println!("RDS depth (pop-out): {} px", self.rds_depth);
            }
            '-' | '_' => {
                self.rds_depth = (self.rds_depth - 2).max(0);
                println!("RDS depth (pop-out): {} px", self.rds_depth);
            }
            'a' | 'A' => {
                self.rds_bg = (self.rds_bg + 1).min(40);
                println!("RDS background depth (convergence): {} px", self.rds_bg);
            }
            'd' | 'D' => {
                self.rds_bg = (self.rds_bg - 1).max(0);
                println!("RDS background depth (convergence): {} px", self.rds_bg);
            }
            '1' => {
                self.scene = SceneMode::HexTri;
                println!("Scene switched to 3dvgl (hexagons/triangles).");
            }
            '2' => {
                self.scene = SceneMode::Rds;
                println!("Scene switched to medimg (random-dot stereogram).");
            }
            '3' => {
                self.scene = SceneMode::AltBlink;
                println!("Scene switched to alternating blue/red (sync check).");
            }
            '4' => {
                self.scene = SceneMode::Pulsar;
                println!("Scene switched to 3dvgl-c (pulsar).");
            }
            _ => {}
        }
        false
    }

    /// Bumps the host-side post-swap phase delay by `delta` us and pushes it to
    /// the context so the shutter IR fire can be aligned live with the frame
    /// boundary without reflashing. Clamped to one frame period.
    fn adjust_delay(&mut self, delta: i32) {
        let new = (self.swap_phase_us as i32 + delta).clamp(0, 8334) as u32;
        self.swap_phase_us = new;
        if let Some(ctx) = self.nv_ctx.as_mut() {
            ctx.set_swap_phase_us(new);
        }
        println!("IR phase delay: {} us", new);
    }
}

/// Draws the frame for the given eye (1 = left, 0 = right). `eye` is the eye
/// actually projected once `force_eye` is applied. Dispatches to whichever of
/// the three merged scenes is active (`scene`): the 3dvgl diagnostic pattern,
/// the 3dvgl-c pulsar, or the medimg random-dot stereogram.
#[allow(clippy::too_many_arguments)]
fn draw(
    gl: &Gl,
    cam: Camera,
    gw: i32,
    gh: i32,
    force_eye: i32,
    eye: i32,
    scene: SceneMode,
    depth: i32,
    bg: i32,
    angle: f32,
) {
    // Reset the clear colour so a previous AltBlink blue/red doesn't leak
    // into the geometry scenes.
    gl.clear_color(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

    let show = match force_eye {
        0 => eye,
        1 => 1,
        _ => 0,
    };

    match scene {
        // The medimg RDS scene draws directly to the framebuffer via
        // glDrawPixels; it needs no camera projection.
        SceneMode::Rds => medimg::draw_rds(gl, gw, gh, show, depth, bg),
        // 3dvgl diagnostic pattern (hexagons / triangles).
        SceneMode::HexTri => {
            stereo_helper::project_camera(gl, cam, gw as f32 / gh as f32, show);
            scene::make_lighting(gl);
            scene::make_geometry(gl, cam, show);
        }
        // 3dvgl-c "pulsar".
        SceneMode::Pulsar => {
            stereo_helper::project_camera(gl, cam, gw as f32 / gh as f32, show);
            pulsar::make_lighting(gl);
            pulsar::make_geometry(gl, angle);
        }
        // Alternating blue/red frames for checking L/R sync: the whole
        // framebuffer is one solid color per eye, so any phase slip or eye
        // swap shows up immediately as a colour cast through the shutter.
        SceneMode::AltBlink => {
            let (r, g, b) = if show != 0 { (0.0, 0.0, 1.0) } else { (1.0, 0.0, 0.0) };
            gl.clear_color(r, g, b, 1.0);
            gl.clear(gl::COLOR_BUFFER_BIT);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Initialize communications with the usb emitter. `--no-emitter` skips
        // this so the scene can be tested without the IR emitter attached.
        //
        // NOTE: the emitter is NOT configured here (rate/phase/paced stream).
        // It is moved into `self.nv_ctx` below and only then configured, so the
        // paced-fallback thread (vblank method 5, started by set_rate) gets a
        // raw pointer to a context at a stable address for the rest of the
        // process lifetime.  Configuring it while it still lives in this
        // function's stack frame and then moving it into `self` left the paced
        // thread pointing at freed stack memory -> SIGSEGV a moment after init.
        let nv_ctx = if self.no_emitter {
            eprintln!("Skipping USB emitter init (--no-emitter).");
            None
        } else {
            match nvstusb::init() {
                Some(ctx) => Some(ctx),
                None => {
                    eprintln!("Could not initialize NVIDIA 3D Vision IR emitter!");
                    event_loop.exit();
                    return;
                }
            }
        };

        // Create the window with a compatible OpenGL config. Start borderless
        // fullscreen by default (desktop mode, so the projector's 120 Hz mode
        // still applies); NVSTUSB_WINDOWED=1 restores the windowed start.
        let fullscreen = std::env::var_os("NVSTUSB_WINDOWED").is_none();
        let mut window_attributes = WindowAttributes::default()
            .with_title("NVIDIA 3D Vision OpenGL on Linux Demo")
            .with_inner_size(LogicalSize::new(800, 600));
        if fullscreen {
            window_attributes = window_attributes.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }
        let template = ConfigTemplateBuilder::new().with_depth_size(24);
        let display_builder = DisplayBuilder::new()
            .with_preference(ApiPreference::FallbackEgl)
            .with_window_attributes(Some(window_attributes));

        let (window, gl_config) =
            match display_builder.build(event_loop, template, |mut configs| {
                configs
                    .next()
                    .expect("no compatible OpenGL configuration found")
            }) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("Failed to initialize OpenGL: {e}");
                    event_loop.exit();
                    return;
                }
            };
        let window = match window {
            Some(window) => window,
            None => {
                event_loop.exit();
                return;
            }
        };

        let raw_window_handle = match window.window_handle() {
            Ok(handle) => handle.as_raw(),
            Err(e) => {
                eprintln!("Failed to get window handle: {e}");
                event_loop.exit();
                return;
            }
        };

        let inner = window.inner_size();
        let width = NonZeroU32::new(inner.width.max(1)).unwrap();
        let height = NonZeroU32::new(inner.height.max(1)).unwrap();
        eprintln!(
            "[init] window inner size: {}x{} physical px",
            inner.width, inner.height
        );

        let display = gl_config.display();

        // Create the surface.
        let surface_attributes = SurfaceAttributesBuilder::<WindowSurface>::new().build(
            raw_window_handle,
            width,
            height,
        );
        let surface =
            match unsafe { display.create_window_surface(&gl_config, &surface_attributes) } {
                Ok(surface) => surface,
                Err(e) => {
                    eprintln!("Failed to create window surface: {e}");
                    event_loop.exit();
                    return;
                }
            };

        // Create a legacy (compatibility) OpenGL context so the fixed-function
        // pipeline used by the scene is available.
        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(2, 1))))
            .build(Some(raw_window_handle));
        let not_current = match unsafe { display.create_context(&gl_config, &context_attributes) } {
            Ok(context) => context,
            Err(e) => {
                eprintln!("Failed to create OpenGL context: {e}");
                event_loop.exit();
                return;
            }
        };
        let context = match not_current.make_current(&surface) {
            Ok(context) => context,
            Err(e) => {
                eprintln!("Failed to make OpenGL context current: {e}");
                event_loop.exit();
                return;
            }
        };

        // Load the fixed-function GL functions through the display.
        let gl = Gl::load_with(|symbol| {
            let cstr = std::ffi::CString::new(symbol).unwrap();
            display.get_proc_address(&cstr)
        });

        // vsync
        if let Err(e) =
            surface.set_swap_interval(&context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()))
        {
            eprintln!("Failed to set swap interval: {e}");
        } else {
            eprintln!("[init] swap interval = 1 (vsync active)");
        }

        // Set up OpenGL state.
        gl.clear_color(0.0, 0.0, 0.0, 1.0);
        gl.enable(gl::DEPTH_TEST);
        gl.shade_model(gl::SMOOTH);
        gl.enable(gl::COLOR_MATERIAL);
        gl.color_material(gl::FRONT_AND_BACK, gl::AMBIENT_AND_DIFFUSE);
        gl.viewport(0, 0, inner.width as i32, inner.height as i32);
        screenshot::init(&gl);

        // Set up our 3D camera (see stereo_helper for more documentation).
        let mut cam = Camera::default();
        cam.camera_type = CameraType::ParallelAxisAsymmetric;
        cam.eye = Vec3::new(39.0, 53.0, 22.0);
        cam.look = Vec3::new(0.0, 0.0, 0.0);
        cam.up = Vec3::new(0.0, 1.0, 0.0);
        cam.focal = 70.0;
        cam.fov = 50.0;
        cam.iod = cam.focal / 30.0;
        cam.near = 1.0;
        cam.far = 200.0;

        self.window = Some(window);
        self.gl_surface = Some(surface);
        self.gl_context = Some(context);
        self.gl = Some(gl);
        self.nv_ctx = nv_ctx;
        self.cam = cam;
        self.gw = inner.width as i32;
        self.gh = inner.height as i32;

        // Auto-config the vsync rate.  Runs now that the context lives at a
        // stable address in `self` (see the note at the top of `resumed`);
        // `config_refresh_rate` may spawn the paced-fallback thread (method 5).
        if let Some(ctx) = self.nv_ctx.as_mut() {
            stereo_helper::config_refresh_rate(ctx);

            // Optional initial shutter delay (us), e.g. NVSTUSB_DELAY_US=5000.
            if let Some(raw) = std::env::var_os("NVSTUSB_DELAY_US") {
                if let Ok(s) = raw.into_string() {
                    if let Ok(v) = s.parse::<u32>() {
                        ctx.set_alarm_delay_us(v);
                        self.alarm_delay_us = v;
                        println!("Set IR alarm delay to {v} us");
                    }
                }
            }

            // Optional host-side phase delay (us), e.g. NVSTUSB_PHASE_US=3500.
            // Not applied for method 4 (DRM), where the default lead of 75us
            // matches the USB write time; use ,/. to tune instead.
            if ctx.vblank_method() != 4 {
                if let Some(raw) = std::env::var_os("NVSTUSB_PHASE_US") {
                    if let Ok(s) = raw.into_string() {
                        if let Ok(v) = s.parse::<u32>() {
                            ctx.set_swap_phase_us(v);
                            self.swap_phase_us = v;
                            println!("Set IR phase delay to {v} us");
                        }
                    }
                }
            }
        }

        if let Some(ctx) = self.nv_ctx.as_ref() {
            eprintln!(
                "[init] vblank method {}, refresh {:.2} Hz",
                ctx.vblank_method(),
                ctx.rate()
            );
            self.swap_phase_us = ctx.swap_phase_us();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.gw = size.width as i32;
                self.gh = size.height as i32;
                if let (Some(surface), Some(context), Some(gl)) =
                    (&self.gl_surface, &self.gl_context, &self.gl)
                {
                    let width = NonZeroU32::new(size.width.max(1)).unwrap();
                    let height = NonZeroU32::new(size.height.max(1)).unwrap();
                    surface.resize(context, width, height);
                    gl.viewport(0, 0, size.width as i32, size.height as i32);
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_once();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                self.handle_key(event_loop, event);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Continuous animation, like the original GLUT idle loop.
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}


/// RAII guard that restores the `NVSTUSB_DRM`/`NVSTUSB_DRM_CARD` process env
/// vars to their prior state on drop. The KMS backend sets them (right before
/// `nvstusb::init()`) so `init()` selects the DRM vblank anchor (method 4);
/// if `run_kms` returns `Err` at any point after that, this guarantees the
/// vars cannot leak into the windowed fallback's own `init()`, which would
/// otherwise force method 4 onto a composited surface and cause wrong-eye
/// flicker. On the KMS success path `run_kms` runs until the user quits, so
/// the guard only drops as the process exits.
struct DrEnvGuard {
    drm: Option<std::ffi::OsString>,
    card: Option<std::ffi::OsString>,
}

impl DrEnvGuard {
    fn capture() -> DrEnvGuard {
        DrEnvGuard {
            drm: std::env::var_os("NVSTUSB_DRM"),
            card: std::env::var_os("NVSTUSB_DRM_CARD"),
        }
    }
}

impl Drop for DrEnvGuard {
    fn drop(&mut self) {
        match &self.drm {
            Some(v) => std::env::set_var("NVSTUSB_DRM", v),
            None => std::env::remove_var("NVSTUSB_DRM"),
        }
        match &self.card {
            Some(v) => std::env::set_var("NVSTUSB_DRM_CARD", v),
            None => std::env::remove_var("NVSTUSB_DRM_CARD"),
        }
    }
}

/// Direct-KMS frame pump. Renders straight to the display engine via GBM/EGL
/// and DRM page flips, so the present vblank is the vblank we predicted. Keys
/// are read from the controlling tty (raw mode); `,`/`.`/`[`/`]`/`p`/`g`
/// adjust shutter sync live, `q` quits.
fn run_kms(no_emitter: bool) -> Result<(), String> {
    if no_emitter {
        return Err("KMS mode requires the IR emitter to drive shutter timing".into());
    }
    // Restore DRM env vars on any error so the windowed fallback never picks
    // up method 4 from a failed KMS attempt.
    let _env_guard = DrEnvGuard::capture();

    // Take over the display and establish the mode FIRST: the DRM vblank
    // anchor (below) probes for a CRTC running at ~120 Hz, which only exists
    // after our SETCRTC. On a bare VT the projector's CRTC is otherwise off.
    // NOTE: `NVSTUSB_DRM` is NOT set here. It is only set below, right before
    // `nvstusb::init()`, i.e. only once we have committed to the KMS backend.
    // Setting it earlier (before the fallible open) would leak it into the
    // windowed fallback's `init()` via the process environment and force
    // vblank method 4 (DRM anchor) onto a composited surface -> wrong-eye
    // flicker.
    let mut kms = KmsDisplay::open()?;
    let gw = kms.width as i32;
    let gh = kms.height as i32;
    eprintln!(
        "[kms] display {}x{} @ {} Hz on {}",
        kms.width, kms.height, kms.mode.vrefresh, kms.card_path
    );

    kms.make_current()?;
    let gl = Gl::load_with(|symbol| {
        let cstr = std::ffi::CString::new(symbol).unwrap();
        kms.get_proc_address(cstr.as_ptr())
    });

    // Set up OpenGL state (mirrors the windowed path).
    gl.clear_color(0.0, 0.0, 0.0, 1.0);
    gl.enable(gl::DEPTH_TEST);
    gl.shade_model(gl::SMOOTH);
    gl.enable(gl::COLOR_MATERIAL);
    gl.color_material(gl::FRONT_AND_BACK, gl::AMBIENT_AND_DIFFUSE);
    gl.viewport(0, 0, gw, gh);
    screenshot::init(&gl);

    // Frame 0: SETCRTC to make the projector's CRTC live at the chosen mode.
    // Synchronous, so the vblank clock is running before the anchor probes it.
    gl.clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
    let _ = kms.present().map_err(|e| format!("initial mode-set failed: {e}"))?;
    eprintln!("[kms] mode set, CRTC live");

    // Now init the emitter + DRM vblank anchor. Force the anchor onto the card
    // we just mode-set (its CRTC is the one running at ~120 Hz), falling back
    // to the card scan if it proves unusable. `NVSTUSB_DRM` is set only now,
    // after the KMS surface is live, so it cannot leak into any fallback path.
    std::env::set_var("NVSTUSB_DRM", "1");
    std::env::set_var("NVSTUSB_DRM_CARD", &kms.card_path);
    let mut nv_ctx = nvstusb::init().ok_or("nvstusb init failed")?;
    if nv_ctx.vblank_method() != 4 {
        return Err("KMS mode requires NVSTUSB_DRM=1 (vblank method 4)".into());
    }
    // Re-anchor the vblank prediction at the real mode rate before the first
    // report, so it never shows 0.00 Hz.
    nv_ctx.set_rate(kms.mode.vrefresh as f32);
    nv_ctx.force_resync();
    eprintln!("[kms] vblank method {}, refresh {:.2} Hz", nv_ctx.vblank_method(), nv_ctx.rate());
    // Tell the emitter sync code the *actual* observed eglSwapInterval(0)
    // behavior instead of always assuming the driver rejected it (see
    // nvstusb::swap, method 4). Ghosting on both eyes is the classic symptom
    // of getting this backwards - the wrong eye's shutter fires every frame.
    // Note: on the GBM platform eglSwapInterval(0) fails unconditionally, so
    // this is only a provisional guess; the measured present error re-decides
    // within a second or two of steady frames (see
    // nvstusb::NvstusbContext::update_kms_inversion).
    nv_ctx.set_kms_vsync_throttled(kms.vsync_throttled());
    eprintln!(
        "[kms] provisional eye inversion = {} (from eglSwapInterval(0) readback; re-derived from present err)",
        if nv_ctx.kms_eye_inverted() {
            "ON"
        } else {
            "OFF"
        }
    );

    if let Some(raw) = std::env::var_os("NVSTUSB_DELAY_US") {
        if let Ok(s) = raw.into_string() {
            if let Ok(v) = s.parse::<u32>() {
                nv_ctx.set_alarm_delay_us(v);
                println!("Set IR alarm delay to {v} us");
            }
        }
    }
    if let Some(raw) = std::env::var_os("NVSTUSB_PHASE_US") {
        if let Ok(s) = raw.into_string() {
            if let Ok(v) = s.parse::<u32>() {
                nv_ctx.set_swap_phase_us(v);
                println!("Set IR phase delay to {v} us");
            }
        }
    }

    let mut cam = Camera::default();
    cam.camera_type = CameraType::ParallelAxisAsymmetric;
    cam.eye = Vec3::new(39.0, 53.0, 22.0);
    cam.look = Vec3::new(0.0, 0.0, 0.0);
    cam.up = Vec3::new(0.0, 1.0, 0.0);
    cam.focal = 70.0;
    cam.fov = 50.0;
    cam.iod = cam.focal / 30.0;
    cam.near = 1.0;
    cam.far = 200.0;

    // Raw-mode stdin key reader. Ctrl+C is disabled in raw mode, so `q` is the
    // only clean exit (the shell restores termios on process exit anyway).
    let (ktx, krx) = std::sync::mpsc::channel::<u8>();
    let _raw = TtyKeys::enter();
    std::thread::spawn(move || {
        let mut b = [0u8; 1];
        loop {
            if unsafe { read_tty(0, b.as_mut_ptr() as *mut c_void, 1) } != 1 {
                break;
            }
            if ktx.send(b[0]).is_err() {
                break;
            }
        }
    });

    // Warm the one-shot scene assets (medimg's base-dot texture, pulsar's
    // display list) while the flip clock is quiet. Building them lazily at a
    // mid-run scene switch stalls this strict KMS loop for ~100 ms in a debug
    // build and permanently de-phases the VT shutters (~112 Hz sub-harmonic).
    medimg::warm(&gl, gw, gh);
    pulsar::warm(&gl);

    let mut app = App {
        window: None,
        gl_surface: None,
        gl_context: None,
        gl: Some(gl),
        nv_ctx: Some(nv_ctx),
        kms: Some(kms),
        cam,
        gw,
        gh,
        force_eye: 0,
        current_eye: 0,
        rds_depth: medimg::DEFAULT_DEPTH_PX,
        rds_bg: medimg::DEFAULT_BG_SHIFT,
        scene: SceneMode::default(),
        pulsar_angle: 0.0,
        pulsar_rotate: true,
        no_emitter: false,
        alarm_delay_us: 0,
        swap_phase_us: 75,
        last_frame_time: None,
        frame_stats: FrameStats::default(),
        last_swap_ret: None,
        swap_stats: FrameStats::default(),
        frame_accum: FrameAccum::default(),
        perf_due: false,
        app_start: Instant::now(),
    };
    if let Some(ctx) = app.nv_ctx.as_ref() {
        app.swap_phase_us = ctx.swap_phase_us();
    }

    let mut quit = false;
    while !quit {
        while let Ok(b) = krx.try_recv() {
            if app.process_key(b as char) {
                quit = true;
            }
        }
        if quit {
            break;
        }
        app.render_once();
    }
    eprintln!("nvstusb: KMS loop exiting");
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Termios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [u8; 32],
    c_ispeed: u32,
    c_ospeed: u32,
}

const ICANON: u32 = 0o0000002;
const ECHO: u32 = 0o0000010;
const ISIG: u32 = 0o0000001;
const VMIN: usize = 6;
const VTIME: usize = 5;
const TCSANOW: c_int = 0;

unsafe extern "C" {
    fn isatty(fd: c_int) -> c_int;
    fn tcgetattr(fd: c_int, t: *mut Termios) -> c_int;
    fn tcsetattr(fd: c_int, a: c_int, t: *const Termios) -> c_int;
    #[link_name = "read"]
    fn read_tty(fd: c_int, buf: *mut c_void, count: usize) -> isize;
}

/// Puts the controlling tty in raw-ish mode (no line buffering, no echo) and
/// restores it on drop. No-op when stdin is not a tty (e.g. piped input).
struct TtyKeys {
    orig: Termios,
}

impl TtyKeys {
    fn enter() -> Option<TtyKeys> {
        if unsafe { isatty(0) } != 1 {
            return None;
        }
        let mut t = Termios {
            c_iflag: 0,
            c_oflag: 0,
            c_cflag: 0,
            c_lflag: 0,
            c_line: 0,
            c_cc: [0; 32],
            c_ispeed: 0,
            c_ospeed: 0,
        };
        if unsafe { tcgetattr(0, &mut t) } != 0 {
            return None;
        }
        let orig = t;
        t.c_lflag &= !(ICANON | ECHO | ISIG);
        t.c_cc[VMIN] = 1;
        t.c_cc[VTIME] = 0;
        if unsafe { tcsetattr(0, TCSANOW, &t) } != 0 {
            return None;
        }
        Some(TtyKeys { orig })
    }
}

impl Drop for TtyKeys {
    fn drop(&mut self) {
        unsafe {
            tcsetattr(0, TCSANOW, &self.orig);
        }
    }
}
