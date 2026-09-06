//! NVIDIA 3D Vision OpenGL on Linux demo, rewritten in Rust.
//!
//! Port of `src/main.cpp` from the original C project. Renders Paul Bourke's
//! "pulsar" scene in stereo and keeps the NVIDIA 3D Vision USB IR emitter in
//! sync with the display, with GLUT replaced by winit + glutin.

pub mod gl {
    //! GL bindings for the demo.
    //!
    //! `glow`'s API is `unsafe`, so no scene code calls these entry points
    //! directly: the safe wrappers in [`crate::gfx`] own every GL call. This
    //! module only re-exports the `glow` context type and the GL constants.

    pub use glow::*;
    pub use glow::HasContext;

    /// The GL context handle passed through every draw call.
    pub type Gl = glow::Context;
}

mod edid;
mod gfx;
mod medimg;
pub mod nvstusb;
pub mod nvtimings;
mod pulsar;
mod scene;
mod screenshot;
mod stamp_diag;
mod stereo_helper;
mod text;

/// Shared-memory ring coupling the `nvstereo3d` helper to wiz3D's
/// `Nvidia3DOutput.dll` (see [`host`]).
pub mod shm;

/// The `nvstereo3d` helper logic (a thin `main` in `src/bin`).
pub mod host;

use gl::Gl;
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{GlSurface, Surface, SurfaceAttributesBuilder, SwapInterval, WindowSurface};
use glutin_winit::{ApiPreference, DisplayBuilder};
use nvstusb::Eye;
use raw_window_handle::HasWindowHandle;
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

    // The shutter is paced exclusively by the DRM/KMS kernel vblank anchor
    // (see drm.rs), which `nvstusb::init()` only arms when `NVSTUSB_DRM` is
    // set (was previously set by the removed KMS backend). Enable it here for
    // the windowed path. Backs off if it cannot be opened (permissions /
    // no DRM master for enumeration), and `NVSTUSB_DRM_CARD` still selects an
    // explicit card when the user wants one.
    if std::env::var_os("NVSTUSB_DRM").is_none() {
        std::env::set_var("NVSTUSB_DRM", "1");
    }

    // Windowed winit/glutin path (Wayland/X11 compositor). Shutter sync uses
    // the DRM/kernel vblank anchor (the only pacing mechanism, see drm.rs),
    // which works as a plain client via legacy WAIT_VBLANK.
    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            eprintln!("nvstusb: event loop unavailable ({e})");
            return;
        }
    };
    let mut app = App::default();
    app.no_emitter = no_emitter;
    event_loop.run_app(&mut app).expect("event loop failed");
}

/// Selectable scene. `Default` is the original 3dvgl per-eye diagnostic
/// pattern (hexagons / triangles); key "2" switches to the medimg
/// random-dot stereogram; key "3" to the alternating blue/red frame sync
/// checker; key "4" to the 3dvgl-c "pulsar".
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

/// Which per-monitor timing the `,`/`.`/`[`/`]` keys adjust.  The first
/// three are the 3DVisionActivator / NV3D-Lib "X/Y/W" shutter registers:
/// X (delay from monitor refresh start to the shutter open edge) is the primary
/// band-position knob and the default; `t` cycles to Y (open window) and W
/// (second T2 timer counter — stored per monitor, rarely needs tuning).
/// `Phase` is the host-side IR packet LEAD before the vblank boundary (the old
/// "shutter phase" knob; the shutter flip lands ~lead us before the edge).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimingTarget {
    X,
    Y,
    W,
    Phase,
}

impl TimingTarget {
    fn next(self) -> Self {
        match self {
            TimingTarget::X => TimingTarget::Y,
            TimingTarget::Y => TimingTarget::W,
            TimingTarget::W => TimingTarget::Phase,
            TimingTarget::Phase => TimingTarget::X,
        }
    }

    fn label(self) -> &'static str {
        match self {
            TimingTarget::X => "X (refresh start -> shutter open)",
            TimingTarget::Y => "Y (shutter open window)",
            TimingTarget::W => "W (2nd T2 counter)",
            TimingTarget::Phase => "LEAD (packet -> vblank)",
        }
    }
}

struct App {
    window: Option<Window>,
    gl_surface: Option<Surface<WindowSurface>>,
    gl_context: Option<glutin::context::PossiblyCurrentContext>,
    gl: Option<Gl>,
    nv_ctx: Option<nvstusb::NvstusbContext>,
    cam: Camera,
    gw: i32,
    gh: i32,
    force_eye: i32,
    current_eye: i32,
    rds_depth: i32,
    rds_bg: i32,
    /// Currently active scene: "1" hexagons/triangles, "2" medimg RDS,
    /// "3" alternating blue/red sync checker, "4" 3dvgl-c pulsar.
    /// Defaults to the 3dvgl diagnostic pattern.
    scene: SceneMode,
    /// Pulsar spin angle (accumulated when `pulsar_rotate` is on).
    pulsar_angle: f32,
    /// Whether the pulsar is rotating (toggled via the emitter 3D button).
    pulsar_rotate: bool,
    no_emitter: bool,
    alarm_delay_us: u32,
    swap_phase_us: u32,
    /// Per-monitor shutter timing profile (3DVisionActivator / NV3D-Lib
    /// "X/Y/W" model).  X = delay from monitor refresh start to the shutter
    /// open edge (us), Y = shutter open window (us), W = unused on most
    /// panels (us); Z (frame time) always follows the refresh rate, so the
    /// "3 params per monitor" are X/Y/W.  Mirrored to `nv_ctx` by
    /// `apply_shutter_timings`.
    timing_x_us: f64,
    timing_y_us: f64,
    timing_w_us: f64,
    /// Which timing value the `,`/`.`/`[`/`]` keys adjust.
    timing_target: TimingTarget,
    /// Step size the timing keys apply: `,`/`.` adjust by this many us,
    /// `[`/`]` by ten times it.  Cycle 100 -> 1000 -> 10 -> 1 with `k`
    /// (mirrors 3DVisionActivator's `I` increment toggle).
    timing_step_us: u32,
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
    /// Name of the wl_output the window was last seen on (Wayland); polled
    /// periodically so moving the window re-targets the emitter's vblank
    /// anchor to the new head.
    last_monitor_name: Option<String>,
    /// The NV3D-Lib `VENDOR_PRODUCT` base key of the current monitor (from its
    /// EDID, e.g. `ACI_23F7`).  This is how the tuned profile is keyed in
    /// `monitor_timings.json` (as `base_<refresh>`, e.g. `ACI_23F7_120`); shown on
    /// the HUD and in logs so the on-disk name is visible.  `None` when the
    /// EDID couldn't be read.
    json_monitor: Option<String>,
    /// Whether any `NVSTUSB_X_US` / `NVSTUSB_Y_US` / `NVSTUSB_W_US` override
    /// was applied at startup.  Env overrides win over `monitor_timings.json`, so
    /// the delayed head-resolution re-apply in `render_once` must not clobber
    /// them.
    env_timings_set: bool,
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
        timings: (f64, f64, f64),
        anchor: &str,
        regs_live: Option<bool>,
        write_stats: (u64, u64, u64, u64),
        wait_stats: (u64, u64, u64),
        swap_stats: &FrameStats,
        drm_present: (u64, i64, i64),
        drm_resync: (u64, i64, i64),
        json_monitor: Option<&str>,
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
        let regs_str = match regs_live {
            Some(true) => "regs live".to_string(),
            Some(false) => "regs unverified".to_string(),
            None => "regs unprobed".to_string(),
        };
        // The key this profile will be saved under in monitor_timings.json (e.g.
        // `ACI_23F7_120`), so the on-disk name shows up in the perf log too.
        let json_name = json_monitor
            .filter(|b| !b.is_empty())
            .map(|b| format!("[json {}]", nvtimings::key_for(b, nvtimings::round_refresh(refresh))))
            .unwrap_or_else(|| "[json unreadable]".to_string());
        eprintln!(
            "[perf]   vblank wait avg={:.0}us max={}us | swap write avg={:.0}us max={}us slow={} | window {}x{} @ {:.2} Hz | phase {}us | shutter X={:.2}us Y={:.2}us W={:.2}us ({regs_str}) | anchor {} | {}",
            g_avg,
            wait_stats.2,
            w_avg,
            write_stats.2,
            write_stats.3,
            gw,
            gh,
            refresh,
            phase_us,
            timings.0,
            timings.1,
            timings.2,
            anchor,
            json_name,
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
                "[perf]   drm present err: avg={:.0}us max={}us (predicted vblank - swap return) | resync: n={} avg={:.0}us max={}us",
                p_avg, drm_present.2, drm_resync.0, r_avg, drm_resync.2
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
            timing_x_us: nvstusb::ShutterTimings::reference().x_us,
            timing_y_us: nvstusb::ShutterTimings::reference().y_us,
            timing_w_us: nvstusb::ShutterTimings::reference().w_us,
            timing_target: TimingTarget::X,
            timing_step_us: 100,
            last_frame_time: None,
            frame_stats: FrameStats::default(),
            last_swap_ret: None,
            swap_stats: FrameStats::default(),
            frame_accum: FrameAccum::default(),
            perf_due: false,
            app_start: Instant::now(),
            last_monitor_name: None,
            json_monitor: None,
            env_timings_set: false,
        }
    }
}

impl App {
    /// Renders and presents one frame via the winit/glutin windowed surface.
    /// Returns early (doing nothing) before the windowed path is set up.
    fn render_once(&mut self) {
        let Some(gl) = self.gl.as_ref() else {
            return;
        };

        // Periodically re-check which wl_output the window is on so the
        // emitter's kernel vblank anchor follows it across monitors (a
        // windowed window can be dragged; a fullscreen one can be moved with
        // compositor keybinds). Cheap: two winit lookups, no syscalls.
        // Before the first output is known, poll every frame (bounded by the
        // same grace period the emitter's anchor waits before falling back to
        // a blind first-head scan): learning the output early is what lets the
        // anchor bind to the RIGHT head, not some other display.
        let boot_poll = self.last_monitor_name.is_none()
            && self.frame_accum.count < crate::nvstusb::ARM_GRACE_SWAPS;
        if boot_poll || self.frame_accum.count % 120 == 0 {
            if let Some(w) = self.window.as_ref() {
                let mon = w.current_monitor();
                let name = mon.as_ref().and_then(|m| m.name());
                let mhz = mon.as_ref().and_then(|m| m.refresh_rate_millihertz());
                if name != self.last_monitor_name {
                    // Re-derive the JSON identity key for the new head so the
                    // HUD / logs / `s` save always name the entry that would
                    // be written to monitor_timings.json.
                    let new_base = name
                        .as_deref()
                        .filter(|n| !n.is_empty())
                        .and_then(|conn| crate::edid::resolve_base(conn, None));
                    eprintln!(
                        "[monitor] window output {:?} -> {:?} (json {})",
                        self.last_monitor_name,
                        name,
                        new_base
                            .as_deref()
                            .unwrap_or("<unreadable EDID>")
                    );
                    self.last_monitor_name = name.clone();
                    self.json_monitor = new_base.clone();
                    if let Some(ctx) = self.nv_ctx.as_mut() {
                        // Re-bind the emitter's kernel vblank anchor to the
                        // output the window is now on.  On a multi-head GPU the
                        // per-head vblank grids share no fixed phase, so an
                        // anchor left on the previous head free-runs at the
                        // same refresh but wrong phase -> the shutter flips
                        // mid-scanout and the frame lock breaks ("not
                        // anchored").  Must run before config/pacing below uses
                        // the new grid.
                        ctx.set_target_connector(name.as_deref());
                        // Follow the new output's mode rate FIRST so the
                        // monitor_timings.json lookup below resolves the profile at
                        // the rate this head actually runs (a different
                        // connector may be a different panel, or the same panel
                        // at a different refresh).  Only with a plausible
                        // wl_output-reported rate: never fall back to the
                        // XWayland global rate here, it may belong to the OTHER
                        // monitor.
                        if let Some(mhz) = mhz.filter(|v| *v >= 60_000) {
                            stereo_helper::config_refresh_rate(ctx, Some(mhz));
                        }
                        // Apply the head's saved monitor_timings.json shutter profile
                        // whenever the window actually switched heads (we are
                        // inside the `name != last_monitor_name` block), so both
                        // a different EDID base and a different refresh are
                        // honored.  Skipped only when a `NVSTUSB_*_US` env
                        // override is in effect — env wins over the JSON.
                        // Resolved inline (not via a `&mut self` method) because
                        // `self.gl` is immutably borrowed for the rest of this
                        // frame.
                        if !self.env_timings_set {
                            let rate_hz = ctx.rate();
                            if rate_hz > 60.0 {
                                if let Some(base) = new_base.as_deref() {
                                    if let Some((key, e)) =
                                        nvtimings::resolve(&nvtimings::load(), base, rate_hz)
                                    {
                                        self.timing_x_us = e.x_us;
                                        self.timing_y_us = e.y_us;
                                        self.timing_w_us = e.w_us;
                                        let lead = e.lead_us.round().max(0.0) as u32;
                                        self.swap_phase_us = lead;
                                        ctx.set_swap_phase_us(lead);
                                        println!(
                                            "Loaded shutter timings from {} [{}]: X={}us Y={}us W={}us LEAD={}us Z={}us",
                                            nvtimings::db_path().display(),
                                            key,
                                            e.x_us,
                                            e.y_us,
                                            e.w_us,
                                            e.lead_us,
                                            e.z_us()
                                        );
                                    }
                                }
                            }
                        }
                        ctx.set_shutter_timings(self.timing_x_us, self.timing_y_us, self.timing_w_us);
                    }
                }
            }
        }

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

        // Left-edge debug HUD: the per-monitor shutter timing profile
        // (X/Y/W, us) and which parameter the timing keys currently adjust,
        // rendered with the bitmap `text` overlay. Always visible so tuning
        // values are readable on the AltBlink checker without a terminal.
        draw_timing_hud(
            gl,
            self.gw,
            self.gh,
            self.timing_target,
            self.timing_x_us,
            self.timing_y_us,
            self.timing_w_us,
            self.nv_ctx.as_ref().map(|c| c.rate()).unwrap_or(0.0),
            self.swap_phase_us,
            self.timing_step_us,
            self.nv_ctx.as_ref().and_then(|c| c.timings_live()),
            self.json_monitor.as_deref(),
        );

        // Present via the windowed backend. `Surface` is not `Clone`, so the
        // swap closure captures the two field borrows directly. The windowed
        // swap (glutin `swap_buffers`) has no flip hardware timestamp to give
        // back, so it always returns `None`.
        let mut swap_fn: Box<dyn FnMut() -> Option<u64> + '_> =
            match (self.gl_surface.as_ref(), self.gl_context.as_ref()) {
                (Some(surface), Some(context)) => {
                    Box::new(move || {
                        let _ = surface.swap_buffers(context);
                        None
                    })
                }
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
                self.frame_stats.report_and_reset(
                    refresh,
                    self.gw,
                    self.gh,
                    self.swap_phase_us,
                    (self.timing_x_us, self.timing_y_us, self.timing_w_us),
                    self.nv_ctx
                        .as_ref()
                        .map(|c| c.anchor_name())
                        .unwrap_or_else(|| "none".to_string())
                        .as_str(),
                    self.nv_ctx.as_ref().and_then(|c| c.timings_live()),
                    write,
                    wait,
                    &self.swap_stats,
                    drm_present,
                    drm_resync,
                    self.json_monitor.as_deref(),
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
            // 'q' or Escape (0x1b; winit delivers Escape as Key::Escape with
            // the \u{1b} character) quits cleanly.
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
            's' => self.save_timings(),
            'S' => {
                if let Some(gl) = &self.gl {
                    screenshot::screenshot(gl, 0, 0, self.gw, self.gh, "screenshot.tga");
                    println!("Wrote frame buffer to screenshot.tga.");
                }
            }
            ',' | ';' => self.adjust_timing(-(self.timing_step_us as i64)),
            '.' | ':' => self.adjust_timing(self.timing_step_us as i64),
            '[' | '{' => self.adjust_timing(-(self.timing_step_us as i64) * 10),
            ']' | '}' => self.adjust_timing(self.timing_step_us as i64 * 10),
            // Cycle which shutter timing the keys above adjust:
            // X (refresh start -> open) -> Y (open window) -> W.
            't' | 'T' => self.cycle_timing_target(),
            // Cycle the timing step size (3DVisionActivator's I key):
            // 100 -> 1000 -> 10 -> 1 us, so coarse sweeps and hairline
            // nudges both are reachable without leaving the keyboard.
            'k' | 'K' => self.cycle_timing_step(),
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

    /// Rebinds the `,`/`.`/`[`/`]` keys to the 3DVisionActivator / NV3D-Lib
    /// per-monitor shutter timings `X`/`Y`/`W` plus the host packet lead
    /// `Phase`: `adjust_timing` bumps whichever is selected and programs the
    /// emitter's timing registers (X/Y/W) or the host lead live.
    /// `,`/`.` apply the current step (`timing_step_us`, 100 by default) and
    /// `[`/`]` ten times it; `k` cycles the step through 100/1000/10/1.
    /// (Previously these keys tuned the host-side packet lead, which only ever
    /// changed clarity/pairing near a boundary — this moves the actual shutter
    /// edge through the frame instead, on firmware that honors the registers;
    /// see `NvstusbContext::timings_live`.)
    fn adjust_timing(&mut self, delta: i64) {
        let period = self
            .nv_ctx
            .as_ref()
            .map(|c| c.rate())
            .filter(|r| *r > 60.0)
            .map_or(8334.0, |r| 1e6 / r as f64);
        let new = match self.timing_target {
            TimingTarget::X => (self.timing_x_us + delta as f64).clamp(0.0, period),
            TimingTarget::Y => (self.timing_y_us + delta as f64).clamp(0.0, period),
            TimingTarget::W => (self.timing_w_us + delta as f64).clamp(0.0, period),
            TimingTarget::Phase => (self.swap_phase_us as f64 + delta as f64)
                .clamp(0.0, 7000.0),
        };
        match self.timing_target {
            TimingTarget::X => self.timing_x_us = new,
            TimingTarget::Y => self.timing_y_us = new,
            TimingTarget::W => self.timing_w_us = new,
            TimingTarget::Phase => {
                self.set_swap_phase_us(new as u32);
            }
        }
        self.apply_shutter_timings();
        println!("Shutter {}: {new:.3} us", self.timing_target.label());
    }

    /// Cycles which timing the `,`/`.`/`[`/`]` keys adjust: X -> Y -> W ->
    /// LEAD (host packet lead) -> X.
    fn cycle_timing_target(&mut self) {
        self.timing_target = self.timing_target.next();
        let v = match self.timing_target {
            TimingTarget::X => self.timing_x_us,
            TimingTarget::Y => self.timing_y_us,
            TimingTarget::W => self.timing_w_us,
            TimingTarget::Phase => self.swap_phase_us as f64,
        };
        println!(
            "Now adjusting shutter {}: {v:.3} us (X/S, Y/A, W/Q in 3DVisionActivator)",
            self.timing_target.label()
        );
    }

    /// Sets the host-side packet lead (us) in the emitter and remembers it in
    /// the App, so a `Phase` key/adjust keeps them in agreement.  No-op
    /// without an emitter.
    fn set_swap_phase_us(&mut self, us: u32) {
        self.swap_phase_us = us;
        if let Some(ctx) = self.nv_ctx.as_mut() {
            ctx.set_swap_phase_us(us);
        }
    }

    /// Cycles the timing step size the `,`/`.` (x1) and `[`/`]` (x10) keys
    /// apply, mirroring 3DVisionActivator's `I` increment toggle:
    /// 100 -> 1000 -> 10 -> 1 us.
    fn cycle_timing_step(&mut self) {
        self.timing_step_us = match self.timing_step_us {
            100 => 1000,
            1000 => 10,
            10 => 1,
            _ => 100,
        };
        println!(
            "Timing step: {} us (`,`,`.` adjust by this; `[`,`]` by {} us)",
            self.timing_step_us,
            self.timing_step_us * 10
        );
    }

    /// Pushes the app's X/Y/W shutter-timing profile into the emitter,
    /// programming the timing registers live.  No-op without an emitter.
    fn apply_shutter_timings(&mut self) {
        if let Some(ctx) = self.nv_ctx.as_mut() {
            ctx.set_shutter_timings(self.timing_x_us, self.timing_y_us, self.timing_w_us);
        }
    }

    /// Resolves the `monitor_timings.json` profile for `base` (`VENDOR_PRODUCT`) at
    /// the current measured refresh and applies it.  Used at startup and again
    /// when the window first lands on a head: at startup the wl_output is
    /// often still `None`, so the EDID identity isn't known until the periodic
    /// `[monitor]` recheck in `render_once` runs.  Only overrides when a
    /// profile actually matches, so defaults survive a missing/empty DB.
    /// Applies the per-monitor values: shutter X/Y/W registers AND the
    /// host-side IR lead (`lead_us`), both stored per monitor in the file.
    fn apply_json_timings_for(&mut self, base: &str) {
        let rate_hz = self.nv_ctx.as_ref().map(|c| c.rate()).unwrap_or(0.0);
        if rate_hz <= 60.0 {
            return;
        }
        if let Some((key, e)) = nvtimings::resolve(&nvtimings::load(), base, rate_hz) {
            self.timing_x_us = e.x_us;
            self.timing_y_us = e.y_us;
            self.timing_w_us = e.w_us;
            println!(
                "Loaded shutter timings from {} [{}]: X={}us Y={}us W={}us LEAD={}us Z={}us",
                nvtimings::db_path().display(),
                key,
                e.x_us,
                e.y_us,
                e.w_us,
                e.lead_us,
                e.z_us()
            );
            self.apply_shutter_timings();
            self.set_swap_phase_us(e.lead_us.round().max(0.0) as u32);
        }
    }

    /// Saves the currently-tuned shutter profile (X/Y/W + host lead + refresh)
    /// to `monitor_timings.json` under the active monitor's `VENDOR_PRODUCT_REFRESH`
    /// key (from its EDID, e.g. `ACI_23F7_120`).  The DB holds only the user's
    /// own tuned monitors, so `s` replaces the file with just this entry.
    /// Bound to `s`. The saved entry is what `nvstereo3d` and the demo
    /// itself re-read at startup, so tuning a monitor once and pressing `s`
    /// persists it across runs.
    fn save_timings(&mut self) {
        // Which refresh are we writing? Prefer the emitter's configured rate
        // (the measured mode rate); fall back to the reference.
        let refresh = self
            .nv_ctx
            .as_ref()
            .map(|c| c.rate())
            .filter(|r| *r > 60.0)
            .unwrap_or(120.0);

        // Monitor identity: the NV3D-Lib `VENDOR_PRODUCT` base (e.g.
        // `ACI_23F7`) for the current head, kept on `self` (from EDID) so the
        // HUD / logs / save all name the same entry.  Falls back to a live
        // EDID read if the field isn't populated yet (pressed before the first
        // render tick).
        let base = self
            .json_monitor
            .clone()
            .or_else(|| {
                self.last_monitor_name
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .and_then(|conn| crate::edid::resolve_base(conn, None))
            });
        let Some(base) = base else {
            eprintln!(
                "Could not read EDID for monitor {:?}; not saving timings \
                 (need the kernel connector name, e.g. DP-1)",
                self.last_monitor_name
            );
            return;
        };
        let key = nvtimings::key_for(&base, nvtimings::round_refresh(refresh));

        // The project's own flat per-monitor entry: the tuned X/Y/W registers,
        // the host-side IR lead (LEAD / `Phase` knob), the measured refresh and
        // a derived pixel-clock figure.  `z` (frame time) is derived as
        // 1e6/refresh at use time, so it is not stored.
        let entry = nvtimings::MonitorEntry {
            refresh_hz: refresh as f64,
            frequency_10khz: (refresh * 1e-3).round() as u64, // ~10 kHz per Hz
            x_us: self.timing_x_us,
            y_us: self.timing_y_us,
            w_us: self.timing_w_us,
            lead_us: self.swap_phase_us as f64,
        };
        // The DB holds ONLY the user's own tuned monitors, so saving replaces
        // the file with just this entry.
        match nvtimings::save_entry(&key, &entry) {
            Ok(path) => println!(
                "Saved shutter timings to {} [{}]: X={:.3}us Y={:.3}us W={:.3}us LEAD={:.0}us Z={:.3}us @ {:.3} Hz",
                path.display(),
                key,
                self.timing_x_us,
                self.timing_y_us,
                self.timing_w_us,
                self.swap_phase_us,
                1_000_000.0 / refresh as f64,
                refresh
            ),
            Err(e) => eprintln!("Failed to save shutter timings: {e}"),
        }
    }
}

/// Reads a float (microseconds, e.g. X=0.5) from an environment variable.
fn env_f64(name: &str) -> Option<f64> {
    let raw = std::env::var_os(name)?;
    raw.into_string().ok()?.trim().parse::<f64>().ok()
}

/// Draws the frame for the given eye (1 = left, 0 = right). `eye` is the eye
/// actually projected once `force_eye` is applied. Dispatches to whichever of
/// the four merged scenes is active (`scene`): the 3dvgl diagnostic pattern,
/// the medimg random-dot stereogram, the alternating blue/red sync checker,
/// or the 3dvgl-c pulsar.

/// Lifts the on-screen eye labels this far up from the bottom edge of the
/// panel, in physical centimetres. Converted to pixels from the assumed
/// panel height `SCREEN_HEIGHT_CM` (a 27" 16:9 monitor - the usual
/// 2560x1440 120 Hz panel - is ~33.6 cm tall), scaling with the framebuffer
/// height so the offset tracks the window's resolution. Adjust
/// `SCREEN_HEIGHT_CM` if the actual monitor's diagonal differs.
const LABEL_LIFT_CM: f64 = 3.0;
/// Assumed physical height of the display, for the cm -> px label lift.
const SCREEN_HEIGHT_CM: f64 = 33.6;
/// Base y (pixels from the bottom scanline) where the labels used to sit.
const LABEL_BASE_Y: i32 = 32;

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
    gfx::clear_color(gl, 0.0, 0.0, 0.0, 1.0);
    gfx::clear(gl, gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

    // The label baseline y, raised the requested 2 cm above the original
    // position (glDrawPixels y counts upward from the bottom scanline).
    let label_y = LABEL_BASE_Y
        + ((gh as f64) * LABEL_LIFT_CM / SCREEN_HEIGHT_CM).round() as i32;

    let show = match force_eye {
        0 => eye,
        1 => 1,
        _ => 0,
    };

    // The camera's projection and per-eye offset are baked into one
    // model-view-projection matrix per frame (the fixed-function matrix stack
    // is gone); the geometry scenes submit their already-transformed vertex
    // data under this `mvp`.
    let mvp = stereo_helper::project_mvp(cam, gw as f32 / gh as f32, show);

    match scene {
        // The medimg RDS scene draws straight to the framebuffer via a
        // full-screen textured quad; it needs no camera projection.
        SceneMode::Rds => medimg::draw_rds(gl, gw, gh, show, depth, bg),
        // 3dvgl diagnostic pattern (hexagons / triangles).
        SceneMode::HexTri => {
            scene::make_geometry(gl, cam, show, mvp);
            // Label which lens should see which pattern, in that eye's own
            // scene colour. Each eye's frame is labelled for itself, so
            // through the shutters the mapping - and any L/R inversion - is
            // easy to read off the screen.
            let (label, (tr, tg, tb)) = if show != 0 {
                ("LEFT: GREEN HEXAGONS", (0.0f32, 0.9f32, 0.1f32))
            } else {
                ("RIGHT: BLUE TRIANGLES", (0.2f32, 0.45f32, 1.0f32))
            };
            let scale = (gw / 640).clamp(2, 6);
            let w = label.len() as i32 * 6 * scale;
            text::draw_text(gl, gw, gh, label, (gw - w) / 2, label_y, scale, tr, tg, tb);
        }
        // 3dvgl-c "pulsar".
        SceneMode::Pulsar => {
            pulsar::make_geometry(gl, angle, mvp);
        }
        // Alternating red/blue frames for checking L/R sync: the whole
        // framebuffer is one solid color per eye, so any phase slip or eye
        // swap shows up immediately as a colour cast through the shutter.
        // Red is the LEFT eye and blue the RIGHT eye.
        SceneMode::AltBlink => {
            let (r, g, b) = if show != 0 { (1.0, 0.0, 0.0) } else { (0.0, 0.0, 1.0) };
            gfx::clear_color(gl, r, g, b, 1.0);
            gfx::clear(gl, gl::COLOR_BUFFER_BIT);
            // Label which lens should see which colour. White text: drawn in
            // the frame's own colour, the label would vanish into the solid
            // background.
            let label = if show != 0 { "LEFT: RED" } else { "RIGHT: BLUE" };
            let scale = (gw / 640).clamp(2, 6);
            let w = label.len() as i32 * 6 * scale;
            text::draw_text(gl, gw, gh, label, (gw - w) / 2, label_y, scale, 1.0, 1.0, 1.0);
        }
    }
}

/// Draws the left-edge debug HUD: which shutter-timing parameter the
/// `,`/`.`/`[`/`]` keys adjust, the current X/Y/W values (the 3D Vision
/// per-monitor timings, microseconds), the refresh rate (Z is fixed to it),
/// the host packet lead, whether the emitter actually honors the timing
/// registers (genuine firmware vs clone) and the current step size.  Drawn
/// with the bitmap `text` overlay after the scene, so it reads on any scene —
/// including the solid AltBlink checker — without a GL font dependency.
#[allow(clippy::too_many_arguments)]
fn draw_timing_hud(
    gl: &Gl,
    gw: i32,
    gh: i32,
    target: TimingTarget,
    x_us: f64,
    y_us: f64,
    w_us: f64,
    rate_hz: f32,
    phase_us: u32,
    step_us: u32,
    regs_live: Option<bool>,
    json_monitor: Option<&str>,
) {
    let scale = (gw / 1500).clamp(1, 3);
    let lh = 7 * scale + 4;
    let x = 12;
    let mut y = 10;

    text::draw_text(gl, gw, gh, "SHUTTER TIMING", x, y, scale, 0.55, 0.55, 0.55);
    y += lh;

    let rows = [
        ("X REFRESH->OPEN", TimingTarget::X, x_us),
        ("Y OPEN WINDOW", TimingTarget::Y, y_us),
        ("W T2 COUNTER", TimingTarget::W, w_us),
        ("LEAD PACKET->VBLANK", TimingTarget::Phase, phase_us as f64),
    ];
    for (label, which, val) in rows {
        let (prefix, (r, g, b)) = if which == target {
            (">", (0.25f32, 1.0f32, 0.4f32))
        } else {
            (" ", (1.0f32, 1.0f32, 1.0f32))
        };
        text::draw_text(
            gl,
            gw,
            gh,
            &format!("{prefix} {label} {val:.2}"),
            x,
            y,
            scale,
            r,
            g,
            b,
        );
        y += lh;
    }

    y += lh / 2;
    let rate_str = if rate_hz > 60.0 {
        format!("  {rate_hz:.2} HZ (Z FIXED)")
    } else {
        "  RATE UNKNOWN".to_string()
    };
    text::draw_text(gl, gw, gh, &rate_str, x, y, scale, 0.55, 0.55, 0.55);
    y += lh;
    // The name the tuned profile is saved under in monitor_timings.json (e.g.
    // `ACI_23F7_120`), so what you press `s` to write is visible on screen.
    let (json_text, (jr, jg, jb)) = match json_monitor {
        Some(base) if rate_hz > 60.0 => (
            format!("  JSON <{}>", nvtimings::key_for(base, nvtimings::round_refresh(rate_hz))),
            (0.55f32, 0.75f32, 1.0f32),
        ),
        Some(base) => (
            format!("  JSON <{}> (rate unknown)", base),
            (0.55f32, 0.75f32, 1.0f32),
        ),
        None => (
            "  JSON <unreadable EDID>".to_string(),
            (0.5f32, 0.5f32, 0.5f32),
        ),
    };
    text::draw_text(gl, gw, gh, &json_text, x, y, scale, jr, jg, jb);
    y += lh;
    // Whether the timing registers were verified live: X/Y/W live means the
    // device echoed the 0x2007 block back with the written values; unverified
    // means the readback was silent or mismatched (NOT proof the block is
    // unimplemented — a device that answers late legitimately lands here).
    let (regs_text, (r, g, b)) = match regs_live {
        Some(true) => ("  TIMING REGS LIVE", (0.25f32, 1.0f32, 0.4f32)),
        Some(false) => ("  TIMING REGS UNVERIFIED", (0.9f32, 0.7f32, 0.2f32)),
        None => ("  TIMING REGS UNCHECKED", (0.5f32, 0.5f32, 0.5f32)),
    };
    text::draw_text(gl, gw, gh, regs_text, x, y, scale, r, g, b);
    y += lh;
    text::draw_text(
        gl,
        gw,
        gh,
        &format!("  STEP {} US (K=SWITCH)", step_us),
        x,
        y,
        scale,
        0.55,
        0.55,
        0.55,
    );
    y += lh;
    text::draw_text(gl, gw, gh, "  T=SWITCH TARGET  I=SWAP EYES", x, y, scale, 0.4, 0.4, 0.4);
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Initialize communications with the usb emitter. `--no-emitter` skips
        // this so the scene can be tested without the IR emitter attached.
        //
        // NOTE: the emitter is NOT configured here (rate/phase). It is moved
        // into `self.nv_ctx` below and only then configured, so any context
        // state gets a stable address for the rest of the process lifetime.
        // Configuring it while it still lives in this function's stack frame
        // and then moving it into `self` left pointers into freed stack
        // memory -> SIGSEGV a moment after init.
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

        // Which output (monitor) is this window on? On Wayland this follows
        // wl_surface.enter/leave, so for a fullscreen window it names the
        // display that scans us out. The name matches the kernel DRM
        // connector name (e.g. DP-1), which is what binds the emitter's
        // vblank anchor to the RIGHT head: with two monitors on one GPU each
        // CRTC free-runs at its own phase, and syncing to the wrong one
        // shifts the shutter flip into mid-scanout - seen through the
        // glasses as a top/bottom red/blue gradient instead of clean
        // per-eye colours.
        let monitor = window.current_monitor();
        let monitor_name = monitor.as_ref().and_then(|m| m.name());
        let monitor_mhz = monitor.as_ref().and_then(|m| m.refresh_rate_millihertz());
        eprintln!(
            "[init] window output: {:?} @ {:?} mHz",
            monitor_name,
            monitor_mhz.map(|m| m as f64 / 1000.0)
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
        let gl = unsafe {
            glow::Context::from_loader_function(|symbol| {
                let cstr = std::ffi::CString::new(symbol).unwrap();
                display.get_proc_address(&cstr)
            })
        };

        // vsync
        if let Err(e) =
            surface.set_swap_interval(&context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()))
        {
            eprintln!("Failed to set swap interval: {e}");
        } else {
            eprintln!("[init] swap interval = 1 (vsync active)");
        }

        // Set up OpenGL state.
        gfx::clear_color(&gl, 0.0, 0.0, 0.0, 1.0);
        gfx::enable(&gl, gl::DEPTH_TEST);
        gfx::viewport(&gl, 0, 0, inner.width as i32, inner.height as i32);
        screenshot::init(&gl);
        text::init(&gl);

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

        // Warm the one-shot scene assets (medimg's base-dot texture, pulsar's
        // display list) while nothing is being paced yet. Building them lazily
        // at a mid-run scene switch stalls the swap loop for ~100 ms in a debug
        // build (~seconds at 2560x1440) and de-phases the shutter packets,
        // which shows as a wrong-eye / wrong-depth flash after switching to
        // scene 2 until the stream re-locks.
        if let Some(gl) = self.gl.as_ref() {
            medimg::warm(gl, self.gw, self.gh);
            scene::warm(gl, self.cam);
            pulsar::warm(gl);
        }

        // Arm OML present feedback (vblank method 1) with the real Xlib
        // display + drawable backing the GL surface, so eye packets follow
        // actual on-screen presents - required on a composited desktop,
        // where the compositor can delay or repeat a frame.
        //
        // Window XIDs are identical between Xlib and XCB, so any window
        // handle variant works; only the display pointer must be an Xlib
        // `Display*`.  If winit hands us an XCB connection instead, we open
        // our own Xlib display on $DISPLAY - sync-value queries are
        // server-side and work from a second connection.
        // (Present-clock / OML feedback targeting removed: shutter pacing uses
        // only the DRM kernel vblank anchor, opened at init.)

        // Auto-config the vsync rate.  Runs now that the context lives at a
        // stable address in `self` (see the note at the top of `resumed`).
        if let Some(ctx) = self.nv_ctx.as_mut() {
            stereo_helper::config_refresh_rate(ctx, monitor_mhz);

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

        // Per-monitor shutter timings (3DVisionActivator / NV3D-Lib X/Y/W
        // model): optional NVSTUSB_TIMINGS_INI, then monitor_timings.json (matched
        // by refresh rate), then NVSTUSB_X_US / NVSTUSB_Y_US / NVSTUSB_W_US
        // overrides.  Later sources override earlier ones; the env overrides
        // always win.  X/Y/W program the emitter's shutter-timing registers
        // (Z/fps stays fixed to the monitor).  Defaults to the 1440p @ 120 Hz
        // reference, and the `,`/`.`/`[`/`]` keys tune them live afterwards.
        // The `s` key writes the tuned profile back into monitor_timings.json.
        let rate_hz = self.nv_ctx.as_ref().map(|c| c.rate()).unwrap_or(0.0);
        if rate_hz > 60.0 {
            if let Some(raw) = std::env::var_os("NVSTUSB_TIMINGS_INI") {
                if let Ok(path) = raw.into_string() {
                    match nvstusb::ShutterTimings::from_ini_file(&path, rate_hz, 0.5) {
                        Some(t) => {
                            self.timing_x_us = t.x_us;
                            self.timing_y_us = t.y_us;
                            self.timing_w_us = t.w_us;
                            println!(
                                "Loaded shutter timings from {path}: X={}us Y={}us Z={}us W={}us",
                                t.x_us, t.y_us, t.z_us, t.w_us
                            );
                        }
                        None => eprintln!(
                            "NVSTUSB_TIMINGS_INI={path}: no profile near {rate_hz:.3} Hz; keeping defaults"
                        ),
                    }
                }
            }
            // monitor_timings.json (the repo's checked-in DB, or a file written by
            // `s`): resolve by this monitor's EDID `VENDOR_PRODUCT` base and
            // the measured refresh rate, exactly as NV3D-Lib looks entries up.
            // Only if a profile is found do we override.  Note the base may be
            // `None` here because the wl_output is usually not known until the
            // first `[monitor]` recheck in `render_once`; that path re-applies
            // the profile once the head is resolved (see `apply_json_timings_for`).
            let json_base = monitor_name
                .as_deref()
                .filter(|s| !s.is_empty())
                .and_then(|conn| crate::edid::resolve_base(conn, None));
            self.json_monitor = json_base.clone();
            if let Some(base) = json_base.as_deref() {
                self.apply_json_timings_for(&base);
            }
            if let Some(v) = env_f64("NVSTUSB_X_US") {
                self.timing_x_us = v;
                self.env_timings_set = true;
                println!("Set shutter X (refresh->open) to {v} us");
            }
            if let Some(v) = env_f64("NVSTUSB_Y_US") {
                self.timing_y_us = v;
                self.env_timings_set = true;
                println!("Set shutter Y (open window) to {v} us");
            }
            if let Some(v) = env_f64("NVSTUSB_W_US") {
                self.timing_w_us = v;
                self.env_timings_set = true;
                println!("Set shutter W to {v} us");
            }
            self.apply_shutter_timings();
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
                    gfx::viewport(gl, 0, 0, size.width as i32, size.height as i32);
                    // Re-warm the RDS base field at the new size while outside
                    // the swap loop. `rds_texture` rebuilds lazily on the first
                    // draw after a size change; doing that inside the live swap
                    // loop de-phases the shutter packets on scene switch. No-op
                    // when the cached texture already matches this size.
                    medimg::warm(gl, size.width.max(1) as i32, size.height.max(1) as i32);
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

impl Drop for App {
    fn drop(&mut self) {
        // Fields are alive here; drops happen after. Nothing custom needs
        // doing (the past clock/present machinery was removed).
    }
}

