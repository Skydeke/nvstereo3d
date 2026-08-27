//! Driver for the NVIDIA 3D Vision USB IR emitter.
//!
//! Port of `lib/nvstusb.c` from the original C project. Talks to the emitter
//! through [`usb`], and keeps the shutter timing in sync with the display by
//! either waiting on GLX video sync, forcing a swap interval, or doing a
//! software vblank wait (reading back the front buffer).
#![allow(dead_code)]

pub mod drm;
pub mod glx;
pub mod kms;
pub mod usb;
pub mod wayland;
pub mod x11glx;

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::gl;
use crate::nvstusb::drm::DrmVblank;
use crate::nvstusb::glx::GlxExtensions;
use crate::nvstusb::usb::{t2_count, UsbDevice};

/// Debug timing accumulators for the per-frame swap path.  Filled in
/// `set_eye()` (swap-packet USB write) and `swap()` (vblank wait); the main
/// loop reads them once per second for the perf report.
static DBG_WRITE_COUNT: AtomicU64 = AtomicU64::new(0);
static DBG_WRITE_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static DBG_WRITE_MAX_US: AtomicU64 = AtomicU64::new(0);
static DBG_WRITE_SLOW: AtomicU64 = AtomicU64::new(0);
static DBG_WAIT_COUNT: AtomicU64 = AtomicU64::new(0);
static DBG_WAIT_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static DBG_WAIT_MAX_US: AtomicU64 = AtomicU64::new(0);
/// Method 4 present diagnostics: predicted vblank minus host-epoch at swap
/// return (positive = the swap returned before the predicted vblank).  A small
///, stable value means every frame lands on the vblank the shutter is synced
/// to; large values mean frames present on a different vblank (ghost/stutter).
static DRM_PRESENT_COUNT: AtomicU64 = AtomicU64::new(0);
static DRM_PRESENT_TOTAL: AtomicI64 = AtomicI64::new(0);
static DRM_PRESENT_MAX_ABS: AtomicI64 = AtomicI64::new(0);
/// Long-running method-4 present-error accumulator used ONLY to auto-decide
/// whether the eye must be inverted (see `NvstusbContext::update_kms_inversion`).
/// Deliberately separate from the perf-window `DRM_PRESENT_*` counters so the
/// once-per-second report reset cannot wipe the decision history.
static KMS_ERR_TOTAL: AtomicI64 = AtomicI64::new(0);
static KMS_ERR_COUNT: AtomicU64 = AtomicU64::new(0);

/// Records a new maximum into an atomic if `us` is larger than the current one.
fn dbg_update_max(atom: &AtomicU64, us: u64) {
    let mut prev = atom.load(Ordering::Relaxed);
    while us > prev {
        match atom.compare_exchange_weak(prev, us, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(cur) => prev = cur,
        }
    }
}

/// Sleeps for `us` microseconds as a pure busy-wait.  This is deliberately a
/// spin loop: `thread::sleep`'s nanosleep can overshoot by hundreds of us and,
/// because this delay positions the next IR packet relative to the swap vblank,
/// any overshoot turns directly into shutter ghost jitter.  Spinning costs a
/// core for a few ms per frame but keeps the phase delay deterministic (us).
fn precise_sleep(us: u64) {
    if us == 0 {
        return;
    }
    let target = Instant::now() + Duration::from_micros(us);
    while Instant::now() < target {
        std::hint::spin_loop();
    }
}

/// Busy-waits until `target`.
fn spin_until(target: Instant) {
    while Instant::now() < target {
        std::hint::spin_loop();
    }
}

/// The emitter's fixed packet -> IR alarm delay in microseconds
/// (FRAME_ALARM_DELAY_US in the RP2040 firmware): the shutter switch fires this
/// long after the eye packet is received.
pub const IR_ALARM_DELAY_US: u64 = 3_000;

/// Which shutter eye is currently active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eye {
    Left = 0,
    Right,
    Quad,
}

impl Eye {
    /// 1/0 for left/right, mirroring the eye toggling in the GLUT idle loop.
    pub fn from_u8(v: u8) -> Eye {
        if v == 0 {
            Eye::Left
        } else {
            Eye::Right
        }
    }
}

/// Key/button/wheel state read back from the emitter.
#[derive(Clone, Copy, Debug, Default)]
pub struct Keys {
    /// Signed amount the wheel was turned without the 3D button pressed.
    pub delta_wheel: i8,
    /// Signed amount the wheel was turned with the 3D button pressed.
    pub pressed_delta_wheel: i8,
    /// Whether the 3D button was pressed since the last poll.
    pub toggled_3d: bool,
}

/// Context for communicating with the NVIDIA 3D Vision IR emitter.

/// Context for communicating with the NVIDIA 3D Vision IR emitter.
pub struct NvstusbContext {
    /// Currently configured refresh rate.
    rate: f32,
    /// Currently active eye.
    eye: Eye,
    /// USB transport.
    device: UsbDevice,
    /// Last toggled 3D state.
    toggled_3d: bool,
    /// Vblank method: 0 = software, 1 = GLX_SGI_video_sync, 2 = env-var
    /// synced, 3 = GLX_SGI_swap_control, 4 = DRM/KMS vblank anchor.
    vblank_method: u8,
    /// Kernel vblank anchor (method 4), if the DRM device opened.
    drm_vblank: Option<DrmVblank>,
    /// Whether the eyes are inverted.
    invert_eyes: bool,
    /// Last swap interval requested via glXSwapIntervalSGI.
    current_swap_interval: i32,
    /// Host-side phase delay (us).  For method 1 this is the delay applied
    /// after swap_buffers before the next eye packet; for method 4 it is the
    /// packet lead before the vblank boundary (see swap()).
    swap_phase_us: u32,
    /// Stereo thread state.
    thread_running: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
    /// Resolved GLX extension entry points.
    glx: GlxExtensions,
    /// Method 4 only: whether the KMS backend's `eglSwapBuffers` eats an
    /// extra vblank of throttle, pushing our manual page flip one vblank late
    /// and requiring the "on_screen" eye inversion in `swap()`.
    ///
    /// This is NOT reliably derived from `eglSwapInterval(0)`: on the GBM
    /// platform that call returns `EGL_BAD_NATIVE_WINDOW` unconditionally
    /// (swap intervals only apply to window surfaces), so a rejection says
    /// nothing about whether `eglSwapBuffers` throttles. It is only a
    /// provisional starting guess; the real decision is re-derived from the
    /// measured present error by [`Self::update_kms_inversion`].
    kms_vsync_throttled: bool,
    /// Method 1, present feedback (OML): instant of the most recent confirmed
    /// on-screen present and the smoothed interval between presents.
    present_instant: Option<Instant>,
    vsync_period_us: u64,
    /// Method 1, present feedback: app Xlib display pointer and drawable that
    /// back the GL surface; both zero/unset when unavailable, which disables
    /// OML attribution and degrades to boundary-only pacing.
    x11_display: usize,
    x11_drawable: u64,
    /// SBC snapshot taken just before each swap (method 1); a later increase
    /// in the OML swap-block counter confirms the buffer actually reached the
    /// screen.
    pre_swap_sbc: i64,
    /// Whether a swap is awaiting present confirmation.
    pending_confirm: bool,
    /// Return instant of the swap before last (method 1), for gap anomalies.
    prev_swap_return: Option<Instant>,
    /// Wayland/EGL present clock (method 1 on native Wayland): a private EGL
    /// display (the same singleton the app's EGL uses) plus a shadow window
    /// surface over the app's `wl_surface`, used only to read the hardware
    /// vblank counter via `eglGetSyncValuesCHROMIUM`.  Never rendered to and
    /// never swapped, so it cannot interfere with the app's buffers.
    egl_clock: Option<EglClock>,
    /// Kernel DRM vblank clock (method 1 on native Wayland): true hardware
    /// boundary timestamps straight from WAIT_VBLANK, no DRM master needed.
    /// Anchoring packets to this grid removes the ppm-level beat between the
    /// compositor's callback clock and the display crystal, which made the
    /// flip point sweep through the frame (seen as a red->blue gradient).
    drm_clock: Option<DrmVblank>,
    /// Whether the lazy DRM probe already ran (avoids retry spam).
    drm_tried: bool,
    /// Normalized name of the output (wl_output / kernel connector) the app
    /// window is currently displayed on, e.g. `dp-1`. The method-1 kernel
    /// anchor is (re-)opened on the CRTC serving THAT output; on a multi-head
    /// GPU each head's vblank grid has an arbitrary phase offset to its
    /// siblings, so anchoring to the wrong one shifts the IR flip into the
    /// scanout - visible through shutter glasses as a top/bottom colour split.
    /// `None` = unknown (blind first-pipe behavior).
    target_connector: Option<String>,
    /// Runtime anchor-pipe override cycled by the demo's `o` key:
    /// 0 = auto (window output / first active head), 1..=8 = CRTC pipe
    /// index 0..=7. Exists for drivers that refuse connector enumeration to
    /// plain clients (nvidia-drm without DRM master): there the name binding
    /// cannot engage and only the user's eyes can pick the right vblank grid.
    pipe_cycle: u32,
    /// Return instant of the previous swap (method 1), for gap anomalies.
    last_swap_return: Option<Instant>,
    /// Boundary the previous packet was scheduled against (target + lead).
    /// Used to tell "packet slipped to a later boundary together with the
    /// throttled present" (render stall - pairing survives on its own) apart
    /// from "packet stayed on schedule but content lost a cycle" (compositor
    /// hiccup - glasses must hold via suppression).
    last_packet_boundary: Option<Instant>,
    /// Set when a swap-gap anomaly shows content held an extra boundary;
    /// suppresses one packet so the glasses hold in lockstep.
    suppress_next_packet: bool,
    /// Number of `swap()` calls made. Used to let the lazy method-1 kernel
    /// anchor fall back to the old first-active-head blind scan after a short
    /// grace period when the window's output is never reported (compositors
    /// without working wl_output), instead of arming blind on whatever pipe is
    /// index 0 immediately at startup - on a multi-head GPU that fires the
    /// first ~hundred packets off the WRONG head's vblank grid and then
    /// re-targets, which reads as the left/right eyes flashing a couple of
    /// times right after launch.
    swap_calls: u64,
}

/// How many swaps to wait for the window's wl_output to be reported before the
/// method-1 anchor gives up and does the old first-active-head blind scan.
pub(crate) const ARM_GRACE_SWAPS: u64 = 240;

/// Runtime-loaded EGL entry points + handles backing [`NvstusbContext::egl_clock`].
struct EglClock {
    dpy: usize,
    surf: usize,
    get_sync_values: unsafe extern "C" fn(
        dpy: usize,
        surface: usize,
        ust: *mut i64,
        msc: *mut i64,
        sbc: *mut i64,
    ) -> u32,
}

/// Initializes the controller: opens the USB device (uploading `nvstusb.fw`
/// first if needed) and picks a vblank synchronization method.
pub fn init() -> Option<NvstusbContext> {
    let usb_context = usb::usb_init()?;
    let device = usb::open_device(usb_context, include_bytes!("../../firmware/nvstusb.fw"))?;

    let glx = GlxExtensions::load();

    // Method 4: anchor the IR to the kernel's real vblank (see drm.rs).
    // Selected when NVSTUSB_DRM=1 (set by the KMS backend only after it has
    // taken the display); on failure we fall back to the normal pick.
    let mut drm_vblank = None;
    let vblank_method = if std::env::var_os("NVSTUSB_DRM").is_some() {
        match DrmVblank::open() {
            Some(d) => {
                drm_vblank = Some(d);
                4
            }
            None => {
                eprintln!("nvstusb: NVSTUSB_DRM requested but DRM anchor failed; falling back");
                0
            }
        }
    } else if std::env::var_os("__GL_SYNC_TO_VBLANK").is_some() {
        eprintln!("__GL_SYNC_TO_VBLANK defined in environment");
        2
    } else {
        let mut method = 0;
        if glx.swap_interval_sgi.is_some() {
            eprintln!("nvstusb: forcing vsync");
            method = 3;
        }
        if glx.wait_video_sync_sgi.is_some() {
            if glx.get_video_sync_sgi.is_some() {
                eprintln!("nvstusb: GLX_SGI_video_sync supported!");
            }
            method = 1;
        }
        method
    };
    eprintln!("nvstusb:selected vblank method: {}", vblank_method);

    let swap_phase_us = match vblank_method {
        // Method 4: host lead before the boundary (~USB write time).
        4 => 75,
        // Method 1 (GLX video sync): lead of the eye packet before the next
        // vblank boundary. The firmware fires the IR one alarm delay
        // (3000 us) after arrival, so 3100 us puts the shutter flip ~100 us
        // ahead of the boundary - the same fixed lead the NVIDIA driver
        // kept. Sweep with `,`/`.`.
        _ => 3100,
    };

    Some(NvstusbContext {
        rate: 0.0,
        eye: Eye::Left,
        device,
        toggled_3d: false,
        vblank_method,
        drm_vblank,
        invert_eyes: false,
        current_swap_interval: -1,
        swap_phase_us,
        thread_running: Arc::new(AtomicBool::new(false)),
        thread: None,
        glx,
        kms_vsync_throttled: true,
        present_instant: None,
        vsync_period_us: 8333,
        x11_display: 0,
        x11_drawable: 0,
        pre_swap_sbc: -1,
        pending_confirm: false,
        prev_swap_return: None,
        egl_clock: None,
        drm_clock: None,
        drm_tried: false,
        target_connector: None,
        pipe_cycle: 0,
        last_swap_return: None,
        last_packet_boundary: None,
        suppress_next_packet: false,
        swap_calls: 0,
    })
}

impl NvstusbContext {
    /// Sets the controller refresh rate (should be the monitor refresh rate).
    pub fn set_rate(&mut self, rate: f32) {
        if rate <= 60.0 {
            eprintln!(
                "nvstusb: refusing to set refresh rate {} (must be > 60)",
                rate
            );
            return;
        }

        if let Err(e) = self.device.configure(rate) {
            eprintln!("nvstusb: emitter configure failed: {e}");
        }

        self.rate = rate;
    }

    /// Sets the RP2040 clone's packet -> IR delay in microseconds (control
    /// register 0x23).  Sweep this live to align the shutter open window with
    /// the eye-frame boundary; the real nvstusb hardware is not affected.
    pub fn set_alarm_delay_us(&mut self, us: u32) {
        let cmd: [u8; 8] = [
            0x01, // write data
            0x23, // to register 0x23
            0x04, // 4 bytes follow
            0x00,
            us as u8,
            (us >> 8) as u8,
            (us >> 16) as u8,
            (us >> 24) as u8,
        ];
        let _ = self.device.write_bulk(2, &cmd);
    }

    /// Currently configured refresh rate (Hz).
    pub fn rate(&self) -> f32 {
        self.rate
    }

    /// Host-side packet lead in microseconds.
    pub fn swap_phase_us(&self) -> u32 {
        self.swap_phase_us
    }

    /// Sets the host-side packet lead (us).  Method 1 sends the eye packet
    /// this many microseconds before the next vblank boundary, so the IR
    /// flip lands at `boundary - lead + alarm delay`; `,`/`.` tuning moves
    /// the shutter switch across the image boundary live.
    pub fn set_swap_phase_us(&mut self, us: u32) {
        self.swap_phase_us = us;
    }

    /// Names the active method-1 pacing anchor for diagnostics. Includes the
    /// anchored display (`DP-1/pipe1`) so a wrong-head bind is visible in the
    /// once-per-second perf report.
    pub fn anchor_name(&self) -> String {
        if let Some(d) = self.drm_clock.as_ref() {
            format!("drm-vblank[{}]", d.label())
        } else if let Some(d) = self.drm_vblank.as_ref() {
            format!("drm-vblank[{}]", d.label())
        } else if self.egl_clock.is_some() {
            "egl-msc".to_string()
        } else if self.x11_display != 0 {
            "oml-sbc".to_string()
        } else {
            "swap-return".to_string()
        }
    }

    /// Tells the context which output (wl_output name == kernel connector
    /// name, e.g. `DP-1`) the app window is displayed on, so the kernel
    /// vblank anchor can be bound to that head's CRTC instead of whatever
    /// pipe happens to be index 0 on a multi-monitor GPU.
    ///
    /// Called again whenever the window moves to another output; that drops
    /// the current anchor and re-opens it on the new CRTC at the next frame.
    pub fn set_target_connector(&mut self, name: Option<&str>) {
        let norm = name.map(crate::nvstusb::drm::normalize_connector_name);
        if norm == self.target_connector {
            return;
        }
        eprintln!(
            "nvstusb: window output {:?} -> {:?}; re-targeting vblank anchor",
            self.target_connector.as_deref(),
            norm.as_deref(),
        );
        self.target_connector = norm;
        // Only the lazy method-1 clock follows the window; the KMS path's
        // `drm_vblank` is bound at init to the card we mode-set ourselves.
        if self.drm_clock.take().is_some() {
            eprintln!(
                "nvstusb: dropped DRM vblank anchor; re-opening on the new CRTC \
                 (one-frame hiccup expected)"
            );
        }
        self.drm_tried = false;
        // The new CRTC's vblank grid is an unrelated timeline: forget which
        // boundary the previous packet rode on so the next frame can't be
        // misread as a slipped-together render stall.
        self.last_packet_boundary = None;
    }

    /// Runtime escape hatch for the demo's `o` key: steps
    /// auto -> pipe0 -> pipe1 ... -> pipe7 -> auto. Each press drops and
    /// re-opens the kernel vblank anchor on the next CRTC pipe, so on drivers
    /// that hide connector names from plain clients (nvidia-drm without DRM
    /// master) the user can still move the IR flip onto the head that is
    /// actually showing the window - watch the blue/red scene, stop when the
    /// split disappears. No-op outside vblank method 1 (nothing else uses the
    /// lazy `drm_clock`).
    pub fn cycle_anchor_pipe(&mut self) {
        if self.vblank_method != 1 {
            return;
        }
        self.pipe_cycle = (self.pipe_cycle + 1) % 9;
        let label = if self.pipe_cycle == 0 {
            "auto".to_string()
        } else {
            format!("pipe{}", self.pipe_cycle - 1)
        };
        eprintln!("nvstusb: anchor pipe override -> {label}");
        if self.drm_clock.take().is_some() {
            eprintln!("nvstusb: re-opening the anchor (one-frame hiccup expected)");
        }
        self.drm_tried = false;
        // The new CRTC's vblank grid is an unrelated timeline: forget which
        // boundary the previous packet rode on so the next frame can't be
        // misread as a slipped-together render stall.
        self.last_packet_boundary = None;
    }

    /// Installs the app's Xlib display + drawable so method 1 can read OML
    /// sync values (`glXGetSyncValuesOML`) for the *real* GL surface.  The
    /// SBC from that call is what lets the paced stream follow actual
    /// on-screen presents instead of raw vblank ticks - required on a
    /// composited desktop, where the compositor can delay or repeat a frame.
    /// Call once per process, after the window exists; without it method 1
    /// falls back to pacing against video-sync boundaries only.
    pub fn set_x11_target(&mut self, display: usize, drawable: u64) {
        if display == 0 || drawable == 0 {
            return;
        }
        self.x11_display = display;
        self.x11_drawable = drawable;
        eprintln!(
            "nvstusb: present feedback armed (OML sync values on drawable {:#x})",
            drawable
        );
    }

    /// Arms the Wayland/EGL present clock: opens the EGL display singleton
    /// for the app's `wl_display` and creates a shadow window surface over
    /// the app's `wl_surface`, used purely to read the hardware vblank
    /// counter (`eglGetSyncValuesCHROMIUM`).  The shadow surface is never
    /// rendered into nor swapped, so it does not touch the app's buffer
    /// queue.  On failure the field stays unset and pacing degrades to the
    /// swap-return anchor.
    pub fn set_wayland_target(&mut self, wl_display: usize, wl_surface: usize) {
        if wl_display == 0 || wl_surface == 0 || self.egl_clock.is_some() {
            return;
        }
        match unsafe { egl_clock_new(wl_display, wl_surface) } {
            Ok(clock) => {
                eprintln!(
                    "nvstusb: present clock armed (EGL sync values, shadow surface {:#x})",
                    wl_surface
                );
                self.egl_clock = Some(clock);
            }
            Err(e) => {
                if self.drm_clock.is_some() {
                    eprintln!(
                        "nvstusb: EGL present clock unavailable ({e}); \
                         kernel vblank anchor active"
                    );
                } else {
                    eprintln!("nvstusb: EGL present clock unavailable ({e}); using swap anchor");
                }
            }
        }
    }

    /// Selected vblank method (0 = software, 1 = GLX_SGI_video_sync,
    /// 2 = env-var synced, 3 = GLX_SGI_swap_control, 4 = DRM vblank).
    pub fn vblank_method(&self) -> u8 {
        self.vblank_method
    }

    /// Debug stats for the swap-packet USB write:
    /// (count, total_us, max_us, slow_writes).
    pub fn dbg_write_stats(&self) -> (u64, u64, u64, u64) {
        (
            DBG_WRITE_COUNT.load(Ordering::Relaxed),
            DBG_WRITE_TOTAL_US.load(Ordering::Relaxed),
            DBG_WRITE_MAX_US.load(Ordering::Relaxed),
            DBG_WRITE_SLOW.load(Ordering::Relaxed),
        )
    }

    /// Debug stats for the vblank wait: (count, total_us, max_us).
    pub fn dbg_wait_stats(&self) -> (u64, u64, u64) {
        (
            DBG_WAIT_COUNT.load(Ordering::Relaxed),
            DBG_WAIT_TOTAL_US.load(Ordering::Relaxed),
            DBG_WAIT_MAX_US.load(Ordering::Relaxed),
        )
    }

    /// Method 4 present diagnostics: (count, total err us, max abs err us).
    /// See [`DRM_PRESENT_COUNT`].
    pub fn drm_present_stats(&self) -> (u64, i64, i64) {
        (
            DRM_PRESENT_COUNT.load(Ordering::Relaxed),
            DRM_PRESENT_TOTAL.load(Ordering::Relaxed),
            DRM_PRESENT_MAX_ABS.load(Ordering::Relaxed),
        )
    }

    /// Resets the method-4 present diagnostics. Call once per perf-report
    /// window (after reading them) so the printed avg/max reflect that
    /// window, not a lifetime-cumulative average since process start.
    pub fn reset_drm_present_stats(&self) {
        DRM_PRESENT_COUNT.store(0, Ordering::Relaxed);
        DRM_PRESENT_TOTAL.store(0, Ordering::Relaxed);
        DRM_PRESENT_MAX_ABS.store(0, Ordering::Relaxed);
    }

    /// Method 4 re-anchor diagnostics: (count, total prediction err us,
    /// max abs err us).
    pub fn drm_resync_stats(&self) -> (u64, i64, i64) {
        self.drm_vblank
            .as_ref()
            .map(|d| d.resync_stats())
            .unwrap_or((0, 0, 0))
    }

    /// Resets the method-4 re-anchor diagnostics. Call once per perf-report
    /// window (after reading them), for the same reason as
    /// [`Self::reset_drm_present_stats`].
    pub fn reset_drm_resync_stats(&mut self) {
        if let Some(d) = self.drm_vblank.as_mut() {
            d.reset_resync_stats();
        }
    }

    /// Sets whether the KMS backend's `eglSwapBuffers` still eats an extra
    /// vblank of throttle, per the real `eglSwapInterval(0)` result
    /// (`KmsDisplay::vsync_throttled`). Only affects method 4. Must be called
    /// before the first `swap()` for method 4 to fire the correct eye.
    pub fn set_kms_vsync_throttled(&mut self, throttled: bool) {
        self.kms_vsync_throttled = throttled;
    }

    /// Method 4: whether the eye is currently being inverted (the page flip
    /// lands one vblank late, so the shutter must track the previous eye).
    pub fn kms_eye_inverted(&self) -> bool {
        self.kms_vsync_throttled
    }

    /// Method 4: re-derives the eye-inversion decision from the measured
    /// present error (predicted vblank - actual flip vblank) instead of the
    /// `eglSwapInterval(0)` readback, which is meaningless on GBM surfaces.
    ///
    /// Ground truth: `~0us` means the flip lands on the predicted vblank (no
    /// throttle) -> the eye must NOT be inverted; `~-one period` means the
    /// flip lands one vblank late (throttle) -> the eye must be inverted.
    /// The two states are a full frame apart, so a short average is decisive;
    /// hysteresis around -half period prevents flapping.
    fn update_kms_inversion(&mut self) {
        let count = KMS_ERR_COUNT.load(Ordering::Relaxed);
        if count < 30 {
            return;
        }
        let total = KMS_ERR_TOTAL.load(Ordering::Relaxed);
        let period = self
            .drm_vblank
            .as_ref()
            .map(|d| d.period_us())
            .unwrap_or(8333) as i64;
        let avg = total / count as i64;
        let want = if avg < -(period * 6) / 10 {
            true
        } else if avg > -(period * 4) / 10 {
            false
        } else {
            KMS_ERR_TOTAL.store(0, Ordering::Relaxed);
            KMS_ERR_COUNT.store(0, Ordering::Relaxed);
            return;
        };
        KMS_ERR_TOTAL.store(0, Ordering::Relaxed);
        KMS_ERR_COUNT.store(0, Ordering::Relaxed);
        if want != self.kms_vsync_throttled {
            self.kms_vsync_throttled = want;
            eprintln!(
                "nvstusb: method 4 present err avg {avg}us -> flip lands {} -> eye inversion {}",
                if want {
                    "one vblank late"
                } else {
                    "on the predicted vblank"
                },
                if want {
                    "KEPT (driver throttles)"
                } else {
                    "DISABLED (driver does not throttle)"
                },
            );
        }
    }

    /// Forces the DRM anchor to re-anchor on the next frame (used after a KMS
    /// mode-set, which can shift the vblank phase).
    pub fn force_resync(&mut self) {
        if let Some(d) = self.drm_vblank.as_mut() {
            d.force_resync();
        }
    }

    /// Sets the clone's inter-token gap in microseconds (control register
    /// 0x24).  Toggle 400 <-> 4000 to test decode robustness.
    pub fn set_frame_duration_us(&mut self, us: u32) {
        let cmd: [u8; 8] = [
            0x01,
            0x24,
            0x04,
            0x00,
            us as u8,
            (us >> 8) as u8,
            (us >> 16) as u8,
            (us >> 24) as u8,
        ];
        let _ = self.device.write_bulk(2, &cmd);
    }

    /// Sets the clone's token pair order (control register 0x25):
    /// `false` = open-then-close (as shipped), `true` = close-then-open.
    pub fn set_pair_order(&mut self, close_then_open: bool) {
        let cmd: [u8; 5] = [0x01, 0x25, 0x01, 0x00, close_then_open as u8];
        let _ = self.device.write_bulk(2, &cmd);
    }

    /// Toggles whether the left/right eyes are swapped.
    pub fn invert_eyes(&mut self) {
        self.invert_eyes = !self.invert_eyes;
    }

    /// Whether the manual left/right eye swap is currently applied.
    pub fn is_inverted(&self) -> bool {
        self.invert_eyes
    }

    /// Tells the emitter which shutter eye is currently active.
    fn set_eye(&mut self, eye: Eye) {
        // Guard against a rate that was never configured (config_refresh_rate
        // always calls set_rate now, but a caller could still leave it at 0,
        // which would divide-by-zero below).  Fall back to 120 Hz.
        let rate = if self.rate > 60.0 {
            self.rate as f64
        } else {
            120.0
        };
        let r = t2_count((1e6 / rate) / 1.8) as u32;

        match eye {
            Eye::Right | Eye::Left => {
                let eye_select = if (eye == Eye::Right) != self.invert_eyes {
                    0xFE
                } else {
                    0xFF
                };
                let buf: [u8; 8] = [
                    0xAA, // set shutter state
                    eye_select,
                    0x00,
                    0x00,
                    r as u8,
                    (r >> 8) as u8,
                    (r >> 16) as u8,
                    (r >> 24) as u8,
                ];
                let t0 = Instant::now();
                let _ = self.device.write_bulk(1, &buf);
                let wus = t0.elapsed().as_micros() as u64;
                DBG_WRITE_COUNT.fetch_add(1, Ordering::Relaxed);
                DBG_WRITE_TOTAL_US.fetch_add(wus, Ordering::Relaxed);
                dbg_update_max(&DBG_WRITE_MAX_US, wus);
                if wus > 5000 {
                    DBG_WRITE_SLOW.fetch_add(1, Ordering::Relaxed);
                    eprintln!("nvstusb: slow swap-packet write took {wus} us");
                }
            }
            Eye::Quad => {
                self.set_eye(Eye::Right);
                self.set_eye(Eye::Left);
            }
        }
    }

    /// Performs a swap and toggles the eyes, keeping the emitter in sync with the
    /// display.
    ///
    /// `swap_func` performs the actual buffer swap (the equivalent of
    /// `glutSwapBuffers` in the original), returning the flip's DRM
    /// hardware vblank timestamp (CLOCK_MONOTONIC us) when the backend can
    /// supply one (currently only the KMS path; other backends return
    /// `None`). `gl` is only used by the software vblank method.
    pub fn swap<F: FnMut() -> Option<u64>>(&mut self, eye: Eye, gl: &gl::Gl, mut swap_func: F) {
        self.swap_calls += 1;
        match self.vblank_method {
            // Software vsync: swap, then read from the front buffer, which can
            // only complete after the swap has finished.
            0 => {
                swap_func();
                let mut pixels = [255u8, 0, 255, 255];
                gl.read_buffer(gl::FRONT);
                gl.read_pixels(
                    1,
                    1,
                    1,
                    1,
                    gl::RGB,
                    gl::UNSIGNED_BYTE,
                    pixels.as_mut_ptr() as *mut c_void,
                );
                self.set_eye(eye);
            }
            // GLX_SGI_video_sync + OML present feedback, NVIDIA-contract
            // pacing: send the eye packet `swap_phase_us` before the
            // boundary where this frame will ACTUALLY appear, then call the
            // blocking swap.  On a composited desktop raw vblank ticks are
            // not enough: the compositor can repeat or delay a buffer, and
            // one such hiccup leaves glasses and content permanently
            // opposite (alternating correct/inverted = "terrible sync").
            // So after each swap we confirm the present via the OML SBC
            // (swap-block counter) and pace the next packet against the
            // confirmed present instant.  If rendering finishes too late
            // for the predicted present, the packet AND the present slip
            // one period together - pairing survives skips.
            1 => {
                // Attribute the previous frame's present first (bounded).
                self.confirm_last_present();

                let period = Duration::from_micros(self.vsync_period_us);

                // Swap-gap anomaly: if the previous swap returned more than
                // ~1.25 periods after its predecessor, content held an extra
                // boundary somewhere (render stall or compositor hiccup).
                // Whether the glasses must hold too depends on which side
                // slipped - decided below once this packet's boundary is
                // known.
                let mut missed_cycle = false;
                if let (Some(prev), Some(last)) =
                    (self.prev_swap_return, self.last_swap_return)
                {
                    let dt = last.duration_since(prev);
                    if dt > period + period / 4 && dt < period * 8 {
                        missed_cycle = true;
                    }
                }
                self.prev_swap_return = self.last_swap_return;

                // Anchor: with the kernel vblank clock armed (native
                // Wayland), fire at a fixed phase of the *display's* own
                // timeline instead of the compositor's callback clock.
                // Callback-clock jitter used to smear the flip position
                // across the scanout (seen as a red->blue gradient on the
                // alternating-color scene).
                //
                // Placement model: the packet arrives ~lead us before the
                // boundary, the firmware fires IR 3000 us later, so the
                // shutter flips at ~b_next - lead + 3000 + usb.  The kernel
                // timestamp is the START of blanking (~230 us at
                // 1440p120), so the default 3100 lands the flip just
                // inside the blanking interval - tune with `,`/`.`.
                // Armed lazily, only when no better ground truth exists:
                // X11 keeps OML SBC attribution, EGL-sync-values systems
                // keep that clock.  WAIT_VBLANK needs no DRM master, so it
                // works in windowed mode alongside the compositor and also
                // seeds the true measured period (8336 us at 119.953 Hz,
                // not the nominal 120 Hz value).
                //
                // The anchor is opened on the CRTC serving the output the
                // window is on (`target_connector`): on multi-head GPUs the
                // heads' vblank grids are phase-offset arbitrarily, and
                // packets paced off the wrong head land mid-scanout.
                if !self.drm_tried
                    && self.x11_display == 0
                    && self.egl_clock.is_none()
                    && (self.pipe_cycle != 0
                        || self.target_connector.is_some()
                        // Blind first-active-head fallback: only after the
                        // wl_output has had a few seconds to be reported. The
                        // gate on target_connector is what keeps the anchor
                        // from arming on an arbitrary head at startup and
                        // then re-targeting (a packet-phase jump mid-launch,
                        // seen as the left/right eyes switching a couple of
                        // times). With the demo polling the output every
                        // frame until it is known, the anchor engages on the
                        // correct head within the first couple of frames.
                        || self.swap_calls >= ARM_GRACE_SWAPS)
                {
                    self.drm_tried = true;
                    // pipe_cycle == 0 -> auto (bind by window output, else
                    // first active head); 1..=8 -> explicit CRTC pipe index.
                    let force_pipe = if self.pipe_cycle == 0 {
                        None
                    } else {
                        Some(self.pipe_cycle - 1)
                    };
                    self.drm_clock = DrmVblank::open_preferring(
                        if force_pipe.is_none() {
                            self.target_connector.as_deref()
                        } else {
                            None
                        },
                        force_pipe,
                    );
                    if let Some(d) = &self.drm_clock {
                        self.vsync_period_us = d.period_us();
                    }
                }
                let now = Instant::now();
                let mut drm_target = None;
                let lead = Duration::from_micros((self.swap_phase_us as u64).min(7000));
                if let Some(drm) = self.drm_clock.as_mut() {
                    match drm.query_vblank() {
                        Some(b_prev) => {
                            let b_next = b_prev + drm.period_us();
                            drm_target = Some(drm.instant_of(b_next) - lead);
                        }
                        None => {} // transient ioctl failure: legacy anchor
                    }
                }
                let mut target = match drm_target {
                    Some(t) => t,
                    None => match self.present_instant {
                        Some(t) => t + period,
                        None => {
                            // No confirmed present yet: seed from a video-sync tick.
                            self.wait_vblank_boundary() + period
                        }
                    },
                };
                // Late render: skip to the next boundary with the packet.
                while target.checked_duration_since(now).map_or(true, |d| d <= lead)
                {
                    target += period;
                }

                // Classify a missed cycle by where THIS packet ended up:
                //
                // Render stall - rendering blew past one or more boundaries,
                // so the skip loop above pushed this packet to a later
                // boundary, exactly where the throttled present lands too.
                // Packet and content slip together; pairing survives on its
                // own and suppressing an extra packet would BREAK it (the
                // glasses hold while the content advances - one inverted
                // cycle, visible as a transient split/gradient).
                //
                // Compositor hiccup - the app submitted on time so the packet
                // is still on its regular next boundary, but the buffer flip
                // missed the deadline and content repeated for a cycle.  Now
                // the glasses must hold via suppression to stay paired.
                let boundary = target + lead;
                let slipped_together = match self.last_packet_boundary {
                    Some(prev_b) => {
                        let d = boundary.saturating_duration_since(prev_b);
                        d > period + period / 4 && d < period * 16
                    }
                    None => false,
                };
                self.last_packet_boundary = Some(boundary);
                if missed_cycle && !slipped_together {
                    self.suppress_next_packet = true;
                }

                if let Some(wait) =
                    target.checked_sub(lead).and_then(|t| t.checked_duration_since(now))
                {
                    precise_sleep(wait.as_micros() as u64);
                }
                if self.suppress_next_packet {
                    // Glasses hold this boundary in lockstep with content.
                    self.suppress_next_packet = false;
                } else {
                    self.pre_swap_sbc =
                        self.read_present_counter().unwrap_or(self.pre_swap_sbc);
                    self.set_eye(eye);
                    if self.x11_display != 0 || self.egl_clock.is_some() {
                        self.pending_confirm = true;
                    }
                }
                swap_func();
                self.last_swap_return = Some(Instant::now());
            }
            // __GL_SYNC_TO_VBLANK is defined: the driver does the syncing.
            2 => {
                swap_func();
                self.set_eye(eye);
            }
            // DRM/KMS vblank anchor: busy-wait until `next present vblank -
            // 3000 - lead`, send the eye packet, then swap blocks to that same
            // vblank.  The IR fires (packet + 3000us) exactly at the boundary.
            // swap_phase_us is the host lead before the boundary (~USB write
            // time); tune with ,/./[ ].
            //
            // The page flip actually lands ONE PERIOD after the predicted
            // vblank: the driver's eglSwapBuffers vblank-throttles to the
            // predicted vblank (swap interval 0 is rejected), so our
            // DRM_IOCTL_MODE_PAGE_FLIP takes effect at the following vblank.
            // The IR therefore fires during the *previous* (opposite) eye's
            // frame, so the packet must carry that eye: opening its shutter at
            // the start of its own frame (the frame currently on screen).
            4 => {
                let period_us = self
                    .drm_vblank
                    .as_ref()
                    .map(|d| d.period_us())
                    .unwrap_or(8333);
                let pred = self.drm_vblank.as_ref().map(|d| d.current_present_us());
                let deadline = self
                    .drm_vblank
                    .as_mut()
                    .and_then(|d| d.frame_start(self.swap_phase_us));
                if let Some(dl) = deadline {
                    spin_until(dl);
                }
                // Only invert to the opposite (currently-on-screen) eye when
                // the page flip actually lands one vblank after the predicted
                // present (driver throttles inside eglSwapBuffers). If the
                // flip lands on the predicted vblank there's no extra latency
                // and `eye` (the one we just rendered / are about to flip to)
                // is the one whose content is on screen when the shutter
                // opens - inverting here in that case fires the wrong shutter
                // every single frame.
                let fire_eye = if self.kms_vsync_throttled {
                    match eye {
                        Eye::Left => Eye::Right,
                        Eye::Right => Eye::Left,
                        Eye::Quad => Eye::Quad,
                    }
                } else {
                    eye
                };
                self.set_eye(fire_eye);
                let flip_mono_us = swap_func();
                // Prefer the flip event's own hardware timestamp over a
                // post-syscall Instant::now() sample - see `frame_end` and
                // `KmsDisplay::present` for why this removes scheduler/wakeup
                // jitter that was otherwise baked into the phase the emitter
                // is synced to.
                let precise_us = flip_mono_us
                    .and_then(|m| self.drm_vblank.as_ref().map(|d| d.host_us_from_mono(m)));
                if let (Some(d), Some(p)) = (self.drm_vblank.as_ref(), pred) {
                    let t = precise_us.unwrap_or_else(|| d.host_epoch_us());
                    let err = p as i64 - t as i64;
                    // Only accumulate sane frames.  Startup (mode-set, initial
                    // anchor convergence) can be off by tens of periods and, if
                    // accumulated, permanently poisons the DRM_PRESENT_* and
                    // resync averages/maxima.  Ignore anything beyond two
                    // periods so the stats reflect steady-state behavior.
                    if err.abs() < (period_us as i64) * 2 {
                        DRM_PRESENT_COUNT.fetch_add(1, Ordering::Relaxed);
                        DRM_PRESENT_TOTAL.fetch_add(err, Ordering::Relaxed);
                        let abs = err.abs();
                        let mut prev = DRM_PRESENT_MAX_ABS.load(Ordering::Relaxed);
                        while abs > prev {
                            match DRM_PRESENT_MAX_ABS.compare_exchange_weak(
                                prev,
                                abs,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => break,
                                Err(cur) => prev = cur,
                            }
                        }
                        // Feed the long-running average used for the inversion
                        // decision (same two-period gate).
                        KMS_ERR_TOTAL.fetch_add(err, Ordering::Relaxed);
                        KMS_ERR_COUNT.fetch_add(1, Ordering::Relaxed);
                    }
                }
                if let Some(d) = self.drm_vblank.as_mut() {
                    d.frame_end(precise_us);
                }
                self.update_kms_inversion();
            }
            // GLX_SGI_swap_control: set the swap interval based on the eye.
            3 => {
                let interval = if eye == Eye::Quad { 2 } else { 1 };
                if self.current_swap_interval != interval {
                    let swap_interval = self
                        .glx
                        .swap_interval_sgi
                        .expect("missing glXSwapIntervalSGI");
                    unsafe {
                        swap_interval(interval);
                    }
                    self.current_swap_interval = interval;
                }
                swap_func();
                self.set_eye(eye);
            }
            other => eprintln!("nvstusb: unknown vblank method {}", other),
        }
    }

    /// Reads the button/wheel status from the emitter.
    pub fn get_keys(&mut self) -> Keys {
        // Read and clear 3 bytes of status from 0x201F.
        let cmd: [u8; 4] = [0x42, 0x18, 0x03, 0x00];
        let _ = self.device.write_bulk(2, &cmd);

        let mut read_buf = [0u8; 7];
        let _ = self.device.read_bulk(4, &mut read_buf);

        // read_buf[0] is the offset, [1] the number of bytes, [2..4] the size of
        // the command; the requested data follows at [4..].
        let delta_wheel = read_buf[4] as i8;
        let pressed_delta_wheel = read_buf[5] as i8;
        let toggled_3d = read_buf[6] & 0x01 != 0;

        if toggled_3d {
            self.toggled_3d = !self.toggled_3d;
        }

        Keys {
            delta_wheel,
            pressed_delta_wheel,
            toggled_3d,
        }
    }

    /// Blocks until the next vblank boundary passes (SGI video sync in the
    /// app's own GLX context).  Returns the boundary instant; also refreshes
    /// the smoothed period estimate.  Used only to seed pacing before the
    /// first confirmed present.
    fn wait_vblank_boundary(&mut self) -> Instant {
        let (get, wait) = match (
            self.glx.get_video_sync_sgi,
            self.glx.wait_video_sync_sgi,
        ) {
            (Some(g), Some(w)) => (g, w),
            _ => return Instant::now(),
        };

        let mut count: u32 = 0;
        unsafe { get(&mut count) };
        // One-counter-tick wait via the parity idiom: block until the
        // counter's parity flips to the opposite of the value just read.
        let rem = (count.wrapping_add(1)) & 1;
        unsafe { wait(2, rem as i32, &mut count) };
        let now = Instant::now();

        if let Some(last) = self.present_instant {
            let dt = now.duration_since(last).as_micros() as u64;
            if (7600..=9000).contains(&dt) {
                self.vsync_period_us = (self.vsync_period_us * 3 + dt) / 4;
            }
        }
        now
    }

    /// Reads the OML swap-block counter for the app's drawable, if armed.
    fn read_sbc(&self) -> Option<i64> {
        let get = self.glx.get_sync_values_oml?;
        let (mut ust, mut msc, mut sbc) = (0i64, 0i64, 0i64);
        // GLX returns False without a current GLX context (e.g. native
        // Wayland, where these symbols resolve but are meaningless) - the
        // counters stay untouched, so treat that as "no data".
        let ok = unsafe {
            get(self.x11_display as *mut _, self.x11_drawable, &mut ust, &mut msc, &mut sbc)
        };
        if ok == 0 {
            return None;
        }
        Some(sbc)
    }

    /// Present-counter snapshot for method 1: the OML SBC on X11/GLX (only
    /// moves when OUR buffer completes), or the hardware vblank MSC via the
    /// shadow EGL surface on native Wayland (moves every boundary; used to
    /// time packets against real vblanks since SGI video sync no-ops there).
    fn read_present_counter(&self) -> Option<i64> {
        if let Some(egl) = self.egl_clock.as_ref() {
            let (mut ust, mut msc, mut sbc) = (0i64, 0i64, 0i64);
            unsafe {
                if (egl.get_sync_values)(egl.dpy, egl.surf, &mut ust, &mut msc, &mut sbc)
                    == 0
                {
                    return None;
                }
            }
            return Some(msc);
        }
        self.read_sbc()
    }

    /// Waits (bounded to ~2 periods) until the present counter shows the
    /// previous frame's swap completed, then records that instant as the
    /// pacing anchor.  On X11 this is exact buffer attribution via SBC; on
    /// Wayland it is the first vblank boundary after the swap call, which
    /// tracks the compositor's cadence closely enough to phase-lock against
    /// instead of the noisy swap-return instant.
    fn confirm_last_present(&mut self) {
        if !self.pending_confirm {
            return;
        }
        self.pending_confirm = false;

        let deadline = Instant::now() + Duration::from_micros(2 * self.vsync_period_us);
        loop {
            match self.read_present_counter() {
                Some(sbc) if sbc > self.pre_swap_sbc => {
                    let now = Instant::now();
                    if let Some(last) = self.present_instant {
                        let dt = now.duration_since(last).as_micros() as u64;
                        if (7600..=9000).contains(&dt) {
                            self.vsync_period_us = (self.vsync_period_us * 3 + dt) / 4;
                        }
                    }
                    self.present_instant = Some(now);
                    return;
                }
                None => return, // OML unavailable: legacy pacing.
                _ => {}
            }
            if Instant::now() >= deadline {
                // Present not observed in time (heavy compositor backlog).
                // Keep the old anchor; prediction re-syncs on the next hit.
                eprintln!("nvstusb: present confirmation timed out");
                return;
            }
            precise_sleep(300);
        }
    }

    /// Starts the stereo thread for `GL_STEREO` (quad-buffered) setups. The
    /// original demo doesn't use this; it drives the emitter from the idle loop
    /// instead.
    pub fn start_stereo_thread(&mut self) {
        if self.thread_running.load(Ordering::SeqCst) {
            return;
        }
        self.thread_running.store(true, Ordering::SeqCst);
        let running = self.thread_running.clone();
        let ctx_ptr = self as *mut NvstusbContext;

        // The context is only accessed from this thread while it exists (it is
        // stopped before the context is dropped), mirroring the unsynchronized
        // sharing in the original C.
        struct SendPtr(*mut NvstusbContext);
        unsafe impl Send for SendPtr {}
        impl SendPtr {
            fn get(&self) -> *mut NvstusbContext {
                self.0
            }
        }
        let ctx_ptr = SendPtr(ctx_ptr);

        self.thread = Some(thread::spawn(move || {
            let ctx = unsafe { &mut *ctx_ptr.get() };
            let hidden = match x11glx::HiddenGlx::create() {
                Some(hidden) => hidden,
                None => {
                    eprintln!("nvstusb: unable to create hidden GLX context for stereo thread");
                    running.store(false, Ordering::SeqCst);
                    return;
                }
            };

            while running.load(Ordering::SeqCst) {
                ctx.swap(Eye::Quad, &hidden.gl, || None);
                let keys = ctx.get_keys();
                if keys.toggled_3d {
                    ctx.invert_eyes();
                }
            }
        }));
    }

    /// Stops the stereo thread and waits for it to finish.
    pub fn stop_stereo_thread(&mut self) {
        if !self.thread_running.load(Ordering::SeqCst) {
            return;
        }
        self.thread_running.store(false, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for NvstusbContext {
    fn drop(&mut self) {
        self.stop_stereo_thread();
    }
}

// ---------------------------------------------------------------------------
// Wayland/EGL present clock (runtime-loaded, no build-time EGL dependency)
// ---------------------------------------------------------------------------

/// EGL config attribute constants used below.
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_WINDOW_BIT: i32 = 0x0004;
const EGL_RED_SIZE: i32 = 0x3024;
const EGL_GREEN_SIZE: i32 = 0x3023;
const EGL_BLUE_SIZE: i32 = 0x3022;
const EGL_NONE: i32 = 0x3038;

/// Builds an [`EglClock`] over the app's Wayland display + surface.
///
/// `eglGetDisplay` is specified to return the same `EGLDisplay` singleton for
/// the same native display, so this joins the app's own EGL instance rather
/// than creating a parallel one.  The shadow surface is created only so the
/// sync-value query has a target; it never receives buffers.
///
/// # Safety
/// `wl_display`/`wl_surface` must be valid for the lifetime of the returned
/// clock (they are the app's own objects, which outlive it).
unsafe fn egl_clock_new(wl_display: usize, wl_surface: usize) -> Result<EglClock, String> {
    use libloading::{Library, Symbol};

    let lib = Library::new("libEGL.so.1")
        .or_else(|_| Library::new("libEGL.so"))
        .map_err(|e| format!("dlopen libEGL: {e}"))?;
    // Leaked on purpose: the entry points must outlive the context.
    let lib: &'static Library = Box::leak(Box::new(lib));

    unsafe {
        let get_display: Symbol<unsafe extern "C" fn(usize) -> usize> =
            lib.get(b"eglGetDisplay").map_err(|e| e.to_string())?;
        let initialize: Symbol<unsafe extern "C" fn(usize, *mut i32, *mut i32) -> u32> =
            lib.get(b"eglInitialize").map_err(|e| e.to_string())?;
        let choose_config: Symbol<
            unsafe extern "C" fn(usize, *const i32, *mut usize, i32, *mut i32) -> u32,
        > = lib.get(b"eglChooseConfig").map_err(|e| e.to_string())?;
        let create_window_surface: Symbol<
            unsafe extern "C" fn(usize, usize, usize, *const i32) -> usize,
        > = lib.get(b"eglCreateWindowSurface").map_err(|e| e.to_string())?;
        let get_sync_values: Symbol<
            unsafe extern "C" fn(usize, usize, *mut i64, *mut i64, *mut i64) -> u32,
        > = lib
            .get(b"eglGetSyncValuesCHROMIUM")
            .map_err(|_| "eglGetSyncValuesCHROMIUM not exposed by this driver".to_string())?;

        let dpy = get_display(wl_display);
        if dpy == 0 {
            return Err("eglGetDisplay failed".into());
        }
        let (mut major, mut minor) = (0i32, 0i32);
        if initialize(dpy, &mut major, &mut minor) == 0 {
            return Err("eglInitialize failed".into());
        }

        let attribs = [
            EGL_SURFACE_TYPE,
            EGL_WINDOW_BIT,
            EGL_RED_SIZE,
            8,
            EGL_GREEN_SIZE,
            8,
            EGL_BLUE_SIZE,
            8,
            EGL_NONE,
        ];
        let mut config = 0usize;
        let mut num_config = 0i32;
        if choose_config(dpy, attribs.as_ptr(), &mut config, 1, &mut num_config) == 0
            || num_config < 1
        {
            return Err("eglChooseConfig found no window config".into());
        }

        let surf = create_window_surface(dpy, config, wl_surface, std::ptr::null());
        if surf == 0 {
            return Err("eglCreateWindowSurface failed".into());
        }

        Ok(EglClock {
            dpy,
            surf,
            get_sync_values: *get_sync_values,
        })
    }
}
