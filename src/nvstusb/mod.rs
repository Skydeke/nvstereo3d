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
use std::sync::{Arc, Mutex};
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

/// Method 5 (paced) fallback stream: a dedicated thread that emits the
/// alternating left/right eye packets on a monotonic clock at the configured
/// refresh period, keeping the RP2040's master mode locked (it needs a steady
/// 7600-9000 us alternating packet stream) even when neither the GLX video
/// sync nor a DRM vblank anchor correlates with the real display vblank -
/// i.e. windowed/composited desktops.  `phase_us` is the offset of the packet
/// stream against the render loop's present instants; it is AtomicI64 because
/// the main thread's `,`/`.` phase tuning (`set_swap_phase_us`) retargets it
/// live.
struct PacedStream {
    running: Arc<AtomicBool>,
    phase_us: Arc<AtomicI64>,
    thread: Option<thread::JoinHandle<()>>,
}

/// Shared frame-present anchor.  The render loop records the instant its swap
/// (present) returns every frame.  With vsync/swap-interval this is within a
/// bounded latency of the real vblank / on-screen frame boundary - a much
/// steadier reference than any process-local clock for *relative* phase.  The
/// paced stream (method 5) locks its packet schedule to these instants, so
/// sweeping the phase with the `,`/`.` keys deterministically moves the
/// shutter switch across the actual image boundaries (a free-running stream's
/// re-anchor lands on an arbitrary `Instant::now()` and cannot be swept
/// reliably by hand).
///
/// The raw swap-return instants carry ~0.5 ms of scheduler/compositor jitter,
/// so the anchor runs a phase-locked estimator (see [`PresentPll`]): it tracks
/// the *mean* present phase and period and predicts the next present.  The
/// paced stream locks its packet grid to that prediction instead of to each
/// noisy sample, which is what previously smeared the `,`/`.` null band into
/// alternating flicker.
struct PresentAnchor {
    base: Instant,
    last_us: AtomicI64,
    /// Predicted instant (us since `base`) of the next present, per the PLL.
    next_us: AtomicI64,
    /// Smoothed present period (us), for diagnostics / fallback.
    period_us: AtomicI64,
    /// Filter state; only written from the render (main) thread via `notify`
    /// (the paced thread never locks it - it only reads the atomics above, so
    /// the Mutex is contention-free and only needed to keep this Sync).
    pll: Mutex<PresentPll>,
}

/// Phase-locked estimator for the frame-present clock.  [`PresentPll::notify`]
/// is called once per rendered frame; it nudges a nominal present time toward
/// each sample (attenuating jitter) and maintains a smoothed period.  The
/// predicted next present is `tick + period`.
struct PresentPll {
    /// Filtered estimate of the current (latest) present, us since `base`.
    tick_us: f64,
    /// Smoothed period between presents, us.
    period_us: f64,
    last_sample_us: i64,
    samples: u32,
    err_sq_sum: f64,
    locked_reported: bool,
}

/// How much each observed present is allowed to move the nominal present
/// phase (1.0 would follow every sample; lower attenuates the swap-return
/// jitter but still converges on the mean present).
const PRESENT_PHASE_GAIN: f64 = 0.2;
/// How fast the smoothed present period tracks measured frame-to-frame
/// deltas.  The display clock is crystal-stable, so a low gain keeps the
/// period estimate quiet.
const PRESENT_PERIOD_GAIN: f64 = 0.05;

impl PresentAnchor {
    fn new() -> Self {
        Self {
            base: Instant::now(),
            last_us: AtomicI64::new(-1),
            next_us: AtomicI64::new(-1),
            period_us: AtomicI64::new(0),
            pll: Mutex::new(PresentPll::new()),
        }
    }

    /// Records the current instant as the latest present and advances the
    /// phase-locked prediction of the next present.
    fn notify(&self) {
        let us = self.base.elapsed().as_micros() as i64;
        self.last_us.store(us, Ordering::Relaxed);
        if let Ok(mut pll) = self.pll.lock() {
            pll.notify(us);
            self.next_us.store(
                (pll.tick_us + pll.period_us).round() as i64,
                Ordering::Relaxed,
            );
            self.period_us
                .store(pll.period_us.round() as i64, Ordering::Relaxed);
        }
    }

    /// The instant of the most recent present, if any.
    fn last(&self) -> Option<Instant> {
        let us = self.last_us.load(Ordering::Relaxed);
        if us < 0 {
            None
        } else {
            Some(self.base + Duration::from_micros(us as u64))
        }
    }

    /// The predicted instant of the next present, if the clock has seen at
    /// least one sample.
    fn next(&self) -> Option<Instant> {
        let us = self.next_us.load(Ordering::Relaxed);
        if us < 0 {
            None
        } else {
            Some(self.base + Duration::from_micros(us as u64))
        }
    }
}

impl PresentPll {
    fn new() -> Self {
        Self {
            tick_us: 0.0,
            // 120 Hz default; corrected from the first frame-to-frame delta.
            period_us: 8333.0,
            last_sample_us: -1,
            samples: 0,
            err_sq_sum: 0.0,
            locked_reported: false,
        }
    }

    fn notify(&mut self, t: i64) {
        if self.last_sample_us < 0 {
            self.tick_us = t as f64;
            self.last_sample_us = t;
            self.samples = 1;
            return;
        }

        let dt = (t - self.last_sample_us) as f64;
        self.last_sample_us = t;

        // Period tracking: only accept plausible frame-to-frame deltas so a
        // stall or doubled present can't corrupt the smoothed period.
        if dt > 0.5 * self.period_us && dt < 1.5 * self.period_us {
            self.period_us += PRESENT_PERIOD_GAIN * (dt - self.period_us);
        }

        self.samples += 1;

        // Phase tracking: nudge the nominal present toward this sample,
        // attenuating the swap-return jitter.  The samples arrive exactly one
        // period apart, so the raw error t - tick is always ~period; wrap it
        // into [-period/2, +period/2] FIRST or the filter would judge every
        // sample "out of range" and resync to it verbatim (i.e. the PLL would
        // do nothing - the old bug made the "smoothing" a raw-anchor no-op,
        // which is why the flicker never improved).  A missed/doubled present
        // shows up as a wrapped error near zero and just doesn't move the
        // phase, which is exactly what we want.
        let mut err = (t as f64 - self.tick_us).rem_euclid(self.period_us);
        let half = self.period_us / 2.0;
        if err > half {
            err -= self.period_us;
        }
        self.tick_us += PRESENT_PHASE_GAIN * err;
        // Ignore the warm-up samples when reporting residual jitter.
        if self.samples >= 10 {
            self.err_sq_sum += err * err;
        }

        if !self.locked_reported && self.samples >= 60 {
            self.locked_reported = true;
            let rms = (self.err_sq_sum / self.samples.saturating_sub(9) as f64).sqrt();
            eprintln!(
                "nvstusb: present PLL locked: period={:.0}us jitter_rms={:.0}us",
                self.period_us, rms
            );
        }
    }
}

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
    /// Method 5 (paced) fallback stream, if active.
    paced: Option<PacedStream>,
    /// Frame-present anchor feed by the render loop (see [`PresentAnchor`]).
    present: Arc<PresentAnchor>,
    /// Native-Wayland presentation-time anchor (see [`wayland::WaylandPresent`]),
    /// fed by the Wayland backend's `wp_presentation_feedback`. On a composited
    /// desktop this is the only clock phase-locked to the *surface's* present,
    /// so the paced stream prefers it over the DRM/OML anchors.
    wayland_present: Option<Arc<wayland::WaylandPresent>>,
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
        // Method 5 (paced): with the DRM vblank anchor, this is the offset of
        // the shutter IR fire from the display boundary; start at the boundary
        // and sweep with ,/. from there.
        5 => 0,
        _ => 2080,
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
        paced: None,
        present: Arc::new(PresentAnchor::new()),
        wayland_present: None,
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

        // Method 5: (re)start the paced packet stream now that the emitter is
        // configured and enabled; it needs the rate for the frame period.
        if self.vblank_method == 5 {
            self.start_paced();
        }
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

    /// Host-side post-swap phase delay in microseconds.
    pub fn swap_phase_us(&self) -> u32 {
        self.swap_phase_us
    }

    /// Sets the host-side post-swap phase delay (us). The delay runs after
    /// swap_buffers returns so the next eye packet's IR fire lands just before
    /// the frame boundary.  For method 5 (paced) it is the offset of the paced
    /// packet stream against the render loop's present instants, retargeted
    /// live so `,`/`.` tuning moves the IR frames against the display without
    /// restarting the stream.
    pub fn set_swap_phase_us(&mut self, us: u32) {
        self.swap_phase_us = us;
        if let Some(p) = self.paced.as_ref() {
            p.phase_us.store(us as i64, Ordering::Relaxed);
        }
    }

    /// Records the frame-present instant.  Call right after the swap (present)
    /// returns.  The paced stream (method 5) locks its packet schedule to
    /// these instants so the shutter phase can be tuned against real frame
    /// boundaries.
    pub fn notify_present(&self) {
        self.present.notify();
    }

    /// Installs the native-Wayland presentation-time anchor so the paced
    /// stream (method 5) locks its packet schedule to the compositor's
    /// `wp_presentation_feedback` ground truth instead of the DRM/OML anchors,
    /// which do not correlate with a composited surface's present. Call with
    /// the `WaylandDisplay.present` anchor once the Wayland backend is up.
    pub fn set_wayland_present(&mut self, present: Arc<wayland::WaylandPresent>) {
        self.wayland_present = Some(present);
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
            // GLX_SGI_video_sync: on X11 this can wait for vblank, but on
            // native Wayland glXWaitVideoSyncSGI is a no-op (observed: returns
            // in ~0us), so the packet position within the frame was random and
            // the IR fired ~5ms too early.  We instead anchor to swap_buffers
            // itself (which blocks until the compositor presents) and push the
            // shutter packet one frame ahead with a host-side phase delay so
            // the IR lands ~1-2ms before the frame boundary.  Tune with ,/. ,
            // coarse with [ ].
            1 => {
                self.set_eye(eye);
                swap_func();
                if self.swap_phase_us > 0 {
                    precise_sleep(self.swap_phase_us as u64);
                }
            }
            // __GL_SYNC_TO_VBLANK is defined: the driver does the syncing.
            2 => {
                swap_func();
                self.set_eye(eye);
            }
            // Paced fallback: a dedicated thread emits the alternating eye
            // packets on a monotonic clock (see start_paced), so the RP2040
            // master lock and steady 120 Hz IR are guaranteed regardless of
            // how (or whether) the present blocks on a real vblank.  The
            // render loop only presents here; phase is tuned with ,/./[ ].
            5 => {
                swap_func();
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

    /// Starts the paced eye-packet stream (method 5).  A dedicated thread
    /// emits alternating left/right packets on a monotonic clock at the
    /// configured refresh period, so the RP2040 sees the steady 7600-9000 us
    /// alternating stream it requires to lock master mode and fire IR at the
    /// display rate - even when the render loop's present does not block on a
    /// real vblank (composited/Wayland desktops).  Idempotent; the stream
    /// re-anchors on the current phase whenever it is retargeted.
    fn start_paced(&mut self) {
        if let Some(p) = self.paced.as_ref() {
            if p.running.load(Ordering::SeqCst) {
                return;
            }
        }

        let period_us = (1_000_000.0 / self.rate as f64).max(1.0) as u64;
        // Clamp to the emitter's allowed packet period (7600-9000 us): a burst
        // below 7600 us would be rejected by the RP2040's master-lock filter,
        // and anything over 9000 stretches the idle timeout.
        let period_us = period_us.clamp(7600, 9000);
        let running = Arc::new(AtomicBool::new(true));
        let phase_us = Arc::new(AtomicI64::new(self.swap_phase_us as i64));

        // The context is only touched from this thread while it exists (it is
        // stopped before the context is dropped), mirroring the unsynchronized
        // sharing of the stereo thread and the original C.
        struct SendPtr(*mut NvstusbContext);
        unsafe impl Send for SendPtr {}
        impl SendPtr {
            fn get(&self) -> *mut NvstusbContext {
                self.0
            }
        }
        let ctx_ptr = SendPtr(self as *mut NvstusbContext);

        let run = running.clone();
        let ph = phase_us.clone();
        let pr = self.present.clone();
        // The native-Wayland presentation-time anchor, if installed. Highest
        // priority: it is phase-locked to *our surface's* present, which is the
        // only clock that correlates with the composited frame boundary.
        let wl = self.wayland_present.clone();

        // Preferred anchor: the kernel's real display vblank clock
        // (DRM_IOCTL_WAIT_VBLANK on the active CRTC).  This is the ONLY clock
        // phase-locked to the physical image alternation on the screen.  On
        // composited desktops the swap return / present instant is decoupled
        // from the vblank, so anchoring the packets there (as before) made the
        // `,`/`.` phase sweep land at arbitrary points of the frame - which is
        // why no phase ever separated the two views.  With the vblank anchor
        // the sweep is deterministic against the actual image switch.
        let drm = DrmVblank::open();
        if drm.is_some() {
            eprintln!("nvstusb: paced stream anchored to DRM vblank");
        }

        // Anchor 1.5: sync-control MSC/UST read off the app's own GLX/EGL
        // context (captured here on the main thread while winit's context is
        // current), tracking the CRTC vblank clock without any /dev/dri access.
        // Created here rather than on the paced thread: capture() reads the
        // *current* context's display/drawable, which only exists on this
        // thread during init.
        let mut oml = if drm.is_none() {
            let msc = x11glx::MscAnchor::capture();
            if msc.is_some() {
                eprintln!("nvstusb: paced stream anchored to OML sync control");
            }
            msc
        } else {
            None
        };

        let thread = thread::spawn(move || {
            let ctx = unsafe { &mut *ctx_ptr.get() };
            let mut drm = drm;
            let period = Duration::from_micros(period_us);
            let mut phase = ph.load(Ordering::Relaxed).max(0) as u64;
            let mut next = Instant::now() + Duration::from_micros(phase);
            let mut eye = Eye::Left;
            // Wayland presentation anchor: last generation consumed, so a new
            // present is waited on every iteration rather than re-reporting the
            // same one in a busy spin.
            let mut wl_gen: u64 = 0;
            while run.load(Ordering::SeqCst) {
                let now = Instant::now();

                // Anchor 1 (preferred on composited desktops): the surface's
                // own `wp_presentation_feedback`.  `wait_next` blocks until the
                // render loop's present is confirmed by the compositor, so we
                // send the packet for the *next* boundary after that present,
                // at `present + period + phase`, and the IR (packet + 3000 us
                // alarm delay) fires exactly at that boundary.  This is the
                // only anchor phase-locked to *our surface's* frame switch on
                // a Wayland desktop; a `None` return means the frame was
                // discarded, so fall through to the DRM/OML/present anchors.
                if let Some(w) = wl.as_ref() {
                    if let Some(b) = w.wait_next(&mut wl_gen) {
                        let p_us = w.period_us().max(7600);
                        let mut target =
                            b + Duration::from_micros(p_us) + Duration::from_micros(phase)
                                - Duration::from_micros(IR_ALARM_DELAY_US);
                        while target <= Instant::now() {
                            target += Duration::from_micros(p_us);
                        }
                        let remain = target.saturating_duration_since(Instant::now());
                        if remain.as_micros() > 2000 {
                            thread::sleep(remain - Duration::from_millis(1));
                        }
                        spin_until(target);
                        ctx.set_eye(eye);
                        eye = match eye {
                            Eye::Left => Eye::Right,
                            _ => Eye::Left,
                        };
                        let new_phase = ph.load(Ordering::Relaxed).max(0) as u64;
                        if new_phase != phase {
                            phase = new_phase;
                        }
                        continue;
                    }
                    eprintln!("nvstusb: wayland frame discarded; using DRM anchor");
                }

                // Anchor 2 (preferred on KMS): the real display boundary.  Block for
                // the vblank that just occurred, then send the packet for the
                // *next* boundary so the IR (packet + 3000 us alarm delay)
                // fires at `next_boundary + phase`.  `phase` therefore sweeps
                // the shutter switch deterministically across the image switch
                // with the `,`/`.` keys.  Consecutive packets stay ~one period
                // apart because we re-anchor off a fresh kernel vblank every
                // iteration.
                if let Some(d) = drm.as_mut() {
                    if let Some(b) = d.wait_vblank() {
                        let p_us = d.period_us().max(7600);
                        let mut target = d.instant_of(b)
                            + Duration::from_micros(p_us)
                            + Duration::from_micros(phase)
                            - Duration::from_micros(IR_ALARM_DELAY_US);
                        while target <= Instant::now() {
                            target += Duration::from_micros(p_us);
                        }
                        let remain = target.saturating_duration_since(Instant::now());
                        if remain.as_micros() > 2000 {
                            thread::sleep(remain - Duration::from_millis(1));
                        }
                        spin_until(target);
                        ctx.set_eye(eye);
                        eye = match eye {
                            Eye::Left => Eye::Right,
                            _ => Eye::Left,
                        };
                        let new_phase = ph.load(Ordering::Relaxed).max(0) as u64;
                        if new_phase != phase {
                            phase = new_phase;
                        }
                        continue;
                    }
                    drm = None;
                    eprintln!("nvstusb: DRM vblank pacing failed; using OML sync anchor");
                }

                // Anchor 1.5: OML sync control read off the app's own GLX/EGL
                // context (captured at startup), used when no /dev/dri vblank
                // clock is available.  Same schedule as Anchor 1, timed off
                // the CRTC master counter.
                if let Some(m) = oml.as_mut() {
                    if let Some(b) = m.wait_vblank() {
                        let p_us = m.period_us().max(7600);
                        let mut target =
                            b + Duration::from_micros(p_us) + Duration::from_micros(phase)
                                - Duration::from_micros(IR_ALARM_DELAY_US);
                        while target <= Instant::now() {
                            target += Duration::from_micros(p_us);
                        }
                        let remain = target.saturating_duration_since(Instant::now());
                        if remain.as_micros() > 2000 {
                            thread::sleep(remain - Duration::from_millis(1));
                        }
                        spin_until(target);
                        ctx.set_eye(eye);
                        eye = match eye {
                            Eye::Left => Eye::Right,
                            _ => Eye::Left,
                        };
                        let new_phase = ph.load(Ordering::Relaxed).max(0) as u64;
                        if new_phase != phase {
                            phase = new_phase;
                        }
                        continue;
                    }
                    oml = None;
                    eprintln!("nvstusb: OML sync pacing failed; using present anchor");
                }

                // Anchor 2 (fallback): lock to the render loop's present clock
                // when it is fresh (the last present happened within the last
                // two periods).  The anchor's PLL tracks the *mean* present
                // phase and predicts the next present, so the packet schedule
                // doesn't chase every swap-return outlier (which previously
                // smeared the `,`/`.` null band into alternating flicker).
                // `phase` sweeps deterministically against the smoothed
                // present grid.  When no present is known (startup, render
                // stalled) fall back to the previous schedule so the RP2040
                // master lock is held.
                let target = if let Some(lp) = pr.last() {
                    if now.duration_since(lp) <= Duration::from_micros(2 * period_us) {
                        let mut t = pr.next().unwrap_or(lp + period) + Duration::from_micros(phase);
                        while t <= now {
                            t += period;
                        }
                        t
                    } else {
                        if next <= now {
                            next = now + period;
                        }
                        next
                    }
                } else {
                    if next <= now {
                        next = now + period;
                    }
                    next
                };

                // Sleep most of the wait, spin only the tail: keeps the packet
                // deadline exact (thread::sleep overshoots by hundreds of us)
                // without pegging a CPU core for the whole frame period.
                let remain = target.saturating_duration_since(Instant::now());
                if remain.as_micros() > 2000 {
                    thread::sleep(remain - Duration::from_millis(1));
                }
                spin_until(target);
                ctx.set_eye(eye);
                eye = match eye {
                    Eye::Left => Eye::Right,
                    _ => Eye::Left,
                };

                // Live phase retarget (','/'.' keys): recompute the next target
                // from the (new) phase on the next iteration immediately.
                let new_phase = ph.load(Ordering::Relaxed).max(0) as u64;
                if new_phase != phase {
                    phase = new_phase;
                    continue;
                }
                next = target + period;
            }
        });
        self.paced = Some(PacedStream {
            running,
            phase_us,
            thread: Some(thread),
        });
    }

    /// Stops the paced eye-packet stream (method 5) and waits for it to exit.
    fn stop_paced(&mut self) {
        if let Some(p) = self.paced.take() {
            p.running.store(false, Ordering::SeqCst);
            if let Some(thread) = p.thread {
                let _ = thread.join();
            }
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
        self.stop_paced();
        self.stop_stereo_thread();
    }
}
