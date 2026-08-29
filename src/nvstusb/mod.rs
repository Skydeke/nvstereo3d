//! Driver for the NVIDIA 3D Vision USB IR emitter.
//!
//! Port of `lib/nvstusb.c` from the original C project. Talks to the emitter
//! through [`usb`], and keeps the shutter timing in sync with the display by
//! pacing the eye packet to the DRM/KMS kernel vblank clock (see [`drm`]).
#![allow(dead_code)]

pub mod drm;
pub mod usb;

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::gl;
use crate::nvstusb::drm::{normalize_connector_name, DrmVblank};
use crate::nvstusb::usb::{t2_count, timings_block, UsbDevice};

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

/// One per-monitor shutter timing profile in the units 3DVisionActivator's
/// `MonitorTimings.ini` (and NV3D-Lib's `nvtimings.json`) uses, all in
/// microseconds.  The semantics come from the firmware reverse-engineering in
/// libnvstusb (`src/nvstusb.c` register annotations) plus 3DVisionActivator's
/// own documentation:
/// - X: delay from the monitor's refresh start to the shutter open edge (a
///   T0 timer reload — 4 MHz — "timer 0 will be started with this value by
///   timer 2"); the primary band-position knob;
/// - Y: shutter open window, the delay until the eye is turned off ("delay
///   until turning eye off?" — also a T0 reload);
/// - Z: full frame time (== 1e6 / refresh rate), the T2 "timer 2 reload
///   value" that keeps the frame period;
/// - W: a second T2 timer counter ("some timer 2 counter, 1020 is subtracted
///   from this, loaded at startup"); a real register that every per-monitor
///   profile stores, but 3DVisionActivator's author measured "no effect" on
///   the panels he tried — leave it alone unless your monitor visibly needs
///   it.
/// The refresh rate (Z) is fixed per monitor; X/Y/W are the values you tune.
///
/// Genuine nvstusb firmware and the RP2040-style clones both implement this
/// register block (the app probes the read-back to verify X/Y/W writes took
/// effect — [`crate::nvstusb::NvstusbContext::timings_live`]).
#[derive(Clone, Copy, Debug)]
pub struct ShutterTimings {
    pub x_us: f64,
    pub y_us: f64,
    pub z_us: f64,
    pub w_us: f64,
}

impl ShutterTimings {
    /// 1440p @ 120 Hz reference from NV3D-Lib's nvtimings.json — byte-identical
    /// to the `Samsung LC27G5xT 1440p 120Hz` profile shipped with
    /// 3DVisionActivator (`MonitorTimings.ini` [0], EDID SAM28794).
    pub const fn reference() -> Self {
        Self {
            x_us: 0.5,
            y_us: 7334.0,
            z_us: 8333.5,
            w_us: 4735.58,
        }
    }

    /// Parses a 3DVisionActivator `MonitorTimings.ini` (sections like
    /// `RefreshRateHz:`, `X_us:`, `Y_us:`, `Z_us:`, `W_us:`) and returns the
    /// first profile whose refresh rate is within `tol_hz` of `rate_hz` (the
    /// monitor's measured rate — the demo always binds Z to the real refresh).
    /// Profiles are matched by refresh rate rather than EDID_ID because the
    /// app only knows the output/connector name, not the monitor's EDID block.
    pub fn from_ini_file(path: &str, rate_hz: f32, tol_hz: f32) -> Option<ShutterTimings> {
        use std::fs;

        let text = fs::read_to_string(path).ok()?;
        #[derive(Default, Clone, Copy)]
        struct Entry {
            rate: Option<f32>,
            x: Option<f64>,
            y: Option<f64>,
            z: Option<f64>,
            w: Option<f64>,
        }
        let mut entries: Vec<Entry> = Vec::new();
        let mut cur: Option<Entry> = None;
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            // A new profile starts at a `Monitor:` line (or a `[n]` / EDID
            // header). Flush the previous one.
            if line.starts_with("Monitor:") || line.starts_with("EDID_ID:") || line.starts_with('[')
            {
                if let Some(e) = cur.take() {
                    entries.push(e);
                }
                cur = Some(Entry::default());
            } else if let Some(e) = cur.as_mut() {
                for (prefix, slot) in [
                    ("RefreshRateHz:", "rate"),
                    ("X_us:", "x"),
                    ("Y_us:", "y"),
                    ("Z_us:", "z"),
                    ("W_us:", "w"),
                ] {
                    if let Some(rest) = line.strip_prefix(prefix) {
                        let rest = rest.trim();
                        let ok = match slot {
                            "rate" => rest.parse::<f32>().ok().map(|v| e.rate = Some(v)),
                            "x" => rest.parse::<f64>().ok().map(|v| e.x = Some(v)),
                            "y" => rest.parse::<f64>().ok().map(|v| e.y = Some(v)),
                            "z" => rest.parse::<f64>().ok().map(|v| e.z = Some(v)),
                            _ => rest.parse::<f64>().ok().map(|v| e.w = Some(v)),
                        };
                        if ok.is_some() {
                            break;
                        }
                    }
                }
            }
        }
        if let Some(e) = cur.take() {
            entries.push(e);
        }
        entries.into_iter().find_map(|e| {
            let rate_ok = match e.rate {
                Some(r) => (r - rate_hz).abs() <= tol_hz,
                None => return None,
            };
            (rate_ok && e.x.is_some() && e.y.is_some() && e.z.is_some() && e.w.is_some()).then(
                || ShutterTimings {
                    x_us: e.x.unwrap(),
                    y_us: e.y.unwrap(),
                    z_us: e.z.unwrap(),
                    w_us: e.w.unwrap(),
                },
            )
        })
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
    /// Kernel DRM/KMS vblank anchor — the ONLY pacing mechanism.  The IR
    /// packet is timed to the display engine's own vblank clock (method 4).
    drm_vblank: Option<DrmVblank>,
    /// Kernel connector name (e.g. `DP-1`) of the display the window is on,
    /// that the vblank anchor should be bound to.  On a multi-head GPU the
    /// per-head vblank grids share no fixed phase, so an anchor on the wrong
    /// head leaves a constant 0..1-frame shutter error (or, with the swap
    /// coming back on a different grid, a broken frame lock).  Keeping this
    /// in lockstep with the window's wl_output and re-opening the anchor with
    /// it is what keeps the anchor on the RIGHT head.  Set via
    /// [`NvstusbContext::set_target_connector`].
    target_connector: Option<String>,
    /// Whether the eyes are inverted.
    invert_eyes: bool,
    /// Packet lead before the vblank boundary (us), see `swap()`.
    swap_phase_us: u32,
    /// Per-monitor shutter-timing profile (3DVisionActivator / NV3D-Lib
    /// "X/Y/W" model): X = delay from monitor refresh start to the shutter
    /// open edge (us), Y = shutter open window (us), W = unused on most
    /// panels (us).  Z (frame time) always follows [`Self::rate`], so the
    /// refresh is fixed and X/Y/W are the live-tuned values.  Written to the
    /// emitter's 0x2007 timing registers by `set_shutter_timings`.
    timing_x_us: f64,
    timing_y_us: f64,
    timing_w_us: f64,
    /// Whether the emitter was verified to store the 0x2007 shutter-timing
    /// register block (X/Y/W): `None` = not probed yet, `Some(true)` = the
    /// device echoed the block back with the values that were just written
    /// (X/Y/W tuning is live), `Some(false)` = the write-back could NOT be
    /// verified (readback silent or echoed mismatched values — the block may
    /// still be implemented by a device that answers the read late).  A failed
    /// readback is deliberately not labelled "ignored": silence is not proof
    /// that X/Y/W writes have no effect.  See
    /// [`NvstusbContext::timings_live`].
    timings_live: Option<bool>,
}

/// How many swaps to wait for the window's wl_output to be reported before the
/// vblank anchor gives up and does the old first-active-head blind scan.
pub(crate) const ARM_GRACE_SWAPS: u64 = 240;

/// Decides whether the DRM vblank anchor must be re-armed onto the window's
/// output.  Re-arming is needed when the window's output is known and the
/// anchor is bound to a DIFFERENT connector (or to no connector at all - a
/// blind first-active-head bind whose head may not be the window's).
///
/// This is the single most important correctness check for multi-head pacing:
/// on a multi-head GPU every CRTC vblank grid runs at an arbitrary fixed phase
/// to its siblings, so an anchor left on the wrong head free-runs at the same
/// refresh with a stable-but-wrong phase - the shutter flips mid-scanout (a
/// red/blue split on the alternating-colour scene) and, worst case, the frame
/// lock never converges.  A windowed window can be dragged across monitors (or
/// moved by compositor keybinds), so this must be re-evaluated repeatedly.
fn needs_anchor_retarget(anchor_connector: Option<&str>, window_connector: Option<&str>) -> bool {
    let Some(want) = window_connector else {
        // Window's output not yet reported (boot): leave the anchor where it
        // is; `ARM_GRACE_SWAPS` bounds how long we tolerate the possibly-wrong
        // head before the demo re-checks.
        return false;
    };
    match anchor_connector {
        // Anchor explicitly on the window's head (or the connector is
        // unknown/empty but the window's name is too): nothing to do.
        Some(anchor) if anchor == want => false,
        // Anchor is on some OTHER named head -> wrong head, must re-arm.
        Some(_) => true,
        // Anchor bound blind (no connector matched) -> assume wrong until the
        // window's output is learned and proven to match.
        None => true,
    }
}

/// Initializes the controller: opens the USB device (uploading `nvstusb.fw`
/// first if needed) and picks the DRM vblank anchor.
pub fn init() -> Option<NvstusbContext> {
    let usb_context = usb::usb_init()?;
    let device = usb::open_device(usb_context, include_bytes!("../../firmware/nvstusb.fw"))?;

    // DRM/KMS vblank anchor: the ONLY pacing mechanism (method 4). Timed to the
    // display engine's real vblank clock (see drm.rs).
    let drm_vblank = if std::env::var_os("NVSTUSB_DRM").is_some() {
        DrmVblank::open()
    } else {
        None
    };
    if drm_vblank.is_none() {
        eprintln!("nvstusb: no DRM vblank anchor; NVSTUSB_DRM must be set");
    }

    let swap_phase_us = 3100;

    Some(NvstusbContext {
        rate: 0.0,
        eye: Eye::Left,
        device,
        toggled_3d: false,
        drm_vblank,
        target_connector: None,
        invert_eyes: false,
        swap_phase_us,
        timing_x_us: ShutterTimings::reference().x_us,
        timing_y_us: ShutterTimings::reference().y_us,
        timing_w_us: ShutterTimings::reference().w_us,
        timings_live: None,
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
        // `configure` programs the reference shutter timings; re-apply the
        // current (possibly live-tuned, per-monitor) X/Y/W profile so a
        // monitor-change re-config never resets the user's tuning.
        if let Err(e) =
            self.device
                .set_timings_us(rate, self.timing_x_us, self.timing_y_us, self.timing_w_us)
        {
            eprintln!("nvstusb: shutter timing write failed: {e}");
        }

        // Verify the timing registers actually took effect, exactly like the
        // 3DVisionActivator write-back check: the device echoes the 0x2007
        // block when it stores it.  Probed once (with read retries for
        // stale/late responses) — later re-configs do not re-slow startup.
        if self.timings_live.is_none() {
            self.timings_live = Some(self.probe_timings_live(rate));
        }

        self.rate = rate;
    }

    /// Asks the emitter to echo the shutter-timing registers it has stored,
    /// mirroring 3DVisionActivator's post-write verification, then checks that
    /// the echoed X/Y/W/Z reload counts are the ones just written.  Returns
    /// `true` only when the device answered with the 28-byte register block
    /// *and* the echoed values match the last profile written — that is the
    /// evidence that the X/Y/W tuning genuinely reaches the timing generator.
    ///
    /// A `false` result is deliberately reported as "unverified", never as
    /// "ignored": an empty readback proves only that the device did not answer
    /// the probe, not that the 0x2007 block is unimplemented (emitters known
    /// to implement it can still answer late or be shadowed by a stale
    /// control-in response from an earlier session).
    fn probe_timings_live(&self, rate: f32) -> bool {
        // Expected reload counts: `configure`/`set_timings_us` just stored
        // exactly this block (timings_block clamps sub-60 Hz rates to 120 Hz,
        // mirroring configure's own use).
        let block = timings_block(rate, self.timing_x_us, self.timing_y_us, self.timing_w_us);
        let le = |i: usize| {
            (block[i] as i32)
                | (block[i + 1] as i32) << 8
                | (block[i + 2] as i32) << 16
                | (block[i + 3] as i32) << 24
        };
        // w/x/y/z live at payload offsets 0/4/8/20 of the written block.
        let expect = [le(4), le(8), le(12), le(24)];

        match self.device.read_timings_registers() {
            Some(back) => {
                if back == expect {
                    eprintln!(
                        "nvstusb: timing registers read back and verified (w={} x={} y={} \
                         z={} counts); X/Y/W tuning is LIVE on this emitter",
                        back[0], back[1], back[2], back[3]
                    );
                    true
                } else {
                    eprintln!(
                        "nvstusb: timing block read back but does not match what was written \
                         (w={} x={} y={} z={}, expected w={} x={} y={} z={}); X/Y/W \
                         write-back unverified",
                        back[0],
                        back[1],
                        back[2],
                        back[3],
                        expect[0],
                        expect[1],
                        expect[2],
                        expect[3]
                    );
                    false
                }
            }
            None => {
                eprintln!(
                    "nvstusb: no timing-register readback (the 0x2007 block read was tried \
                     several times and always came back empty or short); the timing write-back \
                     could not be verified"
                );
                false
            }
        }
    }

    /// Whether the shutter-timing register block (X/Y/W via
    /// [`Self::set_shutter_timings`]) was verified to be honored by this
    /// emitter: `None` = not probed yet, `Some(true)` = live (the device
    /// echoed the block back with the written values), `Some(false)` =
    /// readback unverified (the block may still be implemented — see
    /// [`Self::probe_timings_live`]).
    pub fn timings_live(&self) -> Option<bool> {
        self.timings_live
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

    /// Per-monitor shutter timing X (us): delay from monitor refresh start to
    /// the shutter open edge.  Primary band-position knob, tuned live with
    /// the demo's `,`/`.`/`[`/`]` keys.
    pub fn timing_x_us(&self) -> f64 {
        self.timing_x_us
    }

    /// Per-monitor shutter timing Y (us): shutter open window.
    pub fn timing_y_us(&self) -> f64 {
        self.timing_y_us
    }

    /// Per-monitor shutter timing W (us): unused on most panels.
    pub fn timing_w_us(&self) -> f64 {
        self.timing_w_us
    }

    /// Programs the emitter's shutter-timing registers (X/Y/W, microseconds)
    /// live and remembers them so a later `set_rate` re-applies them.  This
    /// is the 3DVisionActivator / NV3D-Lib per-monitor tuning step: X = delay
    /// from monitor refresh start to the shutter open edge, Y = shutter open
    /// window, W = unused.  Z (frame time) always follows the refresh rate.
    pub fn set_shutter_timings(&mut self, x_us: f64, y_us: f64, w_us: f64) {
        let rate = if self.rate > 60.0 { self.rate } else { 120.0 };
        self.timing_x_us = x_us;
        self.timing_y_us = y_us;
        self.timing_w_us = w_us;
        if let Err(e) = self.device.set_timings_us(rate, x_us, y_us, w_us) {
            eprintln!("nvstusb: shutter timing write failed: {e}");
        }
    }

    /// Sets the host-side packet lead (us).  Method 1 sends the eye packet
    /// this many microseconds before the next vblank boundary, so the IR
    /// flip lands at `boundary - lead + alarm delay`; `,`/`.` tuning moves
    /// the shutter switch across the image boundary live.
    pub fn set_swap_phase_us(&mut self, us: u32) {
        self.swap_phase_us = us;
    }

    /// Re-targets the DRM vblank anchor to the kernel connector the window is
    /// now on (e.g. `DP-1`), re-opening it with the window's output as the
    /// preferred connector.  On a multi-head GPU the per-head vblank grids are
    /// phase-independent, so this is REQUIRED to keep the shutter locked to the
    /// display actually showing our window - an anchor left on the other head's
    /// grid free-runs at the same refresh but with an arbitrary phase, so the
    /// shutter flips mid-scanout and the frame lock breaks.
    ///
    /// Re-opening swaps in a freshly measured period for the new head (it is
    /// `resync`-able immediately, `frame_start` re-anchors until synced), so
    /// the next frame already runs on the new head's grid.  A no-op when the
    /// connector is unchanged, so the per-frame monitor poll does not churn
    /// the anchor.
    pub fn set_target_connector(&mut self, name: Option<&str>) {
        let norm = name.map(normalize_connector_name);
        if norm == self.target_connector {
            return;
        }
        // Only re-arm when the anchor is actually on the wrong (or no) head.
        // On a single monitor `open()` already binds to the window's output,
        // so the first per-frame poll must NOT cause a needless re-open
        // (a re-open re-measures the period and, at worst, churns the phase).
        let anchor_conn = self
            .drm_vblank
            .as_ref()
            .and_then(|d| d.connector_name())
            .map(normalize_connector_name);
        if !needs_anchor_retarget(anchor_conn.as_deref(), norm.as_deref()) {
            // Anchor already on the right head; just remember the target so
            // the poll does not re-run this branch.
            self.target_connector = norm.clone();
            return;
        }
        let old = self.anchor_name();
        self.target_connector = norm.clone();
        // Keep pacing live even while the anchor re-opens: open on the new
        // head (or fall back to the first active head if the new one cannot
        // be bound), so we never drop to "no-anchor" on a transient monitor
        // report.
        let reopened = norm
            .as_deref()
            .and_then(|_| DrmVblank::open_preferring(norm.as_deref(), None));
        self.drm_vblank = reopened.or_else(|| DrmVblank::open());
        let new = self.anchor_name();
        eprintln!("nvstusb: anchor re-target {:?} -> {new} (was {old})", norm);
    }

    /// Names the active pacing anchor for diagnostics (the DRM/KMS vblank
    /// anchor — always). Includes the anchored display (`DP-1/pipe1`) so a
    /// wrong-head bind is visible in the once-per-second perf report.
    pub fn anchor_name(&self) -> String {
        match self.drm_vblank.as_ref() {
            Some(d) => format!("drm-vblank[{}]", d.label()),
            None => "no-anchor".to_string(),
        }
    }

    /// Selected vblank method — always 4 (DRM/KMS vblank anchor), the only
    /// pacing mechanism.
    pub fn vblank_method(&self) -> u8 {
        4
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

    /// Forces the DRM anchor to re-anchor on the next frame (used after a mode
    /// change, which can shift the vblank phase).
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
    /// supply one; the windowed backend returns `None`.
    pub fn swap<F: FnMut() -> Option<u64>>(&mut self, eye: Eye, _gl: &gl::Gl, mut swap_func: F) {
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
        // Fire the eye the caller asked for.  The windowed path's swap (via
        // glutin `swap_buffers`) presents on the predicted vblank, so no
        // whole-frame eye inversion is needed here (unlike the removed KMS
        // manual page-flip backend, whose flip landed one vblank late).
        self.set_eye(eye);
        let flip_mono_us = swap_func();
        // Prefer the flip event's own hardware timestamp over a
        // post-syscall Instant::now() sample - see `frame_end` for why
        // this removes scheduler/wakeup jitter that was otherwise baked
        // into the phase the emitter is synced to.
        //
        // When the backend supplies no flip clock (the windowed backend
        // returns `None` from `swap_func`), fall back to the anchor's OWN
        // vblank clock - the host-epoch timestamp of the last real vblank
        // on the anchored output (`query_vblank`, the non-master-safe
        // equivalent of a DRM_MODE_PAGE_FLIP_EVENT - page-flip events need
        // DRM master, which a windowed client does not hold).  A real
        // vblank timestamp - not the jittery post-swap `Instant::now()` -
        // is what lets `frame_end`'s per-frame phase lock converge instead
        // of chasing sample noise and drifting off the grid.
        let precise_us = if let (Some(m), Some(d)) =
            (flip_mono_us, self.drm_vblank.as_ref())
        {
            Some(d.host_us_from_mono(m))
        } else {
            self.drm_vblank.as_mut().and_then(|d| d.query_vblank())
        };
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
            }
        }
        if let Some(d) = self.drm_vblank.as_mut() {
            d.frame_end(precise_us);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the multi-head re-target logic.  The demo must
    /// re-arm the DRM vblank anchor onto the output the window is on; an
    /// anchor left on another head of a multi-head GPU free-runs at the same
    /// refresh but arbitrary phase, so the shutter flips mid-scanout and the
    /// frame lock never converges.  This pins the decision that was broken
    /// when the re-target call was dropped from the window-move path.
    #[test]
    fn anchor_retarget_follows_the_window_output() {
        // Window on DP-2 while the anchor binds blind (first active head):
        // HEAD regression - the anchor stayed on the wrong head ("not
        // anchored at all").  Must re-arm.
        assert!(needs_anchor_retarget(None, Some("DP-2")));
        // Window on DP-2, anchor still on the other head DP-1: must re-arm.
        assert!(needs_anchor_retarget(Some("dp-1"), Some("DP-2")));
        // Window on DP-2, anchor (however it binds) on DP-2: nothing to do.
        assert!(!needs_anchor_retarget(Some("DP-2"), Some("DP-2")));
        // Window's output not yet reported (boot): leave the anchor alone.
        assert!(!needs_anchor_retarget(Some("DP-1"), None));
        assert!(!needs_anchor_retarget(None, None));
        // Connector-name spelling variants must still count as "the same head"
        // (normalize is applied before this is called, but pin the exact
        // equality so silent case/alias regressions are caught).
        assert_eq!(normalize_connector_name("DP-2"), normalize_connector_name("dp-2"));
        assert_eq!(normalize_connector_name("DisplayPort-1"), "dp-1");
        assert_eq!(normalize_connector_name("HDMI-1"), "hdmi-a-1");
    }

    /// Pins that the anchor re-target decision is rejection-consistent: the
    /// moment the anchor lands on the correct head it stops re-arming, and any
    /// second head at all forces a re-arm (never a silently-wrong bind).
    #[test]
    fn anchor_retarget_matches_exactly_one_head() {
        let heads = ["DP-1", "DP-2", "HDMI-A-1"];
        for window in heads {
            for anchor in heads {
                let same = normalize_connector_name(window) == normalize_connector_name(anchor);
                assert_eq!(
                    needs_anchor_retarget(Some(anchor), Some(window)),
                    !same,
                    "anchor={anchor} window={window}"
                );
            }
        }
    }

    #[test]
    fn ini_selects_profile_by_refresh_rate() {
        let dir = std::env::temp_dir().join("nvstusb_timings_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("MonitorTimings.ini");
        std::fs::write(
            &path,
            "[0]\n\
             Monitor: Samsung LC27G5xT 1440p 120Hz\n\
             EDID_ID: SAM28794\n\
             RefreshRateHz: 119.998\n\
             X_us: 0.50\n\
             Y_us: 7334.00\n\
             Z_us: 8333.50\n\
             W_us: 4735.58\n\
             [1]\n\
             Monitor: AsusPG248Q 100Hz original\n\
             EDID_ID: AUS24B1\n\
             RefreshRateHz: 99.931\n\
             X_us: 203.25\n\
             Y_us: 8800.00\n\
             Z_us: 10006.92\n\
             W_us: 5204.33\n",
        )
        .unwrap();

        let s = path.to_str().unwrap();
        // 120 Hz monitor picks the Samsung profile...
        let t = ShutterTimings::from_ini_file(s, 120.0, 0.5).unwrap();
        assert_eq!(t.x_us, 0.5);
        assert_eq!(t.y_us, 7334.0);
        assert_eq!(t.z_us, 8333.5);
        assert_eq!(t.w_us, 4735.58);
        // ...100 Hz picks the Asus entry...
        let t = ShutterTimings::from_ini_file(s, 100.0, 0.5).unwrap();
        assert_eq!(t.x_us, 203.25);
        assert_eq!(t.y_us, 8800.0);
        // ...and an unlisted rate matches nothing.
        assert!(ShutterTimings::from_ini_file(s, 144.0, 0.5).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
