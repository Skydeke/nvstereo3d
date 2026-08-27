//! `nvstusb-host` — Linux host helper for the NVIDIA 3D Vision USB IR emitter.
//!
//! wiz3D's `Nvidia3DOutput.dll` (running under Wine/Proton) pushes one
//! eye-swap command per presented frame into a shared-memory ring
//! (`/tmp/nvstusb.shm`, exposed to Wine as `Z:\tmp\nvstusb.shm`) and sends a
//! non-blocking one-byte UDP datagram to wake us.  This helper owns the USB
//! emitter and fires the shutter packet in step with the display.
//!
//! ## Why the eye comes from a strict alternator, not from the ring
//!
//! Earlier builds popped each fired eye straight off the ring ("FIFO
//! content-follow"): the Nth queued swap was fired into the Nth vblank slot.
//! That scheme assumes the helper can know WHICH display slot each queued
//! swap's frame scans out in.  It cannot:
//!
//! 1. The DLL's swap stream is a *synthetic* strict alternation.
//!    `CBaseSwapChain::PresentData()` (SHUTTER_MODE_SIMPLE) calls
//!    `Output(true); Present; Output(false); Present` unconditionally, so the
//!    ring bytes are L,R,L,R,... by construction -- regardless of what the
//!    compositor actually scanned out.  Order carries zero phase information.
//! 2. With vsync-blocking presents (the normal case under Wine), a swap lands
//!    in the ring only AFTER its frame has already flipped: Present returns
//!    just past the boundary its frame appeared at.  Every fire deadline we
//!    own (~3 ms BEFORE the next boundary) precedes the arrival of the eye
//!    that belongs to it.  Popping the oldest queued swap therefore fires each
//!    eye one slot late -- a persistent inversion -- and scheduler/compositor
//!    jitter around the drain points makes correctness flicker between
//!    correct and inverted: both frames visible in both eyes.
//! 3. When presents do NOT block (SyncInterval=0 games, coalescing), swaps
//!    arrive bursty or half-rate relative to the display and NO fixed
//!    queue-to-slot offset exists at all.
//!
//! Meanwhile the *timing* half of the problem is solved and proven (the 3dv3d
//! demo shutters perfectly): anchor every packet to the display engine's real
//! vblank clock via `DRM_IOCTL_WAIT_VBLANK` (see `drm.rs`), pre-firing each
//! packet ~ALARM_DELAY_US before its target boundary.  The glasses lock when
//! consecutive packets strictly alternate at a stable ~120 Hz period --
//! exactly what a strict alternator on that grid produces, forever, with no
//! dependence on the game's present behaviour.
//!
//! So this build fires the STRICT ALTERNATOR as the eye source and uses the
//! DLL's stream for the one thing it reliably measures: HOW MANY presents
//! happened.  Each dropped or doubled present shifts `(swaps seen - packets
//! fired)` permanently; once such a step persists for REALIGN_HOLD_SWAPS
//! swaps, we re-align the alternator with a single same-eye pair (one-frame
//! hitch, like the old alternator build's correction).  A sustained rate
//! mismatch (content not landing one-per-boundary) is NOT chased -- flipping
//! polarity periodically would be worse than the disease -- it is latched and
//! reported loudly instead, because that failure must be fixed upstream
//! (force SyncInterval=1 / full-rate presents); no emitter-side trick can
//! shutter content whose eyes don't alternate at the display rate.
//!
//! A constant +/-1 SLOT offset between the two streams stays invisible to
//! every in-band check (both streams remain self-consistent under a whole-
//! stream shift), so the emitter's 3D button toggles manual eye inversion --
//! press it whenever depth perception says the eyes are swapped.

use std::env;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use crate::nvstusb::drm;
use crate::nvstusb::usb;
use crate::shm;
use crate::shm::{FLAG_EMITTER_PRESENT, FLAG_FIRMWARE_LOADED, Shm,
                STATUS_ERROR, STATUS_OPENING, STATUS_READY};

/// Embedded firmware image (must match the one shipped with `3dv3d`).
const FIRMWARE: &[u8] = include_bytes!("../firmware/nvstusb.fw");

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Per-second telemetry for the fired eye stream (see main loop).  The glasses
/// only lock into master mode when consecutive packets strictly alternate
/// L/R and the period is inside the emitter's lock window.
#[derive(Default)]
struct StreamStats {
    last_dt: Option<std::time::Instant>,
    last_right: bool,
    count: u64,
    alternating: u64,
    same_eye: u64,
    min_period: u64,
    max_period: u64,
    sum_period: u64,
    period_n: u64,
    last_report: Option<std::time::Instant>,
}

const MASTER_LOCK_MIN_US: u64 = 7600;
const MASTER_LOCK_MAX_US: u64 = 9000;

/// Fixed firmware packet -> IR alarm delay (see `drm.rs`), used to pre-fire
/// each eye packet so the shutter opens exactly when the frame presents.
const ALARM_DELAY_US: u64 = 3_000;
/// Default extra lead folded into the pre-fire busy-wait (see
/// [`fire_lead_us`]).
const DEFAULT_HOST_LEAD_US: u64 = 250;

/// Pre-fire lead in microseconds (`NVSTUSB_HOST_LEAD_US` overrides).  The IR
/// flip lands roughly this many microseconds before the target vblank
/// timestamp minus the USB write time.  Too large and the shutter switches
/// while the previous eye's frame is still scanning out: the tail of the old
/// frame leaks into the new eye exactly like a mis-tuned phase in the demo
/// (sweepable there with `,`/`.`).  The demo's working default corresponds to
/// ~100 us before the boundary - just inside blanking; 250 leaves margin for
/// USB write-time jitter.  Sweep this if games show edge ghosting that the
/// demo does not.
fn fire_lead_us() -> u64 {
    env_or("NVSTUSB_HOST_LEAD_US", "250")
        .parse()
        .unwrap_or(DEFAULT_HOST_LEAD_US)
}

impl StreamStats {
    fn record(&mut self, right: bool) {
        let now = std::time::Instant::now();
        if let Some(t) = self.last_dt {
            let period = now.duration_since(t).as_micros() as u64;
            if self.period_n == 0 || period < self.min_period {
                self.min_period = period;
            }
            if period > self.max_period {
                self.max_period = period;
            }
            self.sum_period += period;
            self.period_n += 1;
            if right != self.last_right {
                self.alternating += 1;
            } else {
                self.same_eye += 1;
            }
        }
        self.count += 1;
        self.last_dt = Some(now);
        self.last_right = right;
    }

    /// Median packet period (us) of the fired stream; 0 until we have samples.
    fn median_period_us(&self) -> u64 {
        if self.period_n == 0 {
            return 0;
        }
        self.sum_period / self.period_n
    }

    #[allow(clippy::too_many_arguments)]
    fn tick(
        &mut self,
        corrections: u64,
        swaps_this_window: &mut u64,
        batches_ge2: &mut u64,
        max_batch: &mut usize,
        lead_n: &mut u64,
        lead_sum: &mut u64,
        lead_min: &mut u64,
        lead_max: &mut u64,
    ) {
        let now = std::time::Instant::now();
        let report = match self.last_report {
            None => true,
            Some(t) => now.duration_since(t).as_secs() >= 1,
        };
        if !report || (self.count == 0 && *swaps_this_window == 0) {
            return;
        }
        let avg = self.median_period_us();
        let in_window = self.count > 0
            && self.min_period >= MASTER_LOCK_MIN_US
            && self.max_period <= MASTER_LOCK_MAX_US
            && self.same_eye == 0;
        // Content health: while we fire one packet per display slot, wiz3D
        // must present (enqueue) at roughly the same rate. Anything far below
        // that means the screen is not showing a fresh eye every slot, which
        // no emitter timing can compensate.
        let starved = self.count >= 60 && *swaps_this_window * 2 < self.count as u64;
        let warn = if starved {
            format!(
                "*** SWAP STARVATION: {} swaps/s vs {} packets/s -- Nvidia3DOutput \
                 is presenting far below display rate (game paused/menu? stereo \
                 disengaged -> mono fallback? presents coalescing?). Glasses are \
                 locked but content cannot follow ***",
                *swaps_this_window, self.count
            )
        } else if in_window {
            "IN-LOCK-WINDOW".to_string()
        } else {
            "*** glasses will NOT lock ***".to_string()
        };
        let mut extra = String::new();
        if *batches_ge2 > 0 {
            extra.push_str(&format!(
                " | bursty-swaps {}x>=2 (max batch {})",
                *batches_ge2, *max_batch
            ));
        }
        if self.period_n > 0 {
            let lead_avg = if *lead_n > 0 { *lead_sum / *lead_n } else { 0 };
            let lead_str = if *lead_n > 0 {
                format!(
                    " | swap-lead {}us (min {}, max {})",
                    lead_avg, *lead_min, *lead_max
                )
            } else {
                String::new()
            };
        // Content fps: SIMPLE mode enqueues two swaps per game frame.
        let content_fps = *swaps_this_window / 2;
        eprintln!(
            "nvstusb-host: {}/s packets | {}/s swaps (content~{}fps) | period {}-{}us (avg {}) | alternating {}/{} | phase-corrections {}{extra}{lead_str} | {warn}",
            self.count, swaps_this_window, content_fps, self.min_period, self.max_period, avg, self.alternating, self.count, corrections,
        );
        } else {
            eprintln!(
                "nvstusb-host: 0 packets | {}/s swaps{extra} | {warn}",
                swaps_this_window
            );
        }
        self.count = 0;
        self.alternating = 0;
        self.same_eye = 0;
        self.min_period = 0;
        self.max_period = 0;
        self.sum_period = 0;
        self.period_n = 0;
        self.last_dt = None;
        self.last_report = Some(now);
        // Window-scoped counters live here so they accumulate a FULL second
        // (resetting them per loop iteration would only ever count one slot).
        *swaps_this_window = 0;
        *batches_ge2 = 0;
        *max_batch = 0;
        *lead_n = 0;
        *lead_sum = 0;
        *lead_min = 0;
        *lead_max = 0;
    }
}

/// Advances the strict alternator after emitting `fire_eye`.  When a polarity
/// re-alignment is pending, we do NOT toggle so the next slot shares the same
/// eye (a single same-eye pair); that shifts the alternation's base by one
/// and re-aligns the shutter with the game's content without breaking the
/// strict cadence past that one pair.
fn advance_eye(fire_eye: &mut bool, pending_flip: &mut bool) {
    if *pending_flip {
        *pending_flip = false;
    } else {
        *fire_eye = !*fire_eye;
    }
}

/// Tracks the DLL's present stream against our fire count and decides when
/// the alternator needs a one-pair re-alignment.
///
/// The only trustworthy invariant is the COUNT relationship: the DLL enqueues
/// exactly one swap per Output() call, and wiz3D presents twice per game
/// frame, so while everything is healthy `swaps_seen - fires_fired` is
/// CONSTANT (a small nonzero value -- a swap is always observed shortly
/// before its slot's fire).  A dropped or doubled present makes the delta
/// STEP to a new value; once it has stopped moving for
/// [`REALIGN_HOLD_SWAPS`] swaps the step is confirmed.  Only an ODD total
/// step shifts eye parity (an even number of dropped/doubled presents leaves
/// L/R alignment intact), so the alternator re-aligns with a single same-eye
/// pair exactly then; every confirmed step becomes the new baseline either
/// way.
///
/// Sustained RATE mismatches (presents trickling or bursting at other than
/// display rate) keep the delta moving, so this tracker goes quiet there and
/// the per-second telemetry reports the starvation instead -- flipping
/// polarity periodically to chase a rate mismatch would be worse than the
/// disease.
struct SlipTracker {
    /// Aligned value of `swaps_seen - fires_fired`.
    baseline: i64,
    /// Last delta value seen (to detect that it stopped moving).
    prev_delta: i64,
    /// `swaps_seen` at which the delta settled on `prev_delta`.
    stable_since: Option<u64>,
}

impl SlipTracker {
    fn new() -> Self {
        // `prev_delta = MIN` guarantees the very first sample takes the seed
        // branch instead of being read as a step.
        Self {
            baseline: 0,
            prev_delta: i64::MIN,
            stable_since: None,
        }
    }
}

/// How many swaps a new delta must persist before we treat it as a confirmed
/// slip. ~200 ms at 120 Hz: long enough to ride out one missed drain hiccup,
/// short enough that a real slip costs under a quarter second of inversion.
const REALIGN_HOLD_SWAPS: u64 = 24;
/// Steps larger than this are resyncs after starvation, not single events:
/// adopt them silently instead of spending a correction pair.
const MAX_SLIP_STEP: i64 = 8;

impl SlipTracker {
    /// Folds one batch of freshly drained swaps in.  `fires_fired` is the
    /// caller's packet counter; `corrections` counts performed re-alignments.
    fn observe(
        &mut self,
        batch_len: usize,
        fires_fired: u64,
        swaps_seen: &mut u64,
        corrections: &mut u64,
        pending_flip: &mut bool,
    ) {
        for _ in 0..batch_len {
            *swaps_seen += 1;
            let delta = *swaps_seen as i64 - fires_fired as i64;
            if let Some(s0) = self.stable_since {
                if delta == self.prev_delta {
                    if swaps_seen.wrapping_sub(s0) >= REALIGN_HOLD_SWAPS {
                        // Delta settled on this value: a confirmed step.
                        let step = delta - self.baseline;
                        if step != 0 {
                            if step.abs() <= MAX_SLIP_STEP && step & 1 == 1 {
                                // Odd step: eye parity inverted -> re-align the
                                // alternator base with one same-eye pair.
                                *pending_flip = true;
                                *corrections += 1;
                            }
                            self.baseline = delta;
                        }
                        self.stable_since = Some(*swaps_seen);
                    }
                } else {
                    self.prev_delta = delta;
                    self.stable_since = Some(*swaps_seen);
                }
            } else {
                // Very first observation seeds the baseline: whatever phase
                // lead the streams start with is healthy by definition.
                self.baseline = delta;
                self.prev_delta = delta;
                self.stable_since = Some(*swaps_seen);
            }
        }
    }
}

pub fn run() {
    let shm_path = env_or("NVSTUSB_SHM_PATH", shm::DEFAULT_SHM_PATH);
    let port: u16 = env_or("NVSTUSB_HOST_PORT", "8777")
        .parse()
        .unwrap_or(8777);
    // How long the wake socket blocks before re-checking the ring (a lost or
    // coalesced wake is still picked up within this interval).
    let poll_timeout_ms: u64 = env_or("NVSTUSB_POLL_MS", "5").parse().unwrap_or(5);

    // --- Shared memory -----------------------------------------------------
    let mut shm = match Shm::open(&shm_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("nvstusb-host: cannot open shm '{}': {e}", shm_path);
            std::process::exit(1);
        }
    };
    if shm.cap() == 0 {
        // Fresh or invalid region: stamp it.  (If the DLL got there first with
        // a valid header, cap() would already be nonzero and we keep its state.)
        shm.init_region();
    }
    eprintln!("nvstusb-host: shared region '{}' ready (cap {})", shm_path, shm.cap());
    shm.set_status(STATUS_OPENING);

    // --- Wake socket -------------------------------------------------------
    let wake_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
    let sock = match UdpSocket::bind(wake_addr) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("nvstusb-host: cannot bind wake socket {wake_addr}: {e}");
            std::process::exit(1);
        }
    };
    sock.set_read_timeout(Some(Duration::from_millis(poll_timeout_ms)))
        .ok();
    eprintln!("nvstusb-host: wake socket listening on {wake_addr}");

    // --- USB emitter -------------------------------------------------------
    let ctx = match usb::usb_init() {
        Some(c) => c,
        None => {
            eprintln!(
                "nvstusb-host: libusb init failed (run as root or via the 98-nvstusb.rules udev rule, on a host with USB)"
            );
            std::process::exit(1);
        }
    };

    let mut device: Option<usb::UsbDevice> = None;
    let mut rate_hz: f32 = 120.0;
    let delay_us: u32 = 0;

    // Initial refresh rate from the DLL (or default).
    let shm_rate = shm.rate_hz();
    if shm_rate > 60.0 {
        rate_hz = shm_rate;
    }

    // Try to (re)open the device until it appears; if we already had it and it
    // dies, drop it and retry on the next iteration.
    if device.is_none() {
        if let Some(d) = usb::open_device(ctx, FIRMWARE) {
            match d.configure(rate_hz) {
                Ok(()) => {
                    let mut flags = FLAG_EMITTER_PRESENT;
                    // Report firmware load status: open_device loads it when the
                    // device reports zero endpoints, so we can't know for sure;
                    // treat the device being present and configured as ready.
                    flags |= FLAG_FIRMWARE_LOADED;
                    shm.set_flags(flags);
                    shm.set_status(STATUS_READY);
                    device = Some(d);
                    eprintln!("nvstusb-host: emitter ready at {rate_hz:.1} Hz");
                }
                Err(e) => {
                    eprintln!("nvstusb-host: configure failed: {e}");
                    shm.set_status(STATUS_ERROR);
                }
            }
        } else {
            shm.set_status(STATUS_ERROR);
        }
    }

    // --- DRM vblank anchor ------------------------------------------------
    // Anchors each eye packet to the display engine's real refresh clock,
    // exactly the mechanism 3dv3d relies on to shutter under a compositor.
    // If unavailable (no /dev/dri vblank), we fall back to firing on the
    // game's presents directly.
    //
    // NVSTUSB_ANCHOR_OUTPUT=<connector> (e.g. DP-2) binds the anchor to that
    // specific head instead of the first usable one.  On multi-head setups
    // every CRTC free-runs with its own phase offset, so the anchor MUST be
    // the monitor that actually displays the game -- a mismatch shows up as a
    // constant wrong-eye bias no button press can fix reliably.
    let pref_connector: Option<String> = std::env::var("NVSTUSB_ANCHOR_OUTPUT")
        .ok()
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty());
    let mut drm_anchor = drm::DrmVblank::open_preferring(pref_connector.as_deref(), None);
    if pref_connector.is_some() {
        eprintln!(
            "nvstusb-host: anchor output preference: {:?}",
            pref_connector
        );
    }

    // --- Main loop --------------------------------------------------------
    let mut last_rate = rate_hz;
    let mut last_delay = delay_us;
    let mut buf = [0u8; 64];

    // Pre-boundary fire lead (NVSTUSB_HOST_LEAD_US).  Fixed for the process
    // lifetime, like the demo's startup phase default.
    let host_lead_us = fire_lead_us();

    // Manual polarity override, toggled by the emitter's 3D button (the
    // host-side equivalent of the demo's `i` key).  A systematic +/-1 slot
    // offset between the game's submit->scanout pipeline depth and our fire
    // grid is invisible to every in-band check -- see the module docs -- so
    // there must be a human switch for it.
    let mut manual_invert = false;

    // Per-second stream telemetry.  The glasses only lock when the packet
    // stream strictly alternates L/R and the period stays inside the RP2040
    // master-lock window (7600-9000 us).  If the DLL is not feeding a valid
    // stream (wrong refresh, dropped presents, jitter), this shows the real
    // numbers instead of us guessing.
    let mut dbg = StreamStats::default();

    // --- Eye source: strict alternator on the vblank grid ------------------
    // See the module docs for why the ring cannot provide the eye.  The
    // alternator produces the one stream the glasses can lock (strictly
    // alternating, stable ~8.3 ms period) and the DLL's present count is used
    // only to detect discrete slips (dropped/doubled presents).
    let mut fire_eye: bool = false; // alternator's next eye (strictly alternates)
    let mut pending_flip = false;   // insert one same-eye pair at next slot
    let mut slip = SlipTracker::new();
    let mut swaps_seen: u64 = 0;    // presents consumed from the ring
    let mut fires_fired: u64 = 0;   // packets emitted
    let mut corrections: u64 = 0;   // polarity re-alignments performed
    let mut last_swap = Instant::now();

    // Burst visibility: how many drains carried >=2 swaps (multiple presents
    // landed since the previous drain -- coalescing/burst signature), and the
    // deepest batch this telemetry window.
    let mut batches_ge2: u64 = 0;
    let mut max_batch: usize = 0;
    let mut swaps_this_window: u64 = 0;

    // Look-ahead diagnostic: how early (in us) do swap arrivals land vs. the
    // vblank grid. Under blocking presents arrivals cluster just AFTER a
    // boundary (small values ~hundreds of us); large spread means the present
    // path is not boundary-locked.
    let mut lead_n: u64 = 0;
    let mut lead_sum: u64 = 0;
    let mut lead_min: u64 = 0;
    let mut lead_max: u64 = 0;

    // Host-epoch of the most recent vblank we waited on (for the lead calc).
    let mut last_vblank_epoch: Option<u64> = None;

    loop {
        // In fallback mode (no DRM anchor) the socket wake drives the loop; in
        // anchored mode the vblank wait below is the pace, so we must NOT
        // block here -- a 5ms recv timeout aliases against the 8.3ms vblank
        // grid and halves the emission to 60Hz.
        if drm_anchor.is_none() {
            let _ = sock.recv_from(&mut buf);
        }

        // (Re)open the device if we don't have one.
        if device.is_none() {
            if let Some(d) = usb::open_device(ctx, FIRMWARE) {
                if d.configure(rate_hz).is_ok() {
                    shm.set_status(STATUS_READY);
                    device = Some(d);
                }
            }
        }

        // Absorb any new swaps the game pushed. A fresh swap means the game is
        // presenting (keeps us alive); it feeds the slip tracker (present
        // COUNT vs our fire count), nothing more -- see the module docs for
        // why the eye itself must come from the alternator.  A second
        // ingestion pass runs right after the vblank wait so swaps pushed mid-
        // slot are counted against the correct fire immediately.
        let batch = shm.drain();
        if !batch.is_empty() {
            if batch.len() >= 2 {
                batches_ge2 += 1;
            }
            max_batch = max_batch.max(batch.len());
            swaps_this_window += batch.len() as u64;
            last_swap = Instant::now();
            slip.observe(
                batch.len(),
                fires_fired,
                &mut swaps_seen,
                &mut corrections,
                &mut pending_flip,
            );
            // Where in the vblank frame this batch of swaps landed: the time
            // remaining to the next vblank is our look-ahead budget.
            if let (Some(anchor), Some(vb)) = (drm_anchor.as_ref(), last_vblank_epoch) {
                let now = anchor.host_epoch_us();
                let p = anchor.period_us();
                if now >= vb && p > 0 {
                    let to_next = p - ((now - vb) % p);
                    lead_n += 1;
                    lead_sum += to_next;
                    if lead_min == 0 || to_next < lead_min {
                        lead_min = to_next;
                    }
                    if to_next > lead_max {
                        lead_max = to_next;
                    }
                }
            }
        }

        if let Some(d) = device.as_ref() {
            // Pick up config changes from the DLL.
            let new_rate = shm.rate_hz();
            if new_rate > 60.0 && (new_rate - last_rate).abs() > 0.01 {
                let _ = d.configure(new_rate);
                last_rate = new_rate;
                rate_hz = new_rate;
            }
            let new_delay = shm.alarm_delay_us();
            if new_delay != last_delay {
                if new_delay > 0 {
                    d.set_alarm_delay_us(new_delay);
                }
                last_delay = new_delay;
            }

            // Stop emitting shortly after the game stops feeding swaps (a dead
            // process must leave the emitter silent, not looping forever).
            let alive = last_swap.elapsed() <= Duration::from_millis(300);

            // Poll the emitter's buttons/wheel (same readback as the demo's
            // get_keys).  The 3D button toggles the manual polarity override:
            // press it whenever depth perception says eyes are swapped.
            // Deliberately placed BEFORE the blocking vblank wait so its USB
            // latency cannot shift the fire deadline.
            {
                let cmd: [u8; 4] = [0x42, 0x18, 0x03, 0x00];
                let _ = d.write_bulk(2, &cmd);
                let mut rb = [0u8; 7];
                let _ = d.read_bulk(4, &mut rb);
                if rb[6] & 0x01 != 0 {
                    manual_invert = !manual_invert;
                    eprintln!(
                        "nvstusb-host: emitter button -> eye inversion {}",
                        if manual_invert { "ON" } else { "OFF" }
                    );
                }
            }

            match drm_anchor.as_mut() {
                // --- Hardware-vblank anchored emission -------------------
                // One packet per real vblank slot, timed to the display vblank
                // clock; the eye is the strict alternation (module docs).
                Some(anchor) if alive => {
                    match anchor.wait_vblank_blocking() {
                        Some(vblank_us) => {
                            last_vblank_epoch = Some(vblank_us);
                            // Late-ingest pass (count accuracy): swaps pushed
                            // since the top-of-loop drain are folded into the
                            // slip tracker HERE, at the boundary itself, so a
                            // present that landed inside this very slot is
                            // attributed to it rather than the next one.
                            let late = shm.drain();
                            if !late.is_empty() {
                                if late.len() >= 2 {
                                    batches_ge2 += 1;
                                }
                                max_batch = max_batch.max(late.len());
                                swaps_this_window += late.len() as u64;
                                last_swap = Instant::now();
                                slip.observe(
                                    late.len(),
                                    fires_fired,
                                    &mut swaps_seen,
                                    &mut corrections,
                                    &mut pending_flip,
                                );
                            }
                            let eye = fire_eye;
                            advance_eye(&mut fire_eye, &mut pending_flip);
                            let alarm = if last_delay > 0 {
                                last_delay as u64
                            } else {
                                ALARM_DELAY_US
                            };
                            let next_present = vblank_us.saturating_add(anchor.period_us());
                            let fire_at = anchor.instant_of(
                                next_present.saturating_sub(alarm + host_lead_us),
                            );
                            while Instant::now() < fire_at {
                                std::hint::spin_loop();
                            }
                            // manual_invert applies at the wire only.
                            let out = eye != manual_invert;
                            d.send_eye(out, rate_hz);
                            dbg.record(out);
                            fires_fired += 1;
                        }
                        None => {
                            // Anchor died: fire immediately to stay live.
                            let eye = fire_eye;
                            advance_eye(&mut fire_eye, &mut pending_flip);
                            let out = eye != manual_invert;
                            d.send_eye(out, rate_hz);
                            dbg.record(out);
                            fires_fired += 1;
                        }
                    }
                }
                // Anchor present but idle (no swaps recently): stay silent.
                Some(_) => {
                    std::thread::sleep(Duration::from_millis(2));
                }
                // --- No DRM anchor: fire on the presents directly. ------
                // Timing quality is limited without the anchor (packets go out
                // on wake arrival); fix the anchor permissions instead of
                // tuning here.  The alternator still keeps the LOCK alive.
                None => {
                    if alive {
                        let eye = fire_eye;
                        advance_eye(&mut fire_eye, &mut pending_flip);
                        let out = eye != manual_invert;
                        d.send_eye(out, rate_hz);
                        dbg.record(out);
                        fires_fired += 1;
                    }
                }
            }

            dbg.tick(
                corrections,
                &mut swaps_this_window,
                &mut batches_ge2,
                &mut max_batch,
                &mut lead_n,
                &mut lead_sum,
                &mut lead_min,
                &mut lead_max,
            );
        }
    }
}
