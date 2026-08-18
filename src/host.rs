//! `nvstusb-host` — Linux host helper for the NVIDIA 3D Vision USB IR emitter.
//!
//! wiz3D's `Nvidia3DOutput.dll` (running under Wine/Proton) pushes one
//! eye-swap command per presented frame into a shared-memory ring
//! (`/tmp/nvstusb.shm`, exposed to Wine as `Z:\tmp\nvstusb.shm`) and sends a
//! non-blocking one-byte UDP datagram to wake us.  This helper owns the USB
//! emitter, consumes the ring, and fires the shutter eye packet in step with
//! the game's presents.
//!
//! The USB layer is ported verbatim from the confirmed-working `3dv3d`
//! project (see `usb.rs`).
//!
//! Unlike the naive helper (which fired on the game's presents and so wobbled
//! at the present cadence, not the display's), this build anchors each eye
//! packet to the display engine's REAL vblank clock via `DRM_IOCTL_WAIT_VBLANK`
//! (see `drm.rs`, ported from 3dv3d).  *When* each packet fires is fixed to
//! the hardware 120Hz grid — the same mechanism that makes 3dv3d shutter
//! correctly even under a Hyprland compositor.
//!
//! *Which* eye fires is taken directly from the DLL's swap stream: each
//! `EnqueueSwap` is a FIFO entry, popped in order, one per vblank fire (see
//! `pending_eyes` in `run()`). Earlier builds instead ran a free-running L/R
//! alternator and only used the DLL's reports to detect, after the fact, that
//! the alternator's assumed polarity had drifted from the game's actual
//! output (`drift`/`DRIFT_WINDOW`) — real desyncs (a dropped or doubled
//! Present under Wine/Proton) went uncorrected for up to `DRIFT_WINDOW`
//! frames, and the "correction" was itself a one-frame same-eye hitch. FIFO
//! consumption has no such lag: the fired eye always matches whatever the
//! game actually reported for that slot, as long as it arrives before the
//! fire deadline. A slot with nothing queued (game stalled or a present was
//! dropped) is a genuine underrun, not a phase to chase — see `underruns`.
//!
//! The FIFO is ideal while the game presents exactly one frame per vblank
//! slot ("every frame is perfect").  Real Wine/Proton streams are not: a
//! dropped or doubled Present shifts the frame-to-slot alignment, after which
//! the Nth queued eye no longer matches the Nth fire and the FIFO would fire
//! the wrong eye persistently.  A polarity tracker over the swap stream (same
//! `drift`/`DRIFT_WINDOW` scheme as the earlier alternator build) detects that
//! sustained inversion and falls back to the strict L/R alternator — with a
//! single same-eye re-align pair — until the stream proves lock-step again,
//! then returns to FIFO.

use std::collections::VecDeque;
use std::env;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use crate::nvstusb::drm;
use crate::nvstusb::usb;
use crate::shm;
use crate::shm::{EYE_RIGHT, FLAG_EMITTER_PRESENT, FLAG_FIRMWARE_LOADED, Shm,
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
/// Extra CPU/scheduling lead folded into the pre-fire busy-wait.
const HOST_LEAD_US: u64 = 500;

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

    fn tick(
        &mut self,
        corrections: u64,
        underruns: &mut u64,
        dropped_stale: &mut u64,
        lead_n: &mut u64,
        lead_sum: &mut u64,
        lead_min: &mut u64,
    ) {
        let now = std::time::Instant::now();
        let report = match self.last_report {
            None => true,
            Some(t) => now.duration_since(t).as_secs() >= 1,
        };
        if !report || self.count == 0 {
            return;
        }
        let avg = self.median_period_us();
        let in_window = self.count > 0
            && self.min_period >= MASTER_LOCK_MIN_US
            && self.max_period <= MASTER_LOCK_MAX_US
            && self.same_eye == 0;
        let warn = if in_window {
            "IN-LOCK-WINDOW"
        } else {
            "*** glasses will NOT lock ***"
        };
        if self.period_n > 0 {
            let lead_avg = if *lead_n > 0 { *lead_sum / *lead_n } else { 0 };
            let lead_str = if *lead_n > 0 {
                format!(" | swap-lead {}us (min {})", lead_avg, *lead_min)
            } else {
                String::new()
            };
            let stale_str = if *dropped_stale > 0 {
                format!(" | dropped-stale {}", *dropped_stale)
            } else {
                String::new()
            };
            eprintln!(
                "nvstusb-host: {}/s packets | period {}-{}us (avg {}) | alternating {}/{} | underruns/s {} | phase-corrections {}{}{} | {warn}",
                self.count, self.min_period, self.max_period, avg, self.alternating, self.count, underruns, corrections, stale_str, lead_str
            );
        } else {
            eprintln!("nvstusb-host: {}/s packets (1 frame since start)", self.count);
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
        *lead_n = 0;
        *lead_sum = 0;
        *lead_min = 0;
        *underruns = 0;
        *dropped_stale = 0;
    }
}

/// Cap on how many un-fired swap reports we'll hold. Normal operation keeps
/// this at 0-1 (one report drained per vblank fire); a deeper backlog means
/// the DLL is bursty relative to our drain cadence (e.g. a brief host
/// scheduling hiccup delayed a drain call) rather than a real desync, so this
/// is sized generously -- dropping real backlog here manufactures underruns
/// that didn't happen, which is worse than holding a slightly stale queue.
const MAX_QUEUED_EYES: usize = 16;

/// Backup polarity slip window: how many consecutively-inverted swap reports
/// (in net) before we conclude the content-follow alignment has genuinely
/// slipped (a dropped/hitched present) and re-align via the alternator.
const DRIFT_WINDOW: i64 = 16;

/// Backup underrun trigger: how many FIFO underruns (a fire slot where the
/// game reported no swap) must accumulate before we stop content-following and
/// switch to the strict alternator.  A repeated underrun makes the fired stream
/// non-alternating (L,L,R,R), which breaks the glasses lock -- the one thing we
/// must never do.  `ur_counter` only resets once the stream proves healthy
/// (HEALTHY_RESET_FIRES clean pops), so a half-rate stream (60fps on a 120Hz
/// display: underrun/clean alternating) still accumulates and triggers.
const BACKUP_UR_TRIGGER: u64 = 8;
/// Consecutive clean (non-underrun) FIFO pops that prove the game is feeding a
/// proper one-frame-per-slot stream again, resetting the underrun counter.
const HEALTHY_RESET_FIRES: u64 = 16;
/// While the alternator backup is active, fires over which we judge whether the
/// game has returned to full cadence (>= ~1 swap per fire slot).
const BACKUP_RESUME_WINDOW: u64 = 16;
/// Minimum swaps received over BACKUP_RESUME_WINDOW fires to treat the stream
/// as full-rate 120Hz again (60fps yields ~half this).
const BACKUP_RESUME_MIN_SWAPS: u64 = 14;

/// Advances the strict alternator after emitting `*fire_eye`.  When a polarity
/// re-alignment is pending, we do NOT toggle so the next slot shares the same
/// eye (a single same-eye pair); that shifts the alternation's base by one and
/// re-aligns the shutter with the game's content without breaking the strict
/// cadence past that one pair.
fn advance_eye(fire_eye: &mut bool, pending_flip: &mut bool) {
    if *pending_flip {
        *pending_flip = false;
    } else {
        *fire_eye = !*fire_eye;
    }
}

/// Chooses the eye to fire for one vblank slot and maintains the FIFO-vs-
/// alternator backup state.
///
/// FIFO (content-follow) is used while the game feeds a clean one-frame-per-
/// slot stream: the eye is the game's actual report for the slot.  When it
/// stops feeding (an underrun every other slot, e.g. 60fps content on a 120Hz
/// display), repeating `last_fired_eye` keeps the content correct but makes the
/// fired stream non-alternating (L,L,R,R) and the glasses fall out of lock.  A
/// sustained run of underruns therefore hands the eye source over to the strict
/// alternator (`fire_eye`), which always alternates so the lock holds, until
/// the game returns to full cadence.
fn pick_eye(
    pending_eyes: &mut VecDeque<bool>,
    last_fired_eye: &mut bool,
    underruns: &mut u64,
    ur_counter: &mut u64,
    clean_run: &mut u64,
    fire_eye: &mut bool,
    pending_flip: &mut bool,
    backup_active: &mut bool,
    backup_swaps: &mut u64,
    backup_fires: &mut u64,
) -> bool {
    if *backup_active {
        // Alternator active: fire the strict alternation so the glasses keep
        // their lock, and periodically check whether the game is back to full
        // cadence (~1 swap per fire slot) -- if so, resume content-follow.
        *backup_fires += 1;
        if *backup_fires >= BACKUP_RESUME_WINDOW {
            if *backup_swaps >= BACKUP_RESUME_MIN_SWAPS {
                *backup_active = false;
                pending_eyes.clear(); // rebuild a fresh, aligned queue
            }
            *backup_fires = 0;
            *backup_swaps = 0;
        }
        let eye = *fire_eye;
        advance_eye(fire_eye, pending_flip);
        return eye;
    }

    // FIFO (content-follow) active.
    match pending_eyes.pop_front() {
        Some(eye) => {
            // A clean pop proves the game is feeding one frame per slot.
            *clean_run += 1;
            if *clean_run >= HEALTHY_RESET_FIRES {
                *clean_run = 0;
                *ur_counter = 0;
            }
            eye
        }
        None => {
            // Underrun: the game didn't present a frame for this slot.
            // Repeating last_fired_eye is content-correct but breaks strict
            // alternation; a run of these loses the lock, so hand over to the
            // alternator once the stream is proven broken.
            *underruns += 1;
            *ur_counter += 1;
            *clean_run = 0;
            if *ur_counter >= BACKUP_UR_TRIGGER {
                *backup_active = true;
                *backup_fires = 0;
                *backup_swaps = 0;
                pending_eyes.clear();
                let eye = *fire_eye;
                advance_eye(fire_eye, pending_flip);
                return eye;
            }
            *last_fired_eye // repeat: the previous frame is still on screen
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
    let mut drm_anchor = drm::DrmVblank::open();

    // --- Main loop --------------------------------------------------------
    let mut last_rate = rate_hz;
    let mut last_delay = delay_us;
    let mut buf = [0u8; 64];

    // Per-second stream telemetry.  The glasses only lock when the packet
    // stream strictly alternates L/R and the period stays inside the RP2040
    // master-lock window (7600-9000 us).  If the DLL is not feeding a valid
    // stream (wrong refresh, dropped presents, jitter), this shows the real
    // numbers instead of us guessing.
    let mut dbg = StreamStats::default();

    // --- Content-follow queue -----------------------------------------------
    // The DLL enqueues one eye per present, in display order, into a lock-free
    // FIFO (`shm.rs`). Each vblank-anchored fire pops exactly one entry and
    // fires it directly -- no prediction, no polarity tracking. This is
    // correct as long as the game presents at most one frame per real vblank
    // slot (true for a frame-sequential stereo present loop): the Nth queued
    // eye IS the eye for the Nth fire, in order, full stop.
    //
    // `last_fired_eye` is only consulted on an underrun (queue empty at fire
    // time -- the game hasn't presented a new frame for this slot). A missed
    // present does NOT blank the display: the previous flip is still on
    // screen, so the physically correct fallback is to REPEAT last_fired_eye,
    // not flip it. Flipping fires the opposite eye's shutter while the old
    // eye's frame is still showing -- a genuine wrong-eye flash, not just a
    // duplicate-frame stat. The queue resumes driving the eye choice the
    // instant new swaps land, and the repeat naturally yields exactly one
    // same-eye pair per real stall (unavoidable -- the content itself
    // repeated), instead of a wrong-eye flash plus a same-eye pair.
    //
    // That repeat is content-correct but is not a strict alternation, so a
    // *sustained* run of underruns (e.g. 60fps content on a 120Hz display)
    // would push the fired stream to L,L,R,R and the glasses would fall out
    // of lock. `pick_eye` hands the source over to the alternator once the
    // underrun run proves the game is no longer feeding one frame per slot.
    let mut pending_eyes: VecDeque<bool> = VecDeque::with_capacity(MAX_QUEUED_EYES);
    let mut last_fired_eye: bool = false;
    let mut underruns: u64 = 0;
    let mut dropped_stale: u64 = 0;
    let mut last_swap = Instant::now();

    // Backup (host (2).rs style): a strict L/R alternator plus a polarity
    // tracker.  The FIFO above is ideal while the game presents one frame per
    // vblank slot; when a dropped/hitched present slips that alignment the
    // FIFO fires the wrong eye persistently, and when the game drops to half
    // rate (60fps on a 120Hz display) the FIFO underruns every other slot and
    // the fired stream stops alternating (L,L,R,R), breaking the glasses lock.
    // Either failure switches the eye source to the strict alternator until
    // the stream proves full-rate and lock-step again.
    let mut backup_active = false;   // fire the alternator, not the FIFO
    let mut fire_eye: bool = false;  // alternator's next eye (strictly alternates)
    let mut base: Option<bool> = None; // tracked L/R polarity of the game
    let mut drift: i64 = 0;          // net agreement of swaps vs. our base
    let mut corrections: u64 = 0;    // polarity re-alignments performed
    let mut pending_flip = false;    // insert one same-eye pair at next slot
    let mut swapped_seen: u64 = 0;   // presents consumed (parity reference)
    // Backup entry/exit bookkeeping (see `pick_eye`).
    let mut ur_counter: u64 = 0;     // FIFO underruns since last healthy proof
    let mut clean_run: u64 = 0;      // consecutive clean FIFO pops
    let mut backup_swaps: u64 = 0;   // swaps seen while the alternator is active
    let mut backup_fires: u64 = 0;   // fires emitted while the alternator is active

    // Look-ahead diagnostic: how early (in us) do swap arrivals land vs. the
    // vblank grid. Large values confirm there's ample lead for FIFO
    // content-follow; values near/at ALARM_DELAY_US+HOST_LEAD_US mean swaps
    // are arriving close to the fire deadline and `underruns` is worth
    // watching.
    let mut lead_n: u64 = 0;
    let mut lead_sum: u64 = 0;
    let mut lead_min: u64 = 0;

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
        // presenting (keeps us alive); it's also queued verbatim, in order, to
        // be popped as the fire eye for the next vblank slot(s) -- no
        // interpretation, just FIFO.
        let got = shm.drain();
        if !got.is_empty() {
            last_swap = Instant::now();
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
                }
            }
            for eye in got {
                let v = eye == EYE_RIGHT;
                // While the alternator backup is active the FIFO is suspended
                // (we do not content-follow); instead we count swaps against
                // fires so `pick_eye` can detect the stream returning to full
                // cadence and resume the FIFO.
                if backup_active {
                    backup_swaps += 1;
                } else {
                    pending_eyes.push_back(v);
                }
                // Backup: track the game's sequence parity (eye XOR present
                // index).  A healthy alternating L/R stream has constant
                // parity; a dropped or hitched present flips it and keeps it
                // flipped -- the exact failure of the "every frame is perfect"
                // FIFO assumption.  Only a sustained inversion (a full drift
                // window) re-aligns the alternator's polarity.
                let sf = v ^ ((swapped_seen & 1) == 1);
                swapped_seen += 1;
                match base {
                    None => {
                        base = Some(sf);
                        drift = 0;
                    }
                    Some(b) => {
                        if sf == b {
                            drift = drift.saturating_add(1);
                        } else {
                            drift = drift.saturating_sub(1);
                        }
                        drift = drift.clamp(-DRIFT_WINDOW, DRIFT_WINDOW);
                        if drift <= -DRIFT_WINDOW {
                            // Confirmed slip: content-follow is now off by a
                            // frame.  Re-align the tracked polarity (a single
                            // same-eye pair via pending_flip) and switch to the
                            // strict alternator; the stale FIFO is cleared so a
                            // fresh, aligned queue is rebuilt once we resume.
                            base = Some(!b);
                            drift = 0;
                            corrections += 1;
                            last_swap = Instant::now();
                            pending_flip = true;
                            backup_active = true;
                            backup_fires = 0;
                            backup_swaps = 0;
                            pending_eyes.clear();
                        }
                    }
                }
            }
            // A backlog this deep means the fire loop has fallen well behind
            // the game's swap stream (host scheduling hiccup, not a real
            // desync) -- drop the oldest rather than let fire-time latency
            // grow unbounded, but count it: unlike a genuine underrun, this
            // is real data we're choosing to discard, worth telling apart in
            // the log.
            while pending_eyes.len() > MAX_QUEUED_EYES {
                pending_eyes.pop_front();
                dropped_stale += 1;
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

            match drm_anchor.as_mut() {
                // --- Hardware-vblank anchored emission -------------------
                // One packet per real vblank slot, timed to the display vblank
                // clock. The eye is popped straight off `pending_eyes` -- the
                // DLL's actual report for that slot -- and pre-fired
                // ~ALARM_DELAY_US before the present vblank so the shutter
                // opens exactly as that frame appears.
                Some(anchor) if alive => {
                    match anchor.wait_vblank_blocking() {
                        Some(vblank_us) => {
                            last_vblank_epoch = Some(vblank_us);
                            let eye = pick_eye(
                                &mut pending_eyes,
                                &mut last_fired_eye,
                                &mut underruns,
                                &mut ur_counter,
                                &mut clean_run,
                                &mut fire_eye,
                                &mut pending_flip,
                                &mut backup_active,
                                &mut backup_swaps,
                                &mut backup_fires,
                            );
                            let alarm = if last_delay > 0 {
                                last_delay as u64
                            } else {
                                ALARM_DELAY_US
                            };
                            let next_present = vblank_us.saturating_add(anchor.period_us());
                            let fire_at = anchor.instant_of(
                                next_present.saturating_sub(alarm + HOST_LEAD_US),
                            );
                            while Instant::now() < fire_at {
                                std::hint::spin_loop();
                            }
                            d.send_eye(eye, rate_hz);
                            dbg.record(eye);
                            last_fired_eye = eye;
                        }
                        None => {
                            // Anchor died: fire immediately to stay live.
                            let eye = pick_eye(
                                &mut pending_eyes,
                                &mut last_fired_eye,
                                &mut underruns,
                                &mut ur_counter,
                                &mut clean_run,
                                &mut fire_eye,
                                &mut pending_flip,
                                &mut backup_active,
                                &mut backup_swaps,
                                &mut backup_fires,
                            );
                            d.send_eye(eye, rate_hz);
                            dbg.record(eye);
                            last_fired_eye = eye;
                        }
                    }
                }
                // Anchor present but idle (no swaps recently): stay silent.
                Some(_) => {
                    // Don't let idle drain batches pollute the backup resume
                    // cadence estimate (swaps with no matching fires).
                    backup_swaps = 0;
                    std::thread::sleep(Duration::from_millis(2));
                }
                // --- No DRM anchor: fire on the presents directly. ------
                None => {
                    if alive {
                        let eye = pick_eye(
                            &mut pending_eyes,
                            &mut last_fired_eye,
                            &mut underruns,
                            &mut ur_counter,
                            &mut clean_run,
                            &mut fire_eye,
                            &mut pending_flip,
                            &mut backup_active,
                            &mut backup_swaps,
                            &mut backup_fires,
                        );
                        d.send_eye(eye, rate_hz);
                        dbg.record(eye);
                        last_fired_eye = eye;
                    }
                }
            }

            dbg.tick(
                corrections,
                &mut underruns,
                &mut dropped_stale,
                &mut lead_n,
                &mut lead_sum,
                &mut lead_min,
            );
        }
    }
}
