//! `nvstusb-host` — Linux host helper for the NVIDIA 3D Vision USB IR emitter.
//!
//! wiz3D's `Nvidia3DOutput.dll` (running under Wine/Proton) pushes one
//! eye-swap command per presented frame into a shared-memory ring
//! (`/tmp/nvstusb.shm`, exposed to Wine as `Z:\tmp\nvstusb.shm`) and sends a
//! non-blocking one-byte UDP datagram to wake us.  This helper owns the USB
//! emitter and fires the shutter packet in step with the display.
//!
//! ## Why the eye comes straight from the game's ring -- and now WHEN
//!
//! The game always tells us which eye it is rendering: wiz3D's
//! `Nvidia3DOutput.dll` tags every presented frame with the concrete
//! `EYE_LEFT`/`EYE_RIGHT` it enqueued (SHUTTER_MODE_SIMPLE alternates
//! `Output(true); Present; Output(false); Present` per frame), so the ring is
//! ground truth for what the monitor is about to show -- not a phase-less
//! count.  This build fires EXACTLY the eye the game reports: each anchored
//! vblank slot (and each no-anchor fallback fire) sends one drained ring eye.
//! No host-side alternator, no slip tracker.
//!
//! Version 2 of the ring layout also carries each swap's PRODUCER-STAMPED
//! SUBMIT TIME (`t_us`, host CLOCK_MONOTONIC -- the clock `drm.rs`'s vblank
//! timestamps use, so the two live in one timebase).  This kills the phase
//! lottery that made earlier builds invert at stream start: without a slot,
//! the ring alone cannot tell WHICH display boundary an eye will scan out on,
//! so the phase was an emergent property of queue fill level/timing -- and
//! every discrete event (launch, resume after a pause, backlog flush,
//! re-anchor) re-lotted it with ~50% chance of a permanent inversion.  With
//! the submit timestamp each eye is pinned to the boundary its present
//! actually blocks to (FIFO-pinned onto the anchored grid), so a
//! stream start is PHASE-DETERMINISTIC and no dropped or collapsed pulse can
//! ever shift the whole stream a slot.
//!
//! The host services one boundary per `wait_vblank_blocking` slot: it fires
//! the eye pinned to that boundary and, when none is due (a dip in the
//! present rate, or a boundary whose eye was lost enqueue-side), FREERUNS by
//! toggling the shutter so the glasses' hardware lock survives a brief blip
//! -- but only for a short stretch (`STALL_HOLD_AFTER_FREERUN_SLOTS`).  Past
//! that the gap is a real stall, not jitter, and the host HOLDS the
//! last-fired eye instead of manufacturing more alternation over a frame
//! that provably has not changed (see that constant's doc comment for why
//! blind strobing there is what causes eye strain).  Either way it is a
//! per-eye loss, never a whole-stream shift.  Off-grid paths (no DRM anchor,
//! or the anchor's wait failing) fire on the presents directly, FIFO, like
//! the legacy build.
//!
//! The version-1 region (an older DLL that does not stamp) degrades to the
//! historical FIFO+hold behavior -- see `EyeQueue` below.  In that fallback
//! mode the emitter's 3D button toggles a manual eye inversion; in the stamped
//! (phase-pinned) mode the same button instead sets the host -> DLL
//! `FLAG_INVERT_EYES` bit on the shared header, commanding wiz3D's
//! `Nvidia3DOutput` DLL to swap the left/right eye it renders AND reports.
//! The glasses are never shifted: they keep shuttering at their natural
//! cadence, so correcting a swapped-eye view costs no extra dark/black period
//! (the old approach re-pinned the whole schedule by +/- one period on the
//! glasses side, which held one eye for the extra slot -- visibly uncomfortable).
//!
//! The ring's COUNT still powers the count-based features: the `alive` gate
//! (emit only while a swap arrived within the last 300 ms, so a dead game
//! leaves the emitter silent) and the swap-starvation telemetry.  Every
//! packet is pre-fired ~ALARM_DELAY_US before its target boundary on the
//! display engine's real vblank clock (see `drm.rs`); the glasses lock when
//! consecutive packets strictly alternate at a stable ~120 Hz period.  A
//! sustained rate mismatch (content not landing one-per-boundary) is NOT
//! chased -- no emitter-side trick can shutter content whose eyes don't
//! alternate at the display rate -- it is latched and reported loudly
//! (swap-starvation telemetry) instead, because that failure must be fixed
//! upstream (force SyncInterval=1 / full-rate presents).

use std::collections::VecDeque;
use std::env;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use crate::nvstusb::drm;
use crate::nvstusb::usb;
use crate::nvtimings;
use crate::shm;
use crate::shm::{EYE_RIGHT, FLAG_EMITTER_PRESENT, FLAG_FIRMWARE_LOADED,
                Shm, STATUS_ERROR, STATUS_OPENING, STATUS_READY};

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
    /// The drop counters reported by the previous tick, so the next tick can
    /// print the delta for its own window (the EyeQueue counters are cumulative
    /// and reset only at report time).
    last_reported_drops: DropDiagnostic,
}

const MASTER_LOCK_MIN_US: u64 = 7600;
const MASTER_LOCK_MAX_US: u64 = 9000;

/// The glasses tolerate a handful of dropped/repeated eyes per second before the
/// stream is treated as unlocked -- a near-boundary mis-pin or a freerun/resume
/// parity collision can drop one eye without actually breaking the shutter
/// cadence the glasses lock to.  A strict `same_eye == 0` flagged the whole
/// window "will NOT lock" on a single drop even though the glasses were holding
/// lock (the verdict is a conservative diagnostic, not the glasses' state).
/// Sub-60fps content makes the host freerun over eye gaps, which legitimately
/// raises same-eyes in direct proportion to the content shortfall, so this is
/// generous enough to not flag a healthy-but-content-limited stream.
const MASTER_LOCK_MAX_SAME_EYE: u64 = 10;

/// The fixed firmware packet -> IR alarm delay is owned by `drm.rs` and
/// applied by `frame_start` (target - 3000 - lead).  The host no longer keeps
/// a private copy: it subtracted `last_delay` (the DLL's register-0x23 value)
/// from the pre-fire deadline on top of this constant, which on real nvstusb
/// hardware (where the delay is a FIXED 3000) double-counted whenever the DLL
/// published anything but 3000 -- firing the packet too early and opening the
/// shutter on the previous eye (weak 3D).  The probe now lets `frame_start`
/// (and, for the RP2040 clone, the separate `set_alarm_delay_us` write) own
/// the alarm.
/// Default extra lead folded into the pre-fire busy-wait (see
/// [`fire_lead_us`]).  Matches the 3dv3d demo's default `swap_phase_us` (3100)
/// -- the value `NvstusbContext::new` sets and the working `cargo run` demo
/// feeds to `DrmVblank::frame_start`.  The host must pre-fire the SAME lead so
/// the shutter window lands in the same place relative to the boundary; the old
/// 250us default opened the shutter ~2850us too late, sliding the X/Y/W window
/// off the eye frame (glasses "darker than they should").
const DEFAULT_HOST_LEAD_US: u64 = 3100;

/// Pre-fire lead in microseconds (`NVSTUSB_HOST_LEAD_US` overrides).  This is
/// the host-side analogue of the demo's `swap_phase_us`; it MUST match it (the
/// demo drives the glasses correctly at its 3100 default).  `frame_start`
/// fires the packet `IR_ALARM_DELAY_US(3000) + lead` before the target boundary
/// so the IR alarm lands `lead` before the boundary; the same value the demo
/// feeds to its `frame_start`.  Tune on a per-monitor basis exactly like the
/// demo's `,`/`.`/`[`/`]` keys.  Too small (the old 250) opens the shutter
/// ~2850us late so the X/Y/W open window slides off the eye frame (dark /
/// crosstalk); too large reaches back into the previous eye's frame.
fn fire_lead_us() -> u64 {
    env_or("NVSTUSB_HOST_LEAD_US", "3100")
        .parse()
        .unwrap_or(DEFAULT_HOST_LEAD_US)
}

/// Smallest boundary on the anchor's vblank grid -- `ref_boundary` is any
/// confirmed boundary and `period` the measured period -- that lies STRICTLY
/// after `t`.  A present submitted at `t` blocks until its own boundary, so
/// the strictly-after mapping puts every stamped swap on the exact display
/// slot it scans out on (mid-period submit -> that boundary, after-boundary
/// submit -> the next one).
///
/// The grid is uniform, so `ref_boundary` serves only as an ORIGIN: the answer
/// is `ref_boundary + n*period` for the signed `n` that lands the first
/// lattice point after `t`.  The signed form matters -- it resolves `t`
/// EARLIER than `ref_boundary` to the actual slot that present scans out on
/// (which lies in the past), instead of clamping to `ref_boundary + period`.
/// A top-of-loop drain and a post-vblank drain confirm different boundaries,
/// so a naive "before-ref -> ref+period" made the SAME swap pin one slot
/// ahead in one drain and (correctly) on its own slot in the other.  That
/// race over-pinned a just-passed eye onto the next future slot, where it
/// collided with the next frame's eye, de-collision bumped the wrong eye, and
/// a freerun stamped the vacated slot with the same eye -- the `alternating`
/// drops and "switching eyes" even at a perfectly steady 120 Hz.  Pinning the
/// true (possibly past) slot makes every drain agree: on-time eyes fire on
/// their own boundary, and eyes that truly latched a past boundary are
/// dropped by `next_eye`'s overdue logic rather than replayed onto the wrong
/// slot.
///
/// NOTE: production pinning no longer feeds on the producer's submit stamp
/// (Wine/Proton buffering makes it many periods stale); `ingest` pins FIFO to
/// the host grid instead.  This pure grid-mapping function is kept for the
/// tests that document the mapping semantics.
#[cfg(test)]
fn boundary_after(ref_boundary: u64, period: u64, t: u64) -> u64 {
    let period = period.max(1);
    // `div_euclid` rounds toward -inf, so an eye submitted before the origin
    // lands in the correct past lattice slot (not clamped to origin + period).
    let n = (t as i64 - ref_boundary as i64).div_euclid(period as i64) + 1;
    (ref_boundary as i64 + n.saturating_mul(period as i64)).max(0) as u64
}

/// Backlog threshold for the eye schedule: when queued presents exceed this we
/// have fallen behind (the host missed several real flips) and must drop the
/// stale prefix instead of replaying eyes the screen already showed.  A few
/// frames deep so a genuine stall re-anchors while normal interleaved drainage
/// never trips it.
const MAX_QUEUE: usize = 8;

/// Consecutive freerun (invented, no-DLL-eye) fires before the stream is
/// treated as having come back from a dip.  A single slot here, when the
/// schedule momentarily empties between two real frames, is a normal blip; a
/// stretch of TWO OR MORE means the host has been emitting a phase the game
/// did not report across a real content gap.
///
/// Why re-anchor at >= 2 (not 1, and not the old 6): a ONE-slot gap can never
/// invert the resumed stream -- the game strictly alternates L/R, so the one
/// freerun-invented eye is always the very eye the game presents next, and
/// re-anchoring it would be a value-noop (see `short_blip_is_not_a_dip`).  The
/// parity only becomes arbitrary once the gap is long enough for the game's
/// own frame counter to drift against the host's freerun toggling -- i.e.
/// every real dip of >= 2 slots.  Those MUST re-anchor on the first fresh DLL
/// swap, or the resumed stream inherits the freerun parity with a ~50/50
/// chance of latching INVERTED (the "eyes swap after a frame dip" symptom).
/// The old threshold of 6 (~50 ms) left every sub-50 ms dip -- the common
/// game-stutter case -- unrecovered and 50/50 for inversion.  Two slots
/// (~17 ms) is the minimum that catches every gap that can invert, while a
/// one-slot blip (normal healthy micro-jitter) still deliberately skips the
/// re-anchor.
const DIP_RESYNC_AFTER_FREERUN_SLOTS: u32 = 2; // ~17 ms at 120 Hz

/// Consecutive freerun fires after which the host stops MANUFACTURING
/// alternation and just holds the last eye instead.  Toggling every ~8.3ms
/// keeps the glasses' hardware lock alive across a genuine one-or-two-slot
/// gap (normal SIMPLE-mode jitter), which is worth it -- the screen content
/// really did just alternate a moment ago.  But once the gap runs this long
/// the screen is provably NOT updating (a real stall: menu, load hitch,
/// stutter) -- there is no new stereo content to separate, so continuing to
/// flip the shutters is pure strobe over a static frame with no benefit,
/// and it is exactly this stretch of blind alternation (not the routine
/// content-limited dips, which stay inside the lock window) that reads as
/// harsh strobing to the eyes.  12 slots (~100ms, comfortably inside the
/// 300ms `alive` gate) distinguishes "brief jitter, keep flickering to hold
/// lock" from "sustained stall, stop flickering a dead frame".  The instant
/// a real DLL eye returns, `freerun_since` resets and normal firing (plus
/// dip-recovery re-anchoring) resumes exactly as before.
const STALL_HOLD_AFTER_FREERUN_SLOTS: u32 = 24;

/// `NVSTUSB_STALL_HOLD_SLOTS` override for [`STALL_HOLD_AFTER_FREERUN_SLOTS`],
/// read once at startup -- comfort here is genuinely a per-person, per-panel
/// judgment call (how visible a repeated-eye flash is depends on the specific
/// glasses, the monitor's persistence, and the viewer), so this is exposed as
/// a runtime knob instead of only a recompile-time constant.  `0` holds on
/// EVERY dip (no toggling at all past the very first slot); a large value
/// (e.g. 999999) restores the old always-toggle behavior for comparison.
fn stall_hold_after_freerun_slots() -> u32 {
    env_or("NVSTUSB_STALL_HOLD_SLOTS", &STALL_HOLD_AFTER_FREERUN_SLOTS.to_string())
        .parse()
        .unwrap_or(STALL_HOLD_AFTER_FREERUN_SLOTS)
}

/// The eye source, in two modes.
///
/// STAMPED (version-2 ring): the producer stamps each swap with a
/// CLOCK_MONOTONIC PRESENT time in `Swap::boundary_us`.  We deliberately do NOT
/// convert that stamp onto the host grid to pin the eye (the old
/// `boundary_after(host_us_from_mono(stamp))` path): under Wine the DLL's QPC
/// epoch does not coincide with the host's CLOCK_MONOTONIC boot epoch that
/// `host_us_from_mono` subtracts, so the mapping translated the whole schedule
/// a constant number of periods behind `armed` (observed ~28 = 233324us) --
/// every eye dropped OVERDUE, the host freeran every slot, and the long-freerun
/// hold let the glasses sit dark.  Production pinning is therefore pure FIFO
/// onto the host's OWN anchored grid (below), immune to that epoch offset.  The
/// stamp is kept on the wire (fixed-size ring layout) and the `boundary_after`
/// mapping is retained as `#[cfg(test)]` documentation of the mapping.
///
/// FIFO (both stamped and legacy rings): each drained swap is pinned to the
/// next unassigned boundary on the host's anchored grid, advancing one period
/// per swap in drain order.  The swap that arrives is the next eye the display
/// will show, so FIFO pinning to the next free slot is correct and stable while
/// the ring stays fed.
///
/// The fired stream is a PURE function of what the game reported -- the host
/// NEVER invents, drops-on-whim, or flips a real eye, so the game's constant
/// eye cadence is preserved and polarity stays locked to the screen:
///
///   * FIFO to the anchored grid, not a `.last()`/FIFO-pop that replays: each
///     eye fires on a distinct upcoming boundary, and the SQLite-style
///     same-boundary advance in `enqueue` keeps a SIMPLE-mode (L,R) pair on
///     CONSECUTIVE slots instead of collapsing onto the SAME eye.
///   * FREERUN, not silence: when nothing is due on a slot (a dip), the game's
///     last-presented eye is still on screen, but instead of freezing on it
///     (which reads as mono and drops lock) we keep the glasses shuttering by
///     toggling to the opposite eye, so lock holds through the dip and the
///     stream resyncs to the game's eyes the moment they return.  A sustained
///     freerun gap means the held eye phase has strayed from the game, so the
///     first fresh DLL-reported eye after that gap clears the freerun parity
///     and re-pins the resumed stream (DIP-RECOVERY).  This is what stops a
///     post-dip session from latching INVERTED (within the FIFO path's
///     best-known recovery).
///   * RE-ANCHOR on backlog, never replay stale eyes: a fall-behind drops the
///     stale prefix and re-anchors on the newest eye.
struct EyeQueue {
    /// Fallback FIFO: `true` = right, in enqueue order.
    fallback: VecDeque<bool>,
    /// Stamped schedule: (target boundary host-epoch us, right), ascending.
    scheduled: VecDeque<(u64, bool)>,
    /// The most recently fired (== on-screen) eye, and the anchor for
    /// freerunning across a dip.  `None` only before the first drain, when we
    /// must stay silent (never invent an eye).
    last_fired: Option<bool>,
    /// Consecutive FREERUN (invented eye, no DLL data) fires since the last
    /// real swap.  Drives dip-recovery: after a substantive gap the held phase
    /// is re-anchored to the DLL's first fresh eye (see `enqueue`).
    freerun_since: u32,
    /// Cumulative per-window drop counters, so a `same_eye` in the fired stream
    /// can be traced to its source instead of diagnosing blind.  None of these
    /// change the fired eye (polarity is untouched) -- they only tag events the
    /// schedule already performs.
    drops: DropDiagnostic,
    /// See [`STALL_HOLD_AFTER_FREERUN_SLOTS`] / [`stall_hold_after_freerun_slots`].
    stall_hold_after_freerun: u32,
    /// A phase discontinuity has happened (an OVERDUE drop, or a FREERUN that
    /// invented an eye after one) and the held/fired phase can no longer be
    /// trusted as an ARBITRARY value -- gluing `last_fired` to whatever the
    /// schedule happened to produce next could latch the stream INVERTED (the
    /// `overdue=1` and `freerun=7` eye-swap symptom).  Set by `next_eye` on a
    /// discontinuity; consumed (and cleared) by `enqueue` on the FIRST fresh,
    /// real, DLL-reported eye that arrives afterward, which re-anchors the held
    /// phase AND re-baselines the whole schedule onto the incoming eye's grid
    /// boundary.  Until then the schedule only ever freely runs, never fires a
    /// presumed-real eye that could bake a shifted phase into the stream.
    pending_reanchor: bool,
}

impl Default for EyeQueue {
    fn default() -> Self {
        EyeQueue {
            fallback: VecDeque::new(),
            scheduled: VecDeque::new(),
            last_fired: None,
            freerun_since: 0,
            drops: DropDiagnostic::default(),
            stall_hold_after_freerun: STALL_HOLD_AFTER_FREERUN_SLOTS,
            pending_reanchor: false,
        }
    }
}

/// Counters for the two ways `next_eye` can fail to fire a real eye -- each of
/// which, when it leaves a hole the next freerun/resume fills with the same
/// parity, is what shows up as an `alternating` dip / `same_eye`>0.
#[derive(Default, Clone, Copy)]
struct DropDiagnostic {
    /// Eyes dropped as OVERDUE: scheduled boundary already >half a period past
    /// the slot being armed (a mis-pin, usually the submit-margin prediction).
    overdue: u64,
    /// Eyes dropped by COALESCING: several due eyes mapped within +/-half of
    /// one armed slot, so only the last ever fires (earlier ones skipped).
    coalesced: u64,
    /// Fire slots served by a FREERUN (invented toggle, no real DLL eye due) --
    /// the parity collision between one of these and the next real eye is the
    /// other same-eye source.
    freerun: u64,
}

impl EyeQueue {
    /// Ingests one drained swap.  With a pinned `target` the eye joins the
    /// schedule; otherwise it goes to the fallback FIFO.
    fn enqueue(&mut self, right: bool, target: Option<u64>, period: u64) {
        match target {
            Some(mut t) => {
                // Frame-sequential SIMPLE content submits both eyes of a frame
                // back-to-back -- often landing in SEPARATE top/late drains --
                // before the one shared vblank they collectively block to, so
                // their submit stamps resolve to the SAME boundary.  Resolve the
                // collision here against the LIVE schedule, in submission order
                // (enqueue is called oldest-first): the first eye to claim a
                // boundary keeps it, and each later eye pinned to that same slot
                // advances a whole period onto the following slot (the display
                // presents consecutive submits on CONSECUTIVE vblanks).  Without
                // this, `next_eye`'s "newest wins the slot" drops every older eye
                // and the fired stream never alternates (all-R or all-L) ->
                // "both eyes of equal strength" ghosting.  The stamp pinning also
                // keeps genuine one-eye-per-slot content (true 120 Hz) at their
                // exact boundaries: no collision, no shift.
                let period = period.max(1);
                while self.scheduled.iter().any(|&(b, _)| b == t) {
                    t = t.saturating_add(period);
                }
                // Phase recovery: this is a fresh, REAL, DLL-reported eye at a
                // grid boundary.  Whenever the held/fired phase is no longer
                // trustworthy -- a substantive freerun gap (`freerun_since`),
                // or an overdue drop that proved the schedule drifted off-grid
                // (`pending_reanchor`) -- re-anchor to THIS eye.  The freerun-
                // invented `last_fired` (or the stale phase left behind by a
                // dropped eye) holds arbitrary game-relative parity, so it must
                // be re-pinned to this real eye.  Pin the held phase to `right`
                // so the next slot alternates off it, and re-baseline the
                // existing schedule onto THIS eye's grid boundary so the stream
                // resumes pinned (absolute) instead of continuing the shifted,
                // drifted phase that caused the inversion (the `overdue=1` /
                // `freerun=7` eye-swap symptom).
                if self.freerun_since >= DIP_RESYNC_AFTER_FREERUN_SLOTS
                    || self.pending_reanchor
                {
                    self.freerun_since = 0;
                    self.pending_reanchor = false; // fresh real eye consumed it
                    self.last_fired = Some(right);
                    self.rebaseline(t, period);
                    // `rebaseline` shifted the survivors onto the grid; a
                    // survivor that lands exactly on THIS fresh eye's boundary
                    // `t` is stale relative to it (the fresh eye owns this slot
                    // -- it is the newest of the recovery) and is dropped.
                    self.scheduled.retain(|&(b, _)| b != t);
                }
                // Swaps arrive in near-submit order, so an insertion sort keeps
                // only a few entries moving per drain (queue is tiny).
                let idx = self
                    .scheduled
                    .iter()
                    .position(|&(b, _)| b > t)
                    .unwrap_or(self.scheduled.len());
                self.scheduled.insert(idx, (t, right));
                if self.scheduled.len() > MAX_QUEUE {
                    // Fall-behind: the screen has already cycled through the
                    // older eyes.  Re-anchor on the newest stamped present: its
                    // boundary is real, so the stream resumes pinned rather
                    // than re-lotting.
                    let newest = *self.scheduled.back().unwrap();
                    self.scheduled.clear();
                    self.scheduled.push_back(newest);
                }
            }
            None => {
                self.fallback.push_back(right);
                if self.fallback.len() > MAX_QUEUE {
                    // Legacy-mode variant of the same re-anchor: the newest eye
                    // is on screen now -- keep it as the anchor (both the held
                    // eye and the next slot's fire) and drop the stale prefix.
                    let newest = *self.fallback.back().unwrap();
                    self.fallback.clear();
                    self.fallback.push_back(newest);
                    self.last_fired = Some(newest);
                    self.freerun_since = 0; // real DLL eye -> phase authoritative
                }
            }
        }
    }

    /// Ingests every swap from a drain, computing each eye's pinned boundary
    /// when the caller can convert its timestamp onto the anchored grid.
    /// Same-boundary collisions are resolved inside `enqueue`, against the
    /// live schedule, so a frame-sequential SIMPLE pair stays alternating even
    /// when its two eyes land in separate drains.
    fn ingest(
        &mut self,
        swaps: &[shm::Swap],
        anchor: Option<&drm::DrmVblank>,
        next_boundary: Option<u64>,
    ) {
        // Each drained swap is pinned to the NEXT unassigned boundary on the
        // display grid in FIFO order: the host knows the grid from the DRM
        // anchor and advances it by one period per swap, so the swap that
        // arrives becomes the next eye the display will show.  We do NOT pin
        // from the producer's per-slot absolute PRESENT time
        // (`Swap::boundary_us`): under Wine the DLL's QPC epoch does not
        // coincide with
        // the host's CLOCK_MONOTONIC boot epoch that `host_us_from_mono`
        // subtracts, so mapping it onto the grid translated the whole schedule
        // a constant number of periods behind the armed boundary -- every eye
        // dropped OVERDUE, the host freeran every slot, and the long-freerun
        // hold let the glasses sit dark (see the module docs).  FIFO onto the
        // host's own grid is immune to that epoch offset.  The existing
        // `enqueue` re-spacing still handles bursty same-boundary pairs (e.g.
        // 30fps where 2 eyes arrive in one game frame, often in separate
        // top/late drains) by bumping the second eye forward.
        let period = anchor.map_or(0, |a| a.period_us().max(1));
        // `next_boundary` is only the CALLER's guess at the earliest boundary
        // this batch could want (derived fresh from the DRM anchor each call,
        // e.g. `last_vblank + 2*period`) -- it does NOT know where the
        // schedule itself already extends to.  SIMPLE-mode content routinely
        // submits both eyes of a frame close together, so the schedule's tail
        // is often already ahead of that guess by the time the NEXT drain
        // calls in.  Starting from behind the tail there makes every eye in
        // this batch collision-walk forward through everything already
        // queued (`enqueue`'s same-boundary loop) -- which, on the normal
        // bursty delivery pattern, can push `scheduled.len()` past
        // `MAX_QUEUE` on a perfectly healthy stream and trip the backlog cut,
        // discarding every already-correctly-scheduled future eye down to
        // one.  Floor the starting boundary at the schedule's own tail so a
        // batch never re-probes ground it has already claimed.
        let mut boundary =
            next_boundary.map(|b| floored_next_boundary(self.scheduled.back().map(|&(t, _)| t), b, period));
        for s in swaps {
            self.enqueue(s.eye == EYE_RIGHT, boundary, period);
            if let Some(b) = boundary {
                boundary = Some(b.saturating_add(period));
            }
        }
    }

    /// Re-aligns the stamped schedule onto the display grid after a phase
    /// discontinuity (an overdue drop, or a freerun gap that broke the held
    /// parity).  The FIFO-pinned boundaries can drift a sub-period off the
    /// armed grid; firing eyes at their drifted offsets lands one eye on the
    /// wrong slot = a shifted (inverted) stream with no recovery.  `rebaseline`
    /// corrects ONLY that sub-period drift, never a whole display slot (a slot
    /// step is exactly an inversion): it measures how far the earliest
    /// still-pending eye has drifted from `grid_origin`, normalizes the offset
    /// into [-half, +half], shifts every still-valid queued eye back onto the
    /// grid, and drops the stale prefix that (after the shift) has already
    /// scanned out.  Eye ORDER -- and therefore the game-absolute left/right
    /// alternation -- is untouched, so a corrected schedule fires the right eye
    /// on the right slot again, deterministically.
    fn rebaseline(&mut self, grid_origin: u64, period: u64) {
        let period = period.max(1);
        let Some(front) = self.scheduled.front().map(|&(b, _)| b) else {
            return;
        };
        // Earliest still-pending eye, as a sub-period offset from the grid.
        let half = (period / 2) as i64;
        let mut delta = front as i64 - grid_origin as i64;
        delta = delta.rem_euclid(period as i64);
        if delta > half {
            delta -= period as i64;
        } else if delta < -half {
            delta += period as i64;
        }
        // Drop eyes that (once shifted back onto the grid) have already
        // scanned out, then re-pin the survivors by `-delta`.
        self.scheduled
            .retain(|&(b, _)| (b as i64 - delta) >= grid_origin as i64);
        for (b, _) in self.scheduled.iter_mut() {
            *b = ((*b as i64) - delta).max(0) as u64;
        }
    }

    /// The eye to fire for the boundary being armed.  Eyes whose boundary has
    /// already passed are dropped (that slot is on screen); the eye bound to
    /// THIS boundary is popped and fired -- if several map to the same slot
    /// (coalesced presents for a boundary that came late) the NEWEST one wins,
    /// exactly the eye the screen latched.  With none due (a dip: a slot whose
    /// game eye never arrived), FREERUNS -- toggles to the opposite eye so the
    /// glasses keep shuttering/alternating at the display rate (lock holds,
    /// `same_eye` stays 0) rather than holding one eye (which reads as mono and
    /// drops lock); the moment the game's pinned eyes return they pop at their
    /// own boundaries and the stream resyncs.
    fn next_eye(&mut self, boundary: u64, period: u64) -> Option<bool> {
        if !self.scheduled.is_empty() {
            let period = period.max(1);
            let half = period / 2;
            // Overdue: the front eye sits more than half a period behind the
            // slot being armed -- its display slot has already been on screen,
            // so it is dropped (the eye that IS on screen fired earlier).  Each
            // such drop leaves a hole a later freerun/resume can fill with the
            // same parity (an `alternating` dip), so count it.
            let mut dropped = false;
            while self
                .scheduled
                .front()
                .map_or(false, |&(b, _)| b + half < boundary)
            {
                self.scheduled.pop_front();
                self.drops.overdue += 1;
                dropped = true;
            }
            if dropped {
                // The front falling > half a period behind the armed grid is
                // proof the FIFO-pinned schedule has DRIFTED off the display
                // grid.  Dropping the eye is necessary but not sufficient: the
                // remaining queued eyes are still off-grid, and firing them at
                // their drifted offsets would land one eye on the WRONG slot =
                // a shifted (inverted) stream with no recovery.  Re-baseline
                // the schedule onto the armed grid AT ONCE so the very next
                // fire is grid-correct and deterministic, and mark the queue so
                // `enqueue` also re-pins the held phase to the first fresh real
                // eye that arrives.
                self.rebaseline(boundary, period);
                self.pending_reanchor = true;
            }
            // Due: every front entry within +/-half belongs to this boundary
            // (the phase-preserving nudge moves the armed slot by sub-periods
            // at most).  Multiple entries mean the earlier eyes were coalesced
            // onto this slot -> fire the last (the earlier ones are dropped and
            // counted below).
            let mut due: Option<bool> = None;
            while self
                .scheduled
                .front()
                .map_or(false, |&(b, _)| b <= boundary + half && b + half >= boundary)
            {
                due = self.scheduled.pop_front().map(|(_, r)| r);
                // Every coalesced-away eye is a potential same-eye source.
                self.drops.coalesced += 1;
            }
            if let Some(e) = due {
                // The single kept (last) due eye is not a drop -- undo that one.
                self.drops.coalesced = self.drops.coalesced.saturating_sub(1);
                self.last_fired = Some(e);
                self.freerun_since = 0; // real DLL eye -> phase authoritative
                return Some(e);
            }
            // None due: freerun.  Toggle to the opposite eye so the fired
            // stream keeps shuttering/alternating (lock holds); the moment the
            // game's pinned eyes return they pop at their own boundaries and
            // the stream resyncs.  Past STALL_HOLD_AFTER_FREERUN_SLOTS this is
            // no longer a blip but a real stall -- stop toggling and hold the
            // last eye so a stuck/stale frame isn't strobed (see the constant's
            // doc comment).
            self.freerun_since = self.freerun_since.saturating_add(1);
            self.drops.freerun += 1;
            // The freerun INVENTED this eye (the game never reported it for
            // this slot), so the held phase has left the game's true schedule.
            // Glue the next fresh DLL eye to a grid re-baseline so the resumed
            // stream cannot inherit this invented parity (which is the 50/50
            // "eyes swapped after a dip" source).
            self.pending_reanchor = true;
            if self.freerun_since > self.stall_hold_after_freerun {
                return self.last_fired;
            }
            return self.last_fired.map(|r| freerun(&mut self.last_fired, r));
        }
        if let Some(e) = self.fallback.pop_front() {
            self.last_fired = Some(e);
            self.freerun_since = 0; // real DLL eye -> phase is authoritative
            Some(e)
        } else if let Some(r) = self.last_fired {
            // Scheduled empty AND fallback empty: a dip in the degraded path.
            self.freerun_since = self.freerun_since.saturating_add(1);
            self.drops.freerun += 1;
            self.pending_reanchor = true;
            if self.freerun_since > self.stall_hold_after_freerun {
                return self.last_fired;
            }
            Some(freerun(&mut self.last_fired, r))
        } else {
            None
        }
    }

    /// Present-driven fire for the off-grid paths (no anchor, or the wait
    /// failing): no grid, so boundaries are meaningless; pop the oldest known
    /// eye regardless of target.
    fn next_eye_present_driven(&mut self) -> Option<bool> {
        if let Some((_, r)) = self.scheduled.pop_front() {
            self.last_fired = Some(r);
            self.freerun_since = 0;
            Some(r)
        } else if let Some(e) = self.fallback.pop_front() {
            self.last_fired = Some(e);
            self.freerun_since = 0;
            Some(e)
        } else {
            self.last_fired
        }
    }

    /// Re-pins every scheduled eye by `delta` us.  Retained as a test-only /
    /// internal capability (it was the old `phase_slots` button action, which
    /// shifted the whole schedule by +/- one period on the glasses side).
    #[cfg(test)]
    fn shift_schedule(&mut self, delta: i64) {
        for (b, _) in self.scheduled.iter_mut() {
            *b = (*b as i64 + delta).max(0) as u64;
        }
    }

    /// Drops the stamped schedule only (producer stopped stamping).
    fn clear_scheduled(&mut self) {
        self.scheduled.clear();
    }

    /// Drops the fallback FIFO only (stamped scheduling has become available;
    /// unstamped startup swaps are stale).
    fn clear_fallback(&mut self) {
        self.fallback.clear();
    }

    /// True once the ring has told us an eye at all -- queued, scheduled, or
    /// fired-and-held.  Guards against firing before first contact with the
    /// game (distinct from `alive`, which is about fresh swaps arriving within
    /// the liveness window).
    fn has_eye(&self) -> bool {
        !self.fallback.is_empty() || !self.scheduled.is_empty() || self.last_fired.is_some()
    }

    /// TEMP DEBUG: front scheduled boundary (pinned boundary of the oldest eye).
    fn front_boundary(&self) -> Option<u64> {
        self.scheduled.front().map(|&(b, _)| b)
    }

    /// Full reset (game stop / re-anchor): queue AND held eye cleared so the
    /// next stream starts from a clean phase reference.
    fn reset(&mut self) {
        self.fallback.clear();
        self.scheduled.clear();
        self.last_fired = None;
        self.freerun_since = 0;
        self.pending_reanchor = false;
    }
}

/// Floors a caller-guessed starting boundary at the schedule's own tail, so
/// `ingest` never re-probes ground a prior bursty batch already claimed (see
/// `ingest`'s doc comment for why re-probing behind the tail is dangerous).
/// `tail` is `scheduled.back()`'s boundary, if any; `guess` is the caller's
/// fresh anchor-derived estimate; `period` is the display period (0 means no
/// anchor -- the guess is used as-is, matching the pre-fix/fallback path).
fn floored_next_boundary(tail: Option<u64>, guess: u64, period: u64) -> u64 {
    match tail {
        Some(t) if t.saturating_add(period) > guess => t.saturating_add(period),
        _ => guess,
    }
}

/// Freerun (invented-eye) fire: toggles the held phase to the opposite eye and
/// returns it, so the fired stream keeps shuttering/alternating across a dip
/// (lock holds) instead of freezing on one eye (which reads as mono).  Records
/// the toggle back into `last_fired` so the next freerun alternates off it.
fn freerun(held: &mut Option<bool>, r: bool) -> bool {
    *held = Some(!r);
    !r
}

/// The stereo monitor's kernel connector name, if the user declared one via
/// `NVSTUSB_ANCHOR_OUTPUT`.  Optional: when unset, the connector is
/// auto-detected from the DRM vblank anchor (the first usable head).  This
/// name identifies the monitor for EDID-based monitor_timings.json lookups.
fn requested_connector() -> Option<String> {
    std::env::var("NVSTUSB_ANCHOR_OUTPUT")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Resolves a value published by the window owner into a concrete kernel
/// connector name suitable for re-anchoring the DRM vblank clock.  The shared
/// header can carry either a concrete connector name (e.g. `DP-2`, as the 3dv3d
/// demo publishes from winit's `current_monitor()`) or an EDID `VENDOR_PRODUCT`
/// base (e.g. `SAM_707A`, as the wiz3D DLL publishes from the target monitor's
/// EDID).  A name that matches an existing connector is used directly; an
/// EDID-shaped base is reverse-resolved to the connector carrying that EDID.
/// Returns `None` when nothing usable can be resolved (no signal, or the value
/// is neither a known connector nor a resolvable base).
fn resolve_published_connector(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    // If it names an existing connector (case-insensitive sysfs check), use it
    // as-is -- don't need to open /dev/dri to confirm, the anchor will.
    if crate::edid::connector_exists(value) {
        return Some(value.to_string());
    }
    // Otherwise treat it as an EDID VENDOR_PRODUCT base and find the connector
    // whose EDID decodes to it (prefers connected heads).
    crate::edid::find_connector_by_base(value)
}

/// Re-derives the DRM vblank anchor to sit on `conn` and refreshes
/// `anchor_conn` (the name used for EDID nvtimings lookups) to match.  On
/// multi-head setups each CRTC free-runs with its own phase, so re-anchoring
/// is what re-aligns the emitter to the head the window actually moved to.
/// If the new anchor can't be opened, the anchor is dropped (fall back to
/// present-driven emission) and `anchor_conn` keeps the requested name so the
/// timings profile for it is still applied.
fn derive_anchor_for(
    conn: &str,
    drm_anchor: &mut Option<drm::DrmVblank>,
    anchor_conn: &mut Option<String>,
) {
    *drm_anchor = drm::DrmVblank::open_preferring(Some(conn), None);
    let resolved = drm_anchor
        .as_ref()
        .and_then(|a| a.connector_name().map(str::to_owned));
    if let Some(r) = &resolved {
        eprintln!("nvstusb-host: DRM vblank anchor now on {r}");
    }
    // Use the resolved connector name (canonical casing) if the anchor opened,
    // else fall back to the requested name so nvtimings lookup still works.
    *anchor_conn = resolved.or_else(|| Some(conn.to_string()));
}

/// Applies a per-monitor shutter profile from `monitor_timings.json` to `device` for
/// `rate_hz`, if the monitor (`connector`, e.g. `DP-1`) has a matching entry.
/// This is the host-side counterpart of the 3dv3d demo's `s`-key save: a
/// profile tuned and saved by the demo is loaded here too, so
/// `nvstereo3d-host` shuts the glasses with the same X/Y/W registers the demo
/// settled on.  It ALSO applies the profile's per-monitor host IR lead and
/// returns it (in us) so the caller can update its `frame_start` lead
/// accordingly.  An explicit `NVSTUSB_HOST_LEAD_US` env override takes
/// precedence over the stored profile lead.  Returns `None` when no profile is
/// applied (so the caller keeps whatever lead it already has).
/// Lookup is by the monitor's EDID `VENDOR_PRODUCT` base and the rounded
/// refresh.
fn apply_timings_json(
    device: &usb::UsbDevice,
    rate_hz: f32,
    connector: Option<&str>,
) -> Option<u64> {
    let Some(conn) = connector.filter(|c| !c.is_empty()) else {
        eprintln!(
            "nvstusb-host: no DRM connector available; not applying monitor_timings.json \
             (the DRM vblank anchor is down)"
        );
        return None;
    };
    let Some(base) = crate::edid::resolve_base(conn, None) else {
        eprintln!(
            "nvstusb-host: could not read EDID for {conn}; not applying monitor_timings.json"
        );
        return None;
    };
    let Some((key, e)) = nvtimings::resolve(&nvtimings::load(), &base, rate_hz) else {
        eprintln!(
            "nvstusb-host: no monitor_timings.json profile for {base} near {rate_hz:.3} Hz; \
             using reference timings"
        );
        return None;
    };
    if let Err(err) = device.set_timings_us(rate_hz, e.x_us, e.y_us, e.w_us) {
        eprintln!("nvstusb-host: could not apply monitor_timings.json profile: {err}");
        return None;
    }
    // Per-monitor host IR lead from the profile.  The env var is an explicit
    // override that wins over the stored value.
    let lead = std::env::var_os("NVSTUSB_HOST_LEAD_US")
        .and_then(|v| v.into_string().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| e.lead_us.round().max(0.0) as u64);
    eprintln!(
        "nvstusb-host: applied monitor_timings.json [{}] @ {rate_hz:.3} Hz: X={:.3}us Y={:.3}us W={:.3}us LEAD={}us Z={:.3}us",
        key, e.x_us, e.y_us, e.w_us, lead, e.z_us()
    );
    Some(lead)
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
        swaps_this_window: &mut u64,
        batches_ge2: &mut u64,
        max_batch: &mut usize,
        lead_n: &mut u64,
        lead_sum: &mut u64,
        lead_min: &mut u64,
        lead_max: &mut u64,
        drops_now: DropDiagnostic,
        mode_tag: &str,
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
            && self.same_eye <= MASTER_LOCK_MAX_SAME_EYE;
        // Content health: while we fire one packet per display slot, wiz3D
        // must present (enqueue) at roughly the same rate. Anything far below
        // that means the screen is not showing a fresh eye every slot, which
        // no emitter timing can compensate.
        let starved = self.count >= 60 && *swaps_this_window * 2 < self.count as u64;
        // Content-limited (milder than starvation): the game is presenting
        // below ~75% of the 120/s display rate (e.g. ~45-58fps content).  The
        // host still emits its steady alternating cadence by freerunning over
        // the missing eyes, but every gap a freerun fills, a returning real eye
        // can legitimately collide with the same parity -- so sub-60 content
        // RAISES same_eye in direct proportion to the shortfall.  That is a
        // content property, not an emitter fault, so don't alarm "will NOT
        // lock"; the glasses are locked to the cadence, the content just can't
        // keep up.
        let content_limited = !starved
            && self.count >= 60
            && *swaps_this_window * 2 < (self.count as u64).saturating_mul(3) / 2;
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
        } else if content_limited {
            format!(
                "CONTENT-LIMITED (~{}fps): glasses locked, stream alternating \
                 in the lock window but the game is presenting below 60fps, so \
                 the eye cadence cannot be truly perfect -- raise the game's \
                 stereo framerate for a spotless lock",
                *swaps_this_window / 2
            )
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
            // Content fps: SIMPLE mode enqueues two swaps per game frame.
            let content_fps = *swaps_this_window / 2;
            // Assemble the diagnostic suffix: optional bursty-swaps / swap-lead
            // notes, then the lock-status verdict.  The fired stream IS the
            // game's stream now, so `alternating`/`same_eye` describe the game
            // eyes that actually went out the emitter.
            let mut suffix = String::new();
            if *batches_ge2 > 0 {
                suffix.push_str(&format!(
                    "bursty-swaps {}x>=2 (max batch {})",
                    *batches_ge2, *max_batch
                ));
            }
            if *lead_n > 0 {
                if !suffix.is_empty() {
                    suffix.push_str(" | ");
                }
                suffix.push_str(&format!(
                    "swap-lead {}us (min {}, max {})",
                    lead_avg, *lead_min, *lead_max
                ));
            }
            // Drop origin diagnostics for this window (delta since the last
            // report): each counts one non-fired real eye -- the direct source
            // of the `alternating`/`same_eye` dips.
            let d_overdue = drops_now.overdue.saturating_sub(self.last_reported_drops.overdue);
            let d_coalesced = drops_now
                .coalesced
                .saturating_sub(self.last_reported_drops.coalesced);
            let d_freerun = drops_now.freerun.saturating_sub(self.last_reported_drops.freerun);
            if d_overdue + d_coalesced + d_freerun > 0 {
                if !suffix.is_empty() {
                    suffix.push_str(" | ");
                }
                suffix.push_str(&format!(
                    "drops overdue={d_overdue} coalesced={d_coalesced} freerun={d_freerun}"
                ));
            }
            if !suffix.is_empty() {
                suffix.push_str(" | ");
            }
            suffix.push_str(&warn);
            eprintln!(
                "nvstusb-host: [{mode_tag}] {}/s packets | {}/s swaps (content~{}fps) | period {}-{}us (avg {}) | alternating {}/{} | {suffix}",
                self.count, swaps_this_window, content_fps, self.min_period, self.max_period, avg, self.alternating, self.count,
            );
        } else {
            eprintln!(
                "nvstusb-host: [{mode_tag}] 0 packets | {}/s swaps{extra} | {warn}",
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
        self.last_reported_drops = drops_now;
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
    // constant wrong-eye bias no button press can fix reliably.  This is
    // optional: without it the anchor picks the first usable head, and the
    // same connector is used for EDID-based monitor_timings.json lookups, so no env
    // var is required to apply a saved shutter profile.
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

    // The kernel connector name of the monitor we're shuttering: the anchor's
    // resolved connector if it matched one, else the user's declared
    // NVSTUSB_ANCHOR_OUTPUT.  Used for EDID-based monitor_timings.json lookups and
    // updated when the window owner republishes a different target connector.
    let mut anchor_conn = drm_anchor
        .as_ref()
        .and_then(|a| a.connector_name().map(str::to_owned))
        .or_else(requested_connector);
    if let Some(conn) = &anchor_conn {
        eprintln!("nvstusb-host: monitor connector for timings: {conn}");
    }

    // Pre-boundary fire lead (us).  Initialized from NVSTUSB_HOST_LEAD_US;
    // updated in place whenever a per-monitor monitor_timings.json profile is
    // applied (the stored `lead_us` wins unless the env override is set), so
    // `frame_start` fires with the monitor's tuned lead.
    let mut host_lead_us = fire_lead_us();

    // Try to (re)open the device until it appears; if we already had it and it
    // dies, drop it and retry on the next iteration.
    if device.is_none() {
        if let Some(d) = usb::open_device(ctx, FIRMWARE) {
            match d.configure(rate_hz) {
                Ok(()) => {
                    // Apply the per-monitor profile from monitor_timings.json (if
                    // any) on top of the reference timings `configure` loaded.
                    // The connector comes from the DRM vblank anchor above
                    // (auto-detected; NVSTUSB_ANCHOR_OUTPUT is optional).
                    if let Some(lead) = apply_timings_json(&d, rate_hz, anchor_conn.as_deref()) {
                        host_lead_us = lead;
                    }
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

    // --- Main loop --------------------------------------------------------
    let mut last_rate = rate_hz;
    let mut last_delay = delay_us;
    let mut buf = [0u8; 64];

    // The most recent connector the window owner published into the shared
    // region, so a republish that resolves to the same head does nothing and a
    // genuinely new head triggers a re-anchor (DRM vblank + nvtimings profile).
    let mut last_published_conn = String::new();

    // Manual polarity override for FALLBACK mode, toggled by the emitter's 3D
    // button (the host-side equivalent of the demo's `i` key).  In a legacy
    // version-1 ring the game's submit->scanout pipeline depth offset is
    // invisible to every in-band check, so there must be a human switch for it.
    let mut manual_invert = false;
    // Stamped-ring tracking: true while the producer stamps 16-byte version-2
    // slots (see `shm::stamped()`).  Gates phase-pinned scheduling vs the FIFO
    // fallback.  The edge is handled in the loop: a mode switch prunes the
    // queue that belongs to the other mode so a stale eye can never mix in.
    let mut stamped_mode = false;
    // Previous read of the emitter's 3D button, so the toggle is EDGE-triggered
    // on a 0->1 transition rather than level-triggered.  Level-triggering could
    // latch a spurious/garbage high bit (or a stuck button) into a PERMANENT
    // inversion with no way to self-correct -- exactly the "eyes just stayed
    // inverted" symptom.  Edge-triggering makes a single glitch immaterial.
    let mut button_was_down = false;

    // Per-second stream telemetry.  The glasses only lock when the packet
    // stream strictly alternates L/R and the period stays inside the RP2040
    // master-lock window (7600-9000 us).  If the DLL is not feeding a valid
    // stream (wrong refresh, dropped presents, jitter), this shows the real
    // numbers instead of us guessing.
    let mut dbg = StreamStats::default();

    // --- Eye source: the game's ring --------------------------------------
    // The game always tells us which eye (left/right) it is rendering -- the
    // ring carries one concrete eye per present -- so the fired stream IS the
    // game's stream.  We drain the ring into a queue and pop exactly ONE eye
    // per anchored slot (and per no-anchor fallback fire), in the game's exact
    // enqueue order (FIFO); no alternator and no slip tracking is involved
    // (see module docs).  The queue starts empty, and while it is empty at a
    // fire slot we stay SILENT rather than inventing an eye -- `alive` below
    // proves the game is pushing swaps, a non-empty queue proves we know which
    // eye.  Popping per-slot (not taking `.last()`) is what keeps the fired
    // stream alternating when SIMPLE mode lands both eyes of a frame in one
    // drain batch.
    let mut pending = EyeQueue::default();
    pending.stall_hold_after_freerun = stall_hold_after_freerun_slots();
    // Most recent game-present arrival.  Deliberately initialized to a stale
    // time so the host is DEAD at startup: it emits nothing until wiz3D/the
    // game actually pushes a swap, instead of firing a ~300 ms burst the
    // moment it starts (the `alive` gate is "a swap within the last 300 ms").
    // Only a real drained swap makes `alive` true.
    let mut last_swap = Instant::now() - Duration::from_secs(3600);
    // Whether the game was feeding swaps on the previous loop iteration, so we
    // can detect the alive->dead edge and clear the eye phase: a game/scene
    // that stops must not let the NEXT launch inherit a stale, possibly
    // wrong-held eye (which would latch the new stream inverted).
    let mut was_alive = false;

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

    // The fire target (host-epoch vblank to arm the IR for) is owned by the
    // DRM anchor itself (`DrmVblank::next_present_us`), advanced by the SAME
    // `frame_start`/`frame_end` pair the working 3dv3d demo drives the emitter
    // with.  The host must NOT keep a second, independently re-anchored clock
    // here: the hand-rolled copy previously tracked `vblank + period` with a
    // phase-preserving fold that lacked `frame_start`'s grid-snap and
    // `frame_end`'s gentle per-frame phase-lock, so its slot prediction could
    // drift a display slot away from the schedule -- dropping a valid eye one
    // slot late (the `alternating` wobble at full content) and, when it landed
    // a frame behind, opening the shutter on the wrong eye (weak 3D even while
    // "locked").  The anchored arm now calls the anchor's own `frame_start`
    // every slot (snaps forward onto the grid, computes the proven
    // `target - 3000 - lead` fire instant) and `frame_end` after firing
    // (advances one period + sub-slot phase-lock) -- identical to the demo.

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
                    if let Some(lead) = apply_timings_json(&d, rate_hz, anchor_conn.as_deref()) {
                        host_lead_us = lead;
                    }
                    shm.set_status(STATUS_READY);
                    device = Some(d);
                }
            }
        }

        // Watch for a ring-layout change from the producer.  A modern DLL
        // bumps the region to version 2 (16-byte stamped slots) when it
        // starts; an older DLL / legacy install keeps version 1.  Switch
        // scheduling modes on the edge and prune the queue that belongs to the
        // other mode so a stale eye can never mix into the new one.
        let mode = shm.stamped();
        if mode != stamped_mode {
            if mode {
                pending.clear_fallback();
            } else {
                pending.clear_scheduled();
            }
            stamped_mode = mode;
            eprintln!(
                "nvstusb-host: shared ring is {}",
                if mode {
                    "phase-pinned (version-2 stamped swaps)"
                } else {
                    "legacy (version-1 unstamped swaps; FIFO+hold fallback)"
                }
            );
        }

        // Absorb any new swaps the game pushed.  A fresh swap means the game
        // is presenting (keeps us alive) and carries the concrete eye it is
        // rendering; each eye is pinned to the vblank boundary its submit-time
        // stamp resolves to (stamped mode + anchored grid) or appended to the
        // fallback FIFO (see module docs).  A second ingestion pass runs right
        // after the vblank wait so an eye tagged in this very slot is picked
        // up before its own boundary.
        let swaps = shm.drain();
        if !swaps.is_empty() {
            if swaps.len() >= 2 {
                batches_ge2 += 1;
            }
            max_batch = max_batch.max(swaps.len());
            swaps_this_window += swaps.len() as u64;
            last_swap = Instant::now();
            let next_boundary = if stamped_mode {
                drm_anchor.as_ref().and_then(|a| {
                    let p = a.period_us().max(1);
                    // FIFO pinning must land on the SAME boundary the service
                    // will arm this iteration.  The late drain (below) pins to
                    // `vblank_us + period`; here `last_vblank_epoch` is the
                    // PREVIOUS confirmation (one slot behind this iteration's),
                    // so the equal-armed pin is `last_vblank_epoch + 2*period`.
                    // Pinning to `+ period` put these swaps one slot behind the
                    // armed boundary -> `next_eye` dropped them overdue and the
                    // host freeran in their place (the observed top-drain
                    // `front=confirmed` oscillation).
                    last_vblank_epoch.map(|v| v.saturating_add(p.saturating_mul(2)))
                })
            } else {
                None
            };
            pending.ingest(&swaps, drm_anchor.as_ref(), next_boundary);
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

        // Re-anchor when the window owner republishes the target connector.
        // The demo writes winit's current_monitor() name (e.g. `DP-2`) and the
        // wiz3D DLL writes the target monitor's EDID base; either resolves to a
        // concrete DRM connector here.  A change (window moved heads) means the
        // old vblank clock is driving the WRONG display's phase, so drop and
        // re-open the anchor on the new head and re-apply its shutter profile.
        let published = shm.connector_name();
        if published != last_published_conn {
            last_published_conn = published.clone();
            if let Some(target) = resolve_published_connector(&published) {
                let same = anchor_conn
                    .as_ref()
                    .map(|cur| cur.eq_ignore_ascii_case(&target))
                    .unwrap_or(false);
                if !same {
                    derive_anchor_for(&target, &mut drm_anchor, &mut anchor_conn);
                    // The fire schedule was computed on the OLD head's grid --
                    // a different CRTC free-runs with its own phase.  Reset the
                    // schedule so the first wait on the new head re-derives the
                    // grid and force the anchor to full-resync: its internal
                    // prediction was on the old head's grid and must re-anchor
                    // on the new head's confirmed vblanks.
                    last_vblank_epoch = None;
                    pending.reset();
                    if let Some(a) = drm_anchor.as_mut() {
                        a.force_resync();
                    }
                    eprintln!(
                        "nvstusb-host: window moved to {} (published {:?}); re-anchored",
                        target, published
                    );
                    if let Some(d) = device.as_ref() {
                        if let Some(lead) =
                            apply_timings_json(d, rate_hz, anchor_conn.as_deref())
                        {
                            host_lead_us = lead;
                        }
                    }
                }
            }
        }

        if let Some(d) = device.as_ref() {
            // Pick up config changes from the DLL.
            let new_rate = shm.rate_hz();
            if new_rate > 60.0 && (new_rate - last_rate).abs() > 0.01 {
                let _ = d.configure(new_rate);
                // Re-apply the matching monitor_timings.json profile for the new
                // rate (configure resets the shutter timing registers to the
                // reference values).
                if let Some(lead) = apply_timings_json(d, new_rate, anchor_conn.as_deref()) {
                    host_lead_us = lead;
                }
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

            // On the alive->dead edge, clear the entire eye phase (queued eyes
            // AND the held eye).  A game/exe that quits often restarts or is
            // followed by a different one; if we kept `last_fired` and the
            // queued backlog, the next stream could inherit a stale phase and
            // latch INVERTED from its first slot.  Clearing returns the queue
            // to its true, uncontacted state so the next launch's first fresh
            // present establishes a clean anchor.
            if was_alive && !alive {
                pending.reset();
            }
            was_alive = alive;

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
                let down = rb[6] & 0x01 != 0;
                // Toggle only on the 0->1 EDGE so a spurious single high-bit
                // read (USB glitch) or a held-down button cannot latch a
                // permanent inversion.
                if down && !button_was_down {
                    if stamped_mode {
                        // Stamped (phase-pinned) session: the button now
                        // commands the GAME to swap its rendered/reported eye
                        // instead of shifting the shutter phase.  The old
                        // behavior re-pinned the whole schedule by +/- one
                        // period (`phase_slots` 0<->1) on the GLASSES side,
                        // which held one eye for the extra slot -- a longer
                        // dark/black period for the viewer that is distinctly
                        // uncomfortable.  Telling the DLL to swap left/right
                        // (via the FLAG_INVERT_EYES shm bit) keeps the glasses
                        // shuttering at their natural cadence: no phase shift,
                        // no extra dark.
                        let new_state = !shm.invert_eyes();
                        if new_state {
                            shm.set_invert_eyes();
                            eprintln!("nvstusb-host: emitter button -> game swaps L/R eyes (FLAG_INVERT_EYES set)");
                        } else {
                            shm.clear_invert_eyes();
                            eprintln!("nvstusb-host: emitter button -> game restores native L/R eyes");
                        }
                    } else {
                        manual_invert = !manual_invert;
                        eprintln!(
                            "nvstusb-host: emitter button -> eye inversion {}",
                            if manual_invert { "ON" } else { "OFF" }
                        );
                    }
                }
                button_was_down = down;
            }

            match drm_anchor.as_mut() {
                // --- Hardware-vblank anchored emission -------------------
                // One packet per real vblank slot, timed to the display vblank
                // clock; the eye is popped FIFO from the game's ring -- see
                // module docs.
                Some(anchor) if alive => {
                    match anchor.wait_vblank_blocking() {
                        Some(vblank_us) => {
                            let first_grid = last_vblank_epoch.is_none();
                            last_vblank_epoch = Some(vblank_us);
                            if first_grid && stamped_mode {
                                // The first confirmed boundary establishes the
                                // grid; startup swaps that landed before it were
                                // necessarily FIFO'd -- drop them so the
                                // stamped schedule (whose targets are
                                // grid-exact) becomes the only source.
                                pending.clear_fallback();
                            }
                            // Late-ingest pass (count accuracy + accurate slot
                            // pinning): swaps pushed since the top-of-loop
                            // drain are absorbed HERE, at the boundary itself,
                            // so an eye tagged inside this very frame is pinned
                            // before the grid reference moves on.
                            let late = shm.drain();
                            if !late.is_empty() {
                                if late.len() >= 2 {
                                    batches_ge2 += 1;
                                }
                                max_batch = max_batch.max(late.len());
                                swaps_this_window += late.len() as u64;
                                last_swap = Instant::now();
                                let next_boundary = vblank_us.saturating_add(anchor.period_us());
                                pending.ingest(&late, Some(anchor), Some(next_boundary));
                            }
                            // Pace the fire target with the anchor's OWN
                            // `frame_start`/`frame_end` -- the exact pair the
                            // working 3dv3d demo uses to drive the emitter
                            // (and, below, the reference both the old full-
                            // resync and the hand-rolled `next_present_us`
                            // clock drifted from).  `frame_start` SNAPS the
                            // predicted present forward in whole periods until
                            // it is still reachable (`target - alarm - lead -
                            // margin` in the future), so the armed boundary is
                            // always one whole slot ahead on the display grid
                            // and can never sit on a stale/duplicate slot;
                            // it then returns the proven fire Instant
                            // (target - 3000 - lead).  The old hand-rolled
                            // `vblank + period` fold had no such snap, so a
                            // whole-period vblank report (or a scheduling
                            // hiccup) could park the target one slot behind
                            // the schedule -- dropping a valid eye as
                            // "overdue" (the `alternating` wobble at full
                            // content) and, when it sank a full frame behind,
                            // opening the shutter on the wrong eye.
                            let period = anchor.period_us();
                            if pending.has_eye() {
                                // Snaps + computes the fire deadline; advances
                                // the anchor's prediction one period and
                                // phase-locks it (called unconditionally so
                                // the grid keeps marching even across the
                                // defensive no-fire below).
                                let fire_at = anchor.frame_start(host_lead_us as u32);
                                let target = anchor.current_present_us();
                                // TEMP DEBUG
                                {
                                    use std::sync::atomic::{AtomicU64, Ordering};
                                    static CNT: AtomicU64 = AtomicU64::new(0);
                                    if CNT.fetch_add(1, Ordering::Relaxed) % 120 == 0 {
                                        let front = pending.front_boundary();
                                        eprintln!(
                                            "DBG confirmed={vblank_us} armed={target} invert={} \
                                             front={front:?} gap_from_armed={:?}",
                                            shm.invert_eyes() as u8,
                                            front.map(|f| target as i64 - f as i64)
                                        );
                                    }
                                }
                                if let Some(eye) = pending.next_eye(target, period) {
                                    if let Some(fire_at) = fire_at {
                                        while Instant::now() < fire_at {
                                            std::hint::spin_loop();
                                        }
                                    }
                                    // manual_invert applies at the wire only
                                    // (and only matters in fallback mode).
                                    let out = eye != manual_invert;
                                    d.send_eye(out, rate_hz);
                                    dbg.record(out);
                                }
                                // Advance the anchor's prediction exactly like
                                // the demo's post-swap `frame_end` (one period
                                // + sub-slot phase-lock).  Confirm against the
                                // kernel vblank this wait just returned: the
                                // armed boundary is exactly one period ahead,
                                // so the phase-preserving normalization folds
                                // that -period to ~0 (no spurious nudge) --
                                // keeping the target pinned to the display
                                // grid.  Confirming against wall-clock `now`
                                // (mid-frame, `target - alarm - lead` before
                                // the boundary) would bake a -3250us error
                                // into every frame and drag the target
                                // off-grid.
                                anchor.frame_end(Some(vblank_us));
                            }
                        }
                        None => {
                            // Anchor died: fire immediately to stay live, with
                            // whatever eye the game has queued (or held).  No
                            // grid, so boundaries are meaningless: pop the
                            // oldest known eye (scheduled or fallback).
                            if let Some(eye) = pending.next_eye_present_driven() {
                                let out = eye != manual_invert;
                                d.send_eye(out, rate_hz);
                                dbg.record(out);
                            }
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
                // tuning here.  The eye still comes from the game's ring.
                None => {
                    if alive {
                        if let Some(eye) = pending.next_eye_present_driven() {
                            let out = eye != manual_invert;
                            d.send_eye(out, rate_hz);
                            dbg.record(out);
                        }
                    }
                }
            }

            dbg.tick(
                &mut swaps_this_window,
                &mut batches_ge2,
                &mut max_batch,
                &mut lead_n,
                &mut lead_sum,
                &mut lead_min,
                &mut lead_max,
                pending.drops,
                if stamped_mode { "phase-pinned" } else { "fifo" },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shm::{EYE_LEFT, EYE_RIGHT};

    /// `boundary_after` is the core mapping that kills the phase lottery: every
    /// stamped submit time resolves to the ONE display boundary the present
    /// blocks to.  Mid-period submits land on their own boundary; submits at or
    /// after a boundary land on the next one (strictly after).
    #[test]
    fn boundary_after_maps_submit_time_to_scanout_slot() {
        let refb = 100_000u64; // a confirmed boundary (host-epoch us)
        let p = 8_333u64;
        // Grid: 100000, 108333, 116666, 124999, ...
        // Mid-period submit -> its own (next) boundary.
        assert_eq!(boundary_after(refb, p, 105_000), 108_333);
        // A submit right after a boundary -> the FOLLOWING boundary (that
        // present missed the one it landed after).
        assert_eq!(boundary_after(refb, p, 108_334), 116_666);
        assert_eq!(boundary_after(refb, p, 112_500), 116_666);
        // Exactly on a boundary (d = k*p): strictly after -> next one.
        assert_eq!(boundary_after(refb, p, 100_000), 108_333);
        assert_eq!(boundary_after(refb, p, 124_999), 133_332);
        // Stale submit earlier than the (already passed) reference boundary
        // resolves to the actual slot that present scanned out on -- the
        // boundary just before `t`-ward, here 100000 (t=99000 lies in the
        // (91667, 100000] slot).  `next_eye` then drops it as overdue rather
        // than replaying it onto the wrong (future) slot.
        assert_eq!(boundary_after(refb, p, 99_000), 100_000);
        // A submit a couple of slots stale resolves to its own past slot too,
        // independent of which boundary serves as the grid origin.
        assert_eq!(boundary_after(refb, p, 90_000), 91_667); // (83334, 91667] slot
        assert_eq!(boundary_after(refb, p, 80_000), 83_334); // (75001, 83334] slot
        // Degenerate period guard must not panic / return 0.
        let p0 = 0u64;
        let r = boundary_after(refb, p0, 105_000);
        assert!(r > 105_000);
    }

    /// The fix's headline: each stamped eye, however its submit is offset
    /// across a stream restart, resolves to the SAME display slot as the eye
    /// the present actually scans out on.  Two different start offsets of the
    /// same (L,R) timeline must pin each eye to a deterministic boundary --
    /// there is no second stable solution, which is exactly what the old
    /// FIFO/`next_present_us` re-derivation could latch on.
    #[test]
    fn stamped_submits_pin_deterministic_slots_across_restarts() {
        let refb = 200_000u64; // last confirmed boundary
        let p = 8_333u64;
        // Game stream A: R submitted mid-slot, L submitted mid-next-slot.
        let r_a = boundary_after(refb, p, 205_000);
        let l_a = boundary_after(refb, p, 213_400);
        let r_b = boundary_after(refb, p, 213_333 + 100);
        let l_b = boundary_after(refb, p, 221_700);
        assert_eq!(r_a, 208_333);
        assert_eq!(l_a, 216_666);
        // Each submit pins to its own boundary regardless of stream phase.
        assert_eq!(r_b, 216_666);
        assert_eq!(l_b, 224_999);
        // Firing each eye when ITS boundary arrives reproduces the game stream
        // in scanout order -- no phase can go missing or double-fire.
        let mut q = EyeQueue::default();
        q.enqueue(true, Some(r_a), p); // R
        q.enqueue(false, Some(l_a), p); // L
        assert_eq!(q.next_eye(208_333, p), Some(true)); // R on its own slot
        assert_eq!(q.next_eye(216_666, p), Some(false)); // L on its own slot
        // Empty slot after the pinned stream: nothing else is pinned, so the
        // host FREERUNS (toggles to the opposite eye) to keep the glasses
        // shuttering/alternating -- lock holds through the gap.
        assert_eq!(q.next_eye(224_999, p), Some(true));
    }

    /// A dip (a slot whose eye never arrived, or a boundary whose enqueue was
    /// lost) FREERUNS to the opposite eye so the glasses keep alternating (lock
    /// holds) and NEVER shifts the phase -- the next stamped eye still fires on
    /// its own boundary.
    #[test]
    fn dip_holds_phase_and_never_shifts_a_slot() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        // R pinned to b, L pinned to b+3p: two slots in between have no new eye.
        q.enqueue(true, Some(b), p);
        q.enqueue(false, Some(b + 3 * p), p);
        assert_eq!(q.next_eye(b, p), Some(true)); // R on its own boundary
        // Dips (slots with no new eye) FREERUN: toggle to the opposite eye so
        // the glasses keep alternating and lock holds.  The next stamped eye
        // still fires on its own boundary -- never a whole-stream phase shift.
        assert_eq!(q.next_eye(b + p, p), Some(false)); // dip: freerun -> L
        assert_eq!(q.next_eye(b + 2 * p, p), Some(true)); // dip: freerun -> R
        assert_eq!(q.next_eye(b + 3 * p, p), Some(false)); // L: own boundary
        assert_eq!(q.next_eye(b + 4 * p, p), Some(true)); // dip: freerun -> R
    }

    /// A GENUINE stall (a stuck/stale frame, e.g. a load hitch or menu) is
    /// not the same as a one- or two-slot blip: past
    /// `STALL_HOLD_AFTER_FREERUN_SLOTS` consecutive freerun fires, the host
    /// must stop toggling and hold the last eye instead of strobing a frame
    /// that provably has not updated.  The moment a real eye returns, normal
    /// firing resumes on its own boundary (no phase shift either way).
    #[test]
    fn sustained_stall_holds_instead_of_strobing() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        q.enqueue(true, Some(b), p);
        assert_eq!(q.next_eye(b, p), Some(true)); // R, real eye, holds the phase

        // The next STALL_HOLD_AFTER_FREERUN_SLOTS dips still toggle (short
        // jitter -- keep the glasses' hardware lock alive).
        let mut slot = b;
        let mut expect_right = false; // R just fired, so the next toggle is L
        for _ in 0..STALL_HOLD_AFTER_FREERUN_SLOTS {
            slot += p;
            assert_eq!(q.next_eye(slot, p), Some(expect_right));
            expect_right = !expect_right;
        }

        // Every dip past the threshold must hold -- same eye, no more
        // toggling -- for as long as the stall lasts.
        let held = q.last_fired;
        for _ in 0..20 {
            slot += p;
            assert_eq!(q.next_eye(slot, p), held, "must hold, not strobe, during a real stall");
        }

        // A real eye returning immediately breaks the hold and fires on its
        // own boundary, exactly as an ordinary dip-recovery would.
        slot += p;
        q.enqueue(false, Some(slot), p);
        assert_eq!(q.next_eye(slot, p), Some(false));
        assert_eq!(q.freerun_since, 0, "real eye clears the freerun/stall counter");
    }

    /// An eye whose boundary has already passed (host fell behind, or the
    /// enqueue was dropped) is dropped -- that display slot is already on
    /// screen.  The remaining eyes keep their own boundaries: no whole-stream
    /// shift, no re-lotting.
    #[test]
    fn overdue_eye_dropped_without_shifting_later_eyes() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        q.enqueue(true, Some(b - p), p); // overdue: its slot (b-p) already passed
        q.enqueue(false, Some(b), p);
        q.enqueue(true, Some(b + p), p);
        // Servicing b: the overdue orphan (target b-p) is dropped, the eye that
        // belongs to b is fired.
        assert_eq!(q.next_eye(b, p), Some(false));
        assert_eq!(q.next_eye(b + p, p), Some(true));
        assert_eq!(q.next_eye(b + 2 * p, p), Some(false)); // empty -> freerun L
    }

    /// THE overdue-inversion regression: at full 1:1 rate the FIFO-pinned
    /// schedule can drift a sub-period OFF the armed grid (front falls behind
    /// `armed` ~50us/sec until it trips the half-period overdue threshold).
    /// Simply dropping the overdue eye is NOT enough -- the remaining eyes are
    /// still off-grid, and firing them at their drifted offsets lands one eye
    /// on the WRONG slot = a shifted (inverted) stream that never recovers (the
    /// `overdue=1` eye-swap symptom).  The overdue drop MUST re-baseline the
    /// remaining schedule onto the armed grid so the next fire is deterministic
    /// and self-correcting: the drifted front (somewhere in the +/-half band
    /// just past `boundary`) is snapped back to the grid, sub-period error only,
    /// never a whole display slot.
    #[test]
    fn overdue_drop_rebaselines_the_schedule_onto_the_grid() {
        let p = 8_333u64;
        let b = 100_000u64;
        let drift = 4_000u64; // a realistic sub-period drift off the grid
        let mut q = EyeQueue::default();
        // Schedule has drifted +drift: the eye "for b" actually sits at
        // b+drift, the next at b+p+drift, etc.  (The eye that should have
        // scanned out at b-p already did and is overdue.)
        q.enqueue(true, Some(b - p), p); // overdue orphan -> dropped
        q.enqueue(false, Some(b + drift), p); // drifted front eye
        q.enqueue(true, Some(b + p + drift), p); // drifted next eye
        assert!(!q.pending_reanchor);

        // Servicing b: the orphan is dropped overdue and the schedule is
        // re-baselined onto the armed grid (front snapped from b+drift back to
        // b -- sub-period error only, no whole-slot step), and the re-anchor is
        // flagged for the next fresh DLL eye.
        assert_eq!(q.next_eye(b, p), Some(false), "grid-aligned L fires due on b");
        assert_eq!(q.pending_reanchor, true, "overdue drop must flag a re-anchor");
        assert_eq!(
            q.front_boundary(),
            Some(b + p),
            "re-baselined front sits exactly one period on the grid (b+p, not b+p+drift)"
        );
        // Every eye that continues to fire is grid-aligned: R on b+p, then the
        // empty schedule freeruns from the last real anchor -- never shifted by
        // the stale drift, so polarity cannot latch inverted.
        assert_eq!(q.next_eye(b + p, p), Some(true), "grid-aligned R fires on b+p");
        assert_eq!(q.next_eye(b + 2 * p, p), Some(false), "clean freerun from the re-anchored phase");
        assert_eq!(q.next_eye(b + 3 * p, p), Some(true), "alternation preserved, no stale drift");
    }

    /// THE freerun-inversion regression: when the game under-presents, the host
    /// FREERUNS invented toggle eyes whose parity is arbitrary relative to the
    /// game's real stream.  Left alone, the resumed stream inherits that parity
    /// and latches INVERTED ~50/50 (the `freerun=7` eye-swap symptom).  Every
    /// freerun must therefore flag `pending_reanchor`, and the FIRST fresh,
    /// real DLL eye must consume it -- re-pinning the held phase to the game's
    /// true eye (so the resumed stream alternates off it deterministically).
    #[test]
    fn freerun_marks_pending_reanchor_consumed_by_next_real_eye() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        // Real L holds the phase.
        q.enqueue(false, Some(b), p);
        assert_eq!(q.next_eye(b, p), Some(false));
        assert!(!q.pending_reanchor);

        // The game skips a slot: the host freeruns an invented eye, which
        // breaks its trust in the held parity.
        q.next_eye(b + p, p);
        assert_eq!(
            q.pending_reanchor, true,
            "a freerun must flag the phase as needing re-anchor"
        );

        // The game resumes with a fresh real eye.  It consumes the flag and
        // re-pins the held phase to the game's own eye -- NOT the freerun's
        // invented parity -- so post-dip parity is deterministic.
        assert!(q.pending_reanchor, "flag still pending before the fresh eye");
        q.enqueue(true, Some(b + 2 * p), p); // fresh real R
        assert_eq!(
            q.pending_reanchor, false,
            "the first fresh real eye consumes the pending re-anchor"
        );
        assert_eq!(
            q.last_fired,
            Some(true),
            "held phase re-pinned to the game's eye (R), not the freerun's"
        );
    }

    /// Dip-recovery: an fps dip empties the schedule and the host FREERUNS
    /// invented eyes to hold lock; the held `last_fired` is then arbitrary
    /// game-relative parity.  The first fresh DLL-reported eye after that
    /// substantive gap MUST re-anchor the held phase to the game's own eye --
    /// otherwise the resumed stream inherits the freerun parity and can latch
    /// INVERTED (the post-dip inversion).  Pinning is absolute, so the re-
    /// anchor is deterministic, never the old ~50/50 re-lot.
    #[test]
    fn dip_recovery_reanchors_held_phase_to_dll_eye() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        // One real pair fires, then the game dips: schedule empties and the
        // host freeruns for many slots (invented eyes, toggling held phase).
        q.enqueue(false, Some(b), p); // L at b
        q.enqueue(true, Some(b + p), p); // R at b+p
        assert_eq!(q.next_eye(b, p), Some(false));
        assert_eq!(q.next_eye(b + p, p), Some(true));
        // Freerun across the dip: toggle L,R,L,R...
        let mut slots_into_dip = 0u32;
        let mut boundary = b + 2 * p;
        loop {
            let eye = q.next_eye(boundary, p).expect("must keep shuttering");
            // Glasses keep alternating during the dip.
            slots_into_dip += 1;
            boundary += p;
            if slots_into_dip >= DIP_RESYNC_AFTER_FREERUN_SLOTS {
                assert_ne!(q.freerun_since, 0, "dip has produced freerun fires");
                break;
            }
            let _ = eye;
        }
        // The game resumes and reports a fresh REAL eye at the next slot.  The
        // schedule was empty (all consumed) and we had been freerunning, so the
        // recovery MUST re-anchor the held phase to this DLL eye (post-dip
        // inversion guard) and clear the gap: the freerun parity that strayed
        // during the dip is statically replaced, deterministically (pinning is
        // absolute, never the old ~50/50 re-lot).
        q.enqueue(false, Some(boundary), p); // fresh L from the DLL
        assert_eq!(
            q.last_fired,
            Some(false),
            "dip-recovery pre-seeds the held phase to the DLL eye"
        );
        assert_eq!(
            q.freerun_since, 0,
            "first resumed DLL eye clears the freerun gap"
        );
        // The resumed stream continues pinned from the DLL's data: the held
        // phase is anchored to L, so L fires on its own boundary and the next
        // slot alternates off it.
        assert_eq!(q.next_eye(boundary, p), Some(false)); // DLL's L on its own slot
        assert_eq!(q.next_eye(boundary + p, p), Some(true)); // freerun from the real anchor
    }

    /// A brief one-slot blip (schedule momentarily empty between two frames)
    /// must NOT count as a dip: `freerun_since` stays below the resync
    /// threshold so no spurious re-anchor happens in normal operation.
    #[test]
    fn short_blip_is_not_a_dip() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        q.enqueue(false, Some(b), p); // L
        assert_eq!(q.next_eye(b, p), Some(false)); // real fire -> authoritative
        assert_eq!(q.freerun_since, 0);
        assert_eq!(q.next_eye(b + p, p), Some(true)); // one blip: freerun -> R
        assert_eq!(q.freerun_since, 1);
        // Real data returns immediately, before any resync threshold: the
        // blip does NOT re-anchor (no spurious phase clobber), and the real
        // pinned eye fires AS REPORTED -- the host never invents or flips a
        // real eye, so the game's R is emitted unchanged even though the blip
        // just freeran an R (an occasional same-eye is the accepted trade-off
        // against ever firing the wrong eye / inverting polarity).
        q.enqueue(true, Some(b + 2 * p), p);
        assert_eq!(q.last_fired, Some(true)); // freerun parity untouched (no re-anchor)
        assert_eq!(q.next_eye(b + 2 * p, p), Some(true)); // real R fires as reported
        assert_eq!(q.freerun_since, 0);
    }

    /// Regression for the "eyes swapped after a frame dip" symptom: a SHORT dip
    /// of only two slots (a common game-stutter -- under the old 6-slot / ~50 ms
    /// threshold, which left these unrecovered) must STILL re-anchor the held
    /// phase to the game's first fresh real eye.  After two invented freerun
    /// fires the freerun parity is arbitrary relative to the game's own frame
    /// counter, exactly ~50/50 wrong afterwards -- re-anchoring makes every
    /// post-dip resume deterministic instead.
    #[test]
    fn short_dip_still_reanchors_held_phase() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        // Real L fires and holds the phase.
        q.enqueue(false, Some(b), p);
        assert_eq!(q.next_eye(b, p), Some(false));
        assert_eq!(q.freerun_since, 0);
        // A two-slot dip: two invented freerun fires (R then L), so
        // `freerun_since` crosses the (now lowered) resync threshold.
        assert_eq!(q.next_eye(b + p, p), Some(true)); // freerun -> R
        assert_eq!(q.freerun_since, 1);
        assert_eq!(q.next_eye(b + 2 * p, p), Some(false)); // freerun -> L
        assert_eq!(q.freerun_since, 2);
        // The game resumes reporting a REAL eye whose parity does not match
        // the freerun's held L (it reports R).  The dip-recovery MUST re-anchor
        // the held phase to this DLL eye or the resumed stream latches inverted.
        q.enqueue(true, Some(b + 3 * p), p); // fresh real R from the DLL
        assert_eq!(
            q.last_fired,
            Some(true),
            "short dip re-anchors held phase to the DLL eye (no more post-dip inversion)"
        );
        assert_eq!(q.freerun_since, 0, "first resumed DLL eye clears the gap");
        // The resumed stream continues pinned from the DLL's data: real R fires
        // on its own slot, then alternates off it.
        assert_eq!(q.next_eye(b + 3 * p, p), Some(true)); // DLL's R on its own slot
        assert_eq!(q.next_eye(b + 4 * p, p), Some(false)); // freerun from the real anchor
    }

    /// Two eyes that resolve to the SAME boundary are consecutive presents of
    /// one frame-sequential pair (both block before one vblank), so they are
    /// spaced onto SUCCESSIVE display slots in submission order -- otherwise the
    /// older eye is dropped and the fired stream never alternates ("both eyes
    /// of equal strength").
    #[test]
    fn same_boundary_collision_fires_oldest_then_later() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        q.enqueue(true, Some(b), p); // R submitted first -> owns boundary b
        q.enqueue(false, Some(b), p); // L submitted next -> bumped to b+p
        assert_eq!(q.next_eye(b, p), Some(true), "first (R) owns boundary b");
        assert_eq!(q.next_eye(b + p, p), Some(false), "second (L) on b+p");
    }

    /// A genuine backlog (host fell behind) drops the stale schedule and
    /// re-anchors on the NEWEST stamped present that caused the overflow --
    /// whose target is a concrete boundary, not a `.last()` parity guess.
    /// Fresh swaps that arrive after the re-anchor continue the schedule from
    /// that anchor point.
    #[test]
    fn backlog_reanchors_on_newest_stamped_boundary() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        for i in 0..MAX_QUEUE + 4 {
            q.enqueue(i % 2 == 0, Some(b + i as u64 * p), p);
        }
        // The overflow (i == MAX_QUEUE) dropped the stale prefix (i = 0..7):
        // nothing remains pinned at the old start of the schedule.
        assert_eq!(q.next_eye(b, p), None); // ancient prefix is gone
        // The stream resumes pinned at the newest survived boundary, concrete
        // from its own stamp, and continues with the swaps after the anchor.
        let anchor = b + MAX_QUEUE as u64 * p;
        assert_eq!(q.next_eye(anchor, p), Some(true)); // i=8 (even -> R)
        assert_eq!(q.next_eye(anchor + p, p), Some(false)); // i=9 -> L
        assert_eq!(q.next_eye(anchor + 2 * p, p), Some(true)); // i=10 -> R
    }

    /// The fallback path (legacy version-1 ring: no stamps) pops the oldest
    /// eye per slot in order, FREERUNS (toggles) across empty slots so the
    /// glasses keep shuttering (lock holds), and preserves order across
    /// interleaved drains.
    #[test]
    fn fallback_fifo_holds_and_preserves_ring_order() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        q.enqueue(true, None, p); // R
        q.enqueue(false, None, p); // L
        q.enqueue(true, None, p); // R
        q.enqueue(false, None, p); // L
        assert_eq!(q.next_eye(b, p), Some(true));
        assert_eq!(q.next_eye(b + p, p), Some(false));
        assert_eq!(q.next_eye(b + 2 * p, p), Some(true));
        assert_eq!(q.next_eye(b + 3 * p, p), Some(false));
        // Empty => FREERUN (toggle to the opposite eye) so the glasses keep
        // alternating and lock holds.
        assert_eq!(q.next_eye(b + 4 * p, p), Some(true));
        // Interleaved drain keeps strict FIFO order: the real eyes pop as
        // reported (host never flips a real eye, preserving polarity), even if
        // that occasionally follows a freerun with a same-eye.
        q.enqueue(true, None, p);
        q.enqueue(false, None, p);
        assert_eq!(q.next_eye(b + 5 * p, p), Some(true)); // real R pops as reported
        assert_eq!(q.next_eye(b + 6 * p, p), Some(false)); // real L pops as reported
        // Queues drained -> back to freerun (toggle off the held L -> R).
        assert_eq!(q.next_eye(b + 7 * p, p), Some(true));
    }

    /// Fallback-backlog overflow still re-anchors on the newest present at the
    /// overflow moment (the legacy spelling of the same rule); swaps arriving
    /// after the anchor continue in order.
    #[test]
    fn fallback_backlog_reanchors_on_newest() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        for i in 0..MAX_QUEUE + 2 {
            q.enqueue(i % 2 == 0, None, p);
        }
        // i=0..9.  The overflow at i=8 re-anchors the LEGACY way: the stale
        // prefix i=0..7 is dropped and the newest present (i=8, R) becomes the
        // anchor (queued + held).  i=9 (L) then queues after it.  The anchored
        // R fires as reported (the held phase is the reference for the next
        // slot), preserving polarity: R, then L, then the drained queue
        // freeruns off the held L.
        assert_eq!(q.next_eye(b, p), Some(true)); // anchor eye i=8 (R)
        assert_eq!(q.next_eye(b + p, p), Some(false)); // i=9 (L)
        assert_eq!(q.next_eye(b + 2 * p, p), Some(true)); // empty -> freerun R
    }

    /// Before the ring reports an eye the host must stay silent; the
    /// present-driven path (no grid) pops the oldest known eye regardless of
    /// target.
    #[test]
    fn empty_queue_is_silent_and_present_driven_pops_oldest() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        assert!(!q.has_eye());
        assert_eq!(q.next_eye(b, p), None);
        assert_eq!(q.next_eye_present_driven(), None);
        // Fallback append then present-driven pop.
        q.enqueue(false, None, p);
        assert_eq!(q.next_eye_present_driven(), Some(false));
        // A future-pinned eye is also popped oldest-first off-grid (degraded
        // path: boundaries are meaningless without the grid).
        q.enqueue(true, Some(b + 10 * p), p);
        assert_eq!(q.next_eye_present_driven(), Some(true));
        // The held eye keeps firing.
        assert_eq!(q.next_eye_present_driven(), Some(true));
    }

    /// The phase_slots button: re-pinning the schedule by +/- one period moves
    /// every pinned eye by exactly one slot and keeps their order.
    #[test]
    fn shift_schedule_repins_targets_on_phase_toggle() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        q.enqueue(true, Some(b), p);
        q.enqueue(false, Some(b + p), p);
        q.shift_schedule(p as i64); // 0 -> 1
        assert_eq!(q.next_eye(b, p), None); // nothing due at b anymore
        assert_eq!(q.next_eye(b + p, p), Some(true));
        q.shift_schedule(-(p as i64)); // 1 -> 0
        assert_eq!(q.next_eye(b + p, p), Some(false)); // L now back at b+p
    }

    /// `reset()` leaves the queue untouched (alive->dead edge + re-anchor): no
    /// queued/scheduled eye, no held eye -- the next stream starts from a clean
    /// phase reference.
    #[test]
    fn reset_clears_held_and_scheduled_phase() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        q.enqueue(true, Some(b), p);
        q.enqueue(false, None, p);
        assert!(q.has_eye());
        q.reset();
        assert!(!q.has_eye(), "reset must clear queued, scheduled AND held eyes");
        assert_eq!(q.next_eye(b, p), None);
        assert_eq!(q.next_eye_present_driven(), None);
        // A fresh stream anchors cleanly on its own first stamps.
        q.enqueue(false, Some(b), p);
        assert_eq!(q.next_eye(b, p), Some(false));
    }

    /// ingest() wires a drained swap into the right queue: stamped swaps with a
    /// grid get pinned boundaries; unstamped swaps (legacy region) go FIFO.
    #[test]
    fn ingest_demuxes_stamped_and_legacy_swaps() {
        let b = 100_000u64;
        let p = 8_333u64;
        let mut q = EyeQueue::default();
        // No anchor/grid: even a stamped swap cannot be pinned -> FIFO fallback.
        q.ingest(&[crate::shm::Swap { eye: EYE_LEFT, t_us: Some(105_000) }], None, None);
        assert_eq!(q.next_eye(b, p), Some(false)); // left
        // Legacy drain (t_us None) -> FIFO.
        q.ingest(&[crate::shm::Swap { eye: EYE_RIGHT, t_us: None }], None, Some(b));
        assert_eq!(q.next_eye(b + p, p), Some(true));
    }

    /// Frame-sequential SIMPLE content submits both eyes of one 60fps frame
    /// back-to-back at the DDI layer before the shared vblank they collectively
    /// block to, so their submit timestamps collapse onto the SAME boundary.
    /// Without re-spacing, `next_eye`'s "newest wins the slot" drops the older
    /// eye and the fired stream never alternates (all-R / all-L) -> "both eyes
    /// of equal strength" ghosting.  `enqueue` must re-space collided eyes one
    /// period apart in submission order so L/R alternate on successive slots --
    /// and it must do so ACROSS drains (each eye enqueued separately), which a
    /// per-drain re-spacer would miss.
    #[test]
    fn ingest_respaces_same_boundary_eyes_and_alternates() {
        let p = 8_333u64;
        let mut q = EyeQueue::default();
        // Frame 1 pair: L and R both resolve to boundary 108333 (submitted
        // before one shared vblank).  Enqueued in SEPARATE drains to prove the
        // cross-drain case is handled.  Second eye bumps to 108333 + p.
        q.enqueue(false, Some(108_333), p); // L (drain 1)
        q.enqueue(true, Some(108_333), p); //  R (drain 2) -> bumped to 116_666
        // Frame 2 pair: resolves to 124999, two vblanks later (60fps content).
        q.enqueue(false, Some(124_999), p); // L -> 124_999
        q.enqueue(true, Some(124_999), p); //  R -> bumped to 133_332
        // Consume one eye per slot: must alternate L,R,L,R.
        assert_eq!(q.next_eye(108_333, p), Some(false), "L on first slot");
        assert_eq!(q.next_eye(116_666, p), Some(true), "R on next slot");
        assert_eq!(q.next_eye(124_999, p), Some(false), "L on next slot");
        assert_eq!(q.next_eye(133_332, p), Some(true), "R on next slot");
    }
    /// The host's phase-preserving fire-target re-anchor must NEVER correct a
    /// whole display slot.  The blocking WAIT_VBLANK result can be reported a
    /// whole period early-or-late depending on where in the cycle the ioctl was
    /// called (the boundary race); a naive `vblank + period` would then step the
    /// schedule one slot = fire the eye on the WRONG slot = inversion.  The
    /// `phase_preserving_step` normalization guarantees a <=1-period-off report
    /// is corrected to ~0 (phase fully preserved), never stepped a whole slot.
    #[test]
    fn host_phase_preserving_target_never_steps_a_slot() {
        use crate::nvstusb::drm::phase_preserving_step;
        let period = 8333u64;
        let half = (period / 2) as i64;
        // The schedule currently points at `target`.  The CORRECT current vblank
        // (one period behind the target) is `target - period`; the driver may
        // report it early/late by a whole period relative to that grid.
        let target = 41_665u64;
        let correct_vblank = target - period; // 33332
        for report in [
            correct_vblank,                 // exactly on the correct grid
            correct_vblank - period,        // reported one period EARLY
            correct_vblank + period,        // reported one period LATE
            correct_vblank + 3,             // +3us real drift
        ] {
            // Mirrors host.rs: `desired = vblank + period`, then phase-preserved
            // against the existing target.
            let desired = report.saturating_add(period);
            let step = phase_preserving_step(target, desired, period);
            assert!(
                step.abs() <= half,
                "re-anchor stepped toward the next/previous slot: report={report} step={step}"
            );
        }
        // A whole-period-early or whole-period-late report is corrected to a
        // ZERO step (slot fully preserved) -- the exact boundary race that used
        // to flip the eye.
        assert_eq!(
            phase_preserving_step(target, correct_vblank - period + period, period),
            0,
            "early report kept the slot"
        );
        assert_eq!(
            phase_preserving_step(target, correct_vblank + period + period, period),
            0,
            "late report kept the slot"
        );
    }

    /// Pins the anchored loop's steady-state math: each kernel-confirmed vblank
    /// must move the fire target EXACTLY one period forward.  The loop is
    /// phase-preserving, but it first dead-reckons the prediction one period
    /// forward (`advanced = prev + period`, mirroring `drm.rs`'s `frame_end`),
    /// so the legitimate one-slot progression survives the +/-half fold.  The
    /// regression this guards: without the advance, `prev == confirmed` and
    /// `desired == confirmed + period` look exactly like a whole-slot report
    /// error and the fold returns a ZERO step -- the target freezes on the
    /// just-confirmed vblank, so every eye after the first fires immediately
    /// after the wait (glasses open ~3000us into the next frame) instead of
    /// `period - alarm - lead` before the NEXT boundary.
    #[test]
    fn anchored_loop_advances_one_slot_per_confirmed_vblank() {
        use crate::nvstusb::drm::phase_preserving_step;
        let p = 8_333u64;
        let mut confirmed = 100_000u64; // first wait returns B0
        let mut target: Option<u64> = None;
        for i in 0..8u64 {
            // The host anchored arm's re-anchor, verbatim: advance the previous
            // target one period, fold toward `confirmed + period`.
            let advanced = target.map(|t| t.saturating_add(p));
            let desired = confirmed.saturating_add(p);
            target = Some(match advanced {
                None => desired,
                Some(prev) => {
                    let step = phase_preserving_step(prev, desired, p);
                    (prev as i64 + step).max(0) as u64
                }
            });
            let t = target.unwrap();
            assert_eq!(
                t,
                confirmed + p,
                "iteration {i}: fire target must sit exactly one period after \
                 the just-confirmed vblank (the legacy `vblank + period` pacing)"
            );
            confirmed = t; // the next wait returns the boundary we armed
        }

        // Regression: the same math WITHOUT the advance folds the one-slot
        // progression to a zero step and freezes the target on the confirmed
        // vblank -- this is the exact defect the advance fixes.
        let frozen_target = 108_333u64; // armed target == first confirmed boundary
        let frozen_confirmed = 108_333u64; // wait returns the armed boundary
        assert_eq!(
            phase_preserving_step(frozen_target, frozen_confirmed + p, p),
            0,
            "prev == confirmed folds desired == confirmed + period to a zero \
             step; this is the freeze the missing advance caused"
        );
    }

    /// THE flicker/crosstalk regression: on real Wine/Proton setups the DLL's
    /// submit-time stamp `t_us` is MULTIPLE display periods stale by the time
    /// the swap shows up in the shared ring (the presentation pipeline buffers
    /// frames ahead), so any pin computed from `t_us` -- `boundary_after` with
    /// any fixed phase -- lands on an already-scanned boundary and `next_eye`
    /// drops EVERY game eye as overdue while the host freeruns invented eyes
    /// (`overdue==swaps, freerun==packets`: "locked" telemetry that is only the
    /// host toggling L/R itself).  The fix: FIFO boundary pinning in `ingest`
    /// -- the swap that arrives is the next eye the display will show, so it is
    /// pinned to the next unassigned boundary on the host's own grid, immune to
    /// how stale `t_us` is.
    #[test]
    fn ingest_fifo_pins_even_stale_submits_to_next_boundary() {
        let p = 8_333u64;
        let confirmed = 1_000_000u64; // wait_vblank_blocking just returned B0
        let armed = confirmed + p; // frame_start arms B1 (one slot ahead)
        let mut q = EyeQueue::default();
        // A submit stamped 13 whole periods in the past (measured Wine/Proton
        // queue depth): under the old t_us-based scheme this pinned to
        // confirmed - 13p and was dropped overdue.  FIFO pinning must ignore
        // t_us entirely and pin it to the next boundary (B1).
        let stale = crate::shm::Swap { eye: EYE_RIGHT, t_us: Some(confirmed - 13 * p) };
        q.ingest(&[stale], None, Some(armed));
        assert_eq!(q.drops.overdue, 0, "no overdue drop at ingest");
        assert_eq!(
            q.next_eye(armed, p),
            Some(true),
            "the stale-submit eye fires DUE on the armed boundary"
        );
        assert_eq!(q.drops.overdue, 0, "real eye fired, nothing dropped");

        // Bursty pair (one game frame of 30fps content submits both eyes
        // together): the FIFO assignment in `ingest` (anchored path) gives the
        // pair CONSECUTIVE boundaries -- first the armed B1, then B2 -- which
        // is exactly what the pair would enter as.  Each fires due, alternating
        // L/R.
        let mut q2 = EyeQueue::default();
        q2.enqueue(false, Some(armed), p); // L -> FIFO: the armed boundary
        q2.enqueue(true, Some(armed + p), p); // R -> next boundary (FIFO advance)
        assert_eq!(q2.next_eye(armed, p), Some(false), "L on B1");
        assert_eq!(q2.next_eye(armed + p, p), Some(true), "R on B2");
        assert_eq!(q2.drops.overdue, 0, "no overdue drops");
    }

    /// THE genuine-60fps-still-thrashing regression: `ingest`'s caller-supplied
    /// `next_boundary` is a fresh anchor guess each call (`last_vblank +
    /// 2*period` at top-of-loop, `vblank + period` at the late pass) that does
    /// NOT know where the schedule's own tail already sits.  SIMPLE-mode
    /// content submits both eyes of a frame close together, so a burst of 2 in
    /// one drain routinely leaves the schedule's tail one slot ahead of the
    /// next call's freshly-guessed anchor.  Before the fix, that next call
    /// started probing from BEHIND the tail and had to collision-walk forward
    /// through every already-queued entry (`enqueue`'s same-boundary loop) --
    /// which, repeated every drain on ordinary bursty-but-full-rate content,
    /// is exactly the extra churn that pushes a healthy stream toward the
    /// backlog cut.  `floored_next_boundary` (used by `ingest`) must prevent
    /// the re-probe entirely: it never starts a batch behind the tail.
    #[test]
    fn floored_next_boundary_never_starts_behind_the_tail() {
        let p = 8_333u64;

        // Empty schedule: no tail to floor against, the guess is used as-is.
        assert_eq!(floored_next_boundary(None, 500_000, p), 500_000);

        // Guess already at/after the tail's next free slot: used as-is (the
        // common case -- nothing to correct).
        assert_eq!(floored_next_boundary(Some(500_000), 500_000 + p, p), 500_000 + p);
        assert_eq!(floored_next_boundary(Some(500_000), 500_000 + 5 * p, p), 500_000 + 5 * p);

        // THE bug case: a fresh anchor guess that lags behind the tail a
        // prior bursty batch already claimed (exactly what a `last_vblank +
        // 2*period` re-derivation produces after an earlier 2-eye burst).
        // Must floor to tail+period, never the stale guess.
        assert_eq!(floored_next_boundary(Some(500_000), 500_000, p), 500_000 + p);
        assert_eq!(floored_next_boundary(Some(500_000), 500_000 - 3 * p, p), 500_000 + p);
    }

    /// Integration-level check: with the flooring in place, a batch whose
    /// caller-supplied guess lags the schedule's tail is pinned starting
    /// right after the tail -- no collision, no coalescing, no wasted
    /// collision-walk -- exactly reproducing the top-of-loop/late-drain
    /// pattern that triggered the bug on real bursty 60fps content.
    #[test]
    fn ingest_floors_a_lagging_guess_onto_the_schedule_tail() {
        let p = 8_333u64;
        let mut q = EyeQueue::default();
        // First burst (a game frame's L,R pair) lands at [B0, B1].
        let mut b = floored_next_boundary(q.scheduled.back().map(|&(t, _)| t), 1_000_000, p);
        q.enqueue(false, Some(b), p); // L -> B0
        b = b.saturating_add(p);
        q.enqueue(true, Some(b), p); // R -> B1
        assert_eq!(
            q.scheduled.iter().map(|&(bd, _)| bd).collect::<Vec<_>>(),
            vec![1_000_000, 1_000_000 + p],
        );

        // Next drain's guess is the STALE top-of-loop style re-derivation --
        // it lags one full period behind the tail this burst already set.
        let stale_guess = 1_000_000; // == B0, already claimed
        let mut b2 = floored_next_boundary(q.scheduled.back().map(|&(t, _)| t), stale_guess, p);
        assert_eq!(b2, 1_000_000 + 2 * p, "must floor to tail+period, not the stale guess");
        q.enqueue(false, Some(b2), p); // L -> B2, no collision
        b2 = b2.saturating_add(p);
        q.enqueue(true, Some(b2), p); // R -> B3, no collision

        assert_eq!(
            q.scheduled.iter().map(|&(bd, _)| bd).collect::<Vec<_>>(),
            vec![1_000_000, 1_000_000 + p, 1_000_000 + 2 * p, 1_000_000 + 3 * p],
            "four distinct, correctly-spaced boundaries -- no collision-walk needed"
        );
        assert_eq!(q.drops.coalesced, 0);
        assert_eq!(q.drops.overdue, 0);
    }
}
