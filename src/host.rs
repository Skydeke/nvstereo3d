//! `nvstusb-host` — Linux host helper for the NVIDIA 3D Vision USB IR emitter.
//!
//! wiz3D's `Nvidia3DOutput.dll` (running under Wine/Proton) pushes one
//! eye-swap command per presented frame into a shared-memory ring
//! (`/tmp/nvstusb.shm`, exposed to Wine as `Z:\tmp\nvstusb.shm`).  This helper
//! owns the USB emitter and fires the shutter packet in step with the display.
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
use std::time::{Duration, Instant};

use crate::nvstusb::drm;
use crate::nvstusb::usb;
use crate::nvtimings;
use crate::shm;
use crate::shm::{EYE_RIGHT, FLAG_EMITTER_PRESENT, FLAG_FIRMWARE_LOADED,
                Shm, STATUS_ERROR, STATUS_OPENING, STATUS_READY};

/// Embedded firmware image (must match the one shipped with `nvstereo-calibrate`).
const FIRMWARE: &[u8] = include_bytes!("../firmware/nvstusb.fw");

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

/// DIAG: wall-clock seconds, so log lines can be correlated against when the
/// viewer noticed a swapped-eye frame during play. Not used for any timing
/// logic -- diagnostics only.
fn diag_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
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

/// Hard lower bound on the inter-packet cadence the host will emit, even across
/// a pathological shorten after a drift/re-anchor timing collapse.
///
/// The trace's end-of-session lock loss showed `period 6779us < MASTER_LOCK_MIN_US`
/// measurably -- a single compressed inter-fire gap (two `send_eye` calls closer
/// than one display period, which the `frame_end` phase-preserving re-anchor's
/// +/-half-period window permits when the overdue-drop rebaseline perturbs the
/// fire schedule) that put the stream outside the emitter's master-lock window
/// and made the glasses "will NOT lock".  Because the emitter only locks when
/// consecutive packets sit inside its [7600, 9000]us window, ONE sub-window gap
/// is all it takes to drop lock for the session.
///
/// This constant is the fire-site backstop: before emitting an eye, if the
/// computed deadline would fire within `LOCK_MIN_FIRE_GAP_US` of the previous
/// send, the packet is held a few hundred us so the cadence never dips below
/// the window.  It fires at most once per pathological shorten (never in steady
/// 120 Hz), so the single packet is merely a touch early -- a far smaller
/// artifact than a full lock drop.  `frame_end(Some(vblank_us))` still confirms
/// against the TRUE kernel vblank afterwards, so the schedule stays pinned to
/// the real display grid regardless of this hold.
const LOCK_MIN_FIRE_GAP_US: u64 = 7900;

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
/// [`fire_lead_us`]).  Matches the nvstereo-calibrate demo's default `swap_phase_us` (3100)
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
/// submit -> the next one).  NOTE: this mapping is now used only for stamp
/// diagnostics (`stamp_diag::log_swap`); pinning is always FIFO.
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
/// The input `t` must already be on the host-epoch clock, i.e. the producer
/// stamp CORRECTED by the DLL-clock offset (see `stamp_diag::offset_est`).
/// Applied to the raw stamp it landed the whole schedule a constant number of
/// periods behind the armed grid (the 28-32-period Wine epoch offset measured
/// on hardware) -- every eye dropped OVERDUE, the host freeran every slot,
/// and the long-freerun hold left the glasses dark.  FIFO pinning is immune
/// to that offset (the only pinning path).  The stamped-pinning path that
/// re-used this function (`stamped_pin` adds one period) is RETIRED: on
/// hardware it moved the shuttering off the armed-service grid and broke clean
/// shuttering; see the `EyeQueue` module docs.
pub(crate) fn boundary_after(ref_boundary: u64, period: u64, t: u64) -> u64 {
    let period = period.max(1);
    // `div_euclid` rounds toward -inf, so an eye submitted before the origin
    // lands in the correct past lattice slot (not clamped to origin + period).
    let n = (t as i64 - ref_boundary as i64).div_euclid(period as i64) + 1;
    (ref_boundary as i64 + n.saturating_mul(period as i64)).max(0) as u64
}

/// The boundary the ARMED grid serves a stamped swap on: one full display
/// period AFTER the frontier slot [`boundary_after`] picks.  The host fires
/// every eye one slot AHEAD of the just-confirmed vblank
/// (`armed = confirmed + period`; see the drain loop), and the content that
/// blocks to `boundary_after(t)` is what armed is serving one slot later -- so
/// the eye is due exactly when the armed grid reaches `boundary_after(t) +
/// period`.
///
/// Pinning the eye to `boundary_after(t)` itself (the +0 form) parked it a
/// whole period BEHIND the armed boundary, where `next_eye`'s due-window edge
/// (`b + half < boundary`) drops it OVERDUE and the host freeruns an invented
/// eye in its place -- measured on hardware as "NVSTUSB_STAMP_PIN=1 no longer
/// shuts the glasses cleanly".  This is the same lesson the top-drain FIFO pin
/// already learned: it MUST be `last_vblank + 2*period` (the armed boundary),
/// not `last_vblank + period`, for the identical reason.  With the +period the
/// stamped schedule aligns slot-for-slot with the FIFO schedule that demonstrably
/// locks and shutters cleanly, while the absolute slot still comes from the
/// stamp + measured offset (deterministic start phase).
pub(crate) fn stamped_pin(ref_boundary: u64, period: u64, converted: u64) -> u64 {
    boundary_after(ref_boundary, period, converted).saturating_add(period.max(1))
}

/// Backlog threshold for the eye schedule: when queued presents exceed this we
/// have fallen behind (the host missed several real flips) and must drop the
/// stale prefix instead of replaying eyes the screen already showed.  A few
/// frames deep so a genuine stall re-anchors while normal interleaved drainage
/// never trips it.
const MAX_QUEUE: usize = 8;

/// Stuck-ahead purge threshold: `ingest` clears the whole schedule when its
/// OLDEST pinned eye sits more than HALF a display period PAST the frontier
/// being armed.  Half a period is exactly `next_eye`'s due window, so a front
/// beyond it cannot fire on that slot; and once the `pending_reanchor`
/// re-anchor loop is live, the same front is re-pinned FORWARD again on every
/// drain (`rebaseline` onto the far-ahead fresh eye), so it never ages into
/// the window either: no eye is ever due, every slot freeruns an invented eye
/// with arbitrary parity vs. the display, and the automatic inversion detector
/// starves for real DLL eyes.  A sustaining storm therefore REQUIRES the front
/// to sit past `armed + period/2` at drain time, and probing the frontier
/// `guess` (== the armed boundary) there catches it deterministically:
///   * trace.log's earlier storm parked the front 2-4 periods past every armed
///     boundary (~978 re-anchors in one session);
///   * phase 3 of the newest trace parks it at ~0.86*period past `guess` at
///     every non-empty drain -- inside the 2*period tolerance the previous
///     guard allowed (whence that storm survived), but still past the due
///     window (whence it can never fire).
/// `floored_next_boundary` is what lets the front get there: it extends the
/// schedule at `tail + period`, so a queue that ever STOPS draining marches its
/// front forward at exactly the rate the armed boundary advances.  Healthy flow
/// keeps the front INSIDE the due window (trace shows <= ~0.05*period of
/// jitter), so this edge never trips there.

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

/// Maximum the FIFO schedule's front may lag the armed display grid before
/// `next_eye` re-baselines the queue onto the grid (a fraction of the period).
///
/// The pin chain that places drained swaps on the grid advances with the
/// anchor's measured `period_us`, which carries a tiny systematic error (a
/// fraction of a us/frame) against the display's true cadence.  The target the
/// host actually arms is corrected to the REAL vblanks every frame
/// (`frame_end`'s phase-preserving re-anchor), but the already-pinned eyes in
/// the queue are never re-corrected -- so the front falls progressively
/// further behind the armed grid.  Trace.log shows this accumulating to ~half
/// a period over ~2 minutes (`gap_from_armed` growing 0 -> ~4140us right
/// before the overdue re-anchor).
///
/// That matters for THREE separate failure modes, all observed at the
/// end-of-session overdue re-anchor:
///   * SHUTTER-vs-IMAGE mismatch: the host fires each eye at its PINNED
///     boundary, which lags its true display slot by the accumulated error.
///     Past ~a quarter period this opens the shutter recognizably off-slot,
///     and at a half period it is a FULL slot off -- the viewer sees the
///     opposite eye's image at that slot = inverted depth, exactly the
///     "inversion at the end" the user reports.
///   * The overdue-drop path only re-anchors when the front is more than HALF
///     a period behind -- and at that point the re-baseline's sub-period
///     normalization is sitting exactly on the polarity-ambiguous midpoint, so
///     the correction is a 50/50 coin toss that can latch the whole stream one
///     slot over (a genuine, persistent inversion).
///   * The automatic inversion detector is STRUCTURALLY blind to this: it
///     compares the fired-eye SEQUENCE at arming-target parity, and the
///     sequence is unchanged by the drift -- only the targets themselves drift
///     off the true grid, which the detector has no reference for.
///
/// Re-baselining EARLY (well before half a period) cures all three: the
/// sub-period shift is small, its direction is unambiguous (the front is
/// clearly "the next slot behind the armed grid", not a coin-flip), and the
/// eye->slot pairing is preserved by construction (`rebaseline` shifts the
/// whole surviving schedule by the same sub-period amount and never steps a
/// whole slot).  The shutter then always opens within ~1/8 period of its true
/// slot -- invisible.  At ~30us/sec drift this fires about once every ~30s
/// with a <= ~period/8 shift, so it never perturbs the stream.  A stall is
/// NOT drift (the armed grid jumps whole slots while the schedule stays on
/// the display grid), so the overdue path drops stall-scanned eyes WITHOUT
/// re-baselining (see `next_eye`'s on-grid drop-only branch) -- only genuine
/// off-grid drift takes the full re-baseline path.
///
/// The units are FRACTIONS OF `period` computed at the call site (`period/8`),
/// so the bound scales with the display rate.
const DRIFT_LAG_ALIGN_PERIOD_DIV: u64 = 8;

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

/// Number of consecutive, same-parity, REAL (non-freerun) fires the detector
/// must observe before it locks the inversion reference.
///
/// The naive detector locked its reference on the FIRST real fire, which
/// happens during the chaotic start-of-session thrash (trace.log shows several
/// re-anchors and freeruns in the opening seconds).  That locked an arbitrary
/// early phase as "correct".  Worse, it is exactly this window where the
/// FIFO/re-anchor polarity is still a coin-flip, so a reference locked there
/// can be a slot off -- after which the detector is permanently blind: every
/// later same-parity fire matches the (wrong) reference and a genuine mid-
/// session flip reads as "no change, invert=0, zero 'inversion detected'"
/// (the exact signature of trace.log's end-of-session eye swap the user saw).
///
/// Requiring this many CONSISTENT same-parity real fires before locking lets
/// the startup thrash settle: until the phase is steady (no oscillation, no
/// re-anchor storm) the accumulator keeps resetting, so the reference only
/// locks once the session has proven a stable, trustworthy phase.  From that
/// point on it stays locked, so a real slot-flip later (e.g. the end-of-session
/// drift re-anchor re-committing the wrong phase) IS detected and toggles
/// FLAG_INVERT_EYES.  ~0.5s at 120Hz (two real eyes/frame on 60fps content).
const INVERSION_REF_STABLE_FIRES: u32 = 120;

/// Tracks the reference for automatic eye-inversion detection.
///
/// The glasses lock to whatever eye the host fired at lock-in and keep that
/// phase for the whole session.  `ref_phase` records that locked eye at the
/// reference boundary `ref_boundary`; at any later SAME-PARITY boundary the
/// glasses still show `ref_phase`.  If a real DLL eye fires the opposite eye
/// at such a boundary, the fired↔display pairing has shifted one slot and
/// FLAG_INVERT_EYES must be toggled so the renderer swaps which eye it
/// renders -- the glasses' shutter timing is never adjusted.
///
/// The reference is NOT taken from the very first real fire (that can land
/// during the startup re-anchor/freerun thrash, locking a slot-off phase and
/// blinding the detector -- see `INVERSION_REF_STABLE_FIRES`).  Instead a
/// candidate accumulator waits for consistent same-parity real fires before
/// locking, so the reference reflects the session's proven, stable phase.
#[derive(Clone, Copy)]
struct InversionState {
    ref_boundary: Option<u64>,
    ref_phase: Option<bool>,
    /// Candidate lock-in: the (boundary, phase) pair currently under test and
    /// how many consistent same-parity real fires have endorsed it.
    cand_boundary: Option<u64>,
    cand_phase: Option<bool>,
    cand_count: u32,
}

impl Default for InversionState {
    fn default() -> Self {
        InversionState {
            ref_boundary: None,
            ref_phase: None,
            cand_boundary: None,
            cand_phase: None,
            cand_count: 0,
        }
    }
}

impl InversionState {
    /// Observe one real (non-freerun) fire at boundary `target` with
    /// period `period`.  Returns `true` when the caller must toggle
    /// `FLAG_INVERT_EYES`.
    fn observe(&mut self, target: u64, period: u64, fired: bool) -> bool {
        // Already locked?  Straight same-parity comparison against the
        // reference (the stable, proven phase).
        if let (Some(rb), Some(rp)) = (self.ref_boundary, self.ref_phase) {
            if target % (period * 2) == rb % (period * 2) {
                return fired != rp;
            }
            return false;
        }
        // Not locked yet: accumulate a stable reference from consistent
        // SAME-PARITY real fires.  Only same-parity boundaries are
        // comparable (different parity always fires the opposite eye = normal
        // alternation, which must not disturb the candidate).
        let p2 = (period * 2).max(1);
        let parity = target % p2;
        match (self.cand_boundary, self.cand_phase) {
            // No candidate yet: this fire starts one.
            (None, _) => {
                self.cand_boundary = Some(target);
                self.cand_phase = Some(fired);
                self.cand_count = 1;
                false
            }
            // Same parity as the candidate.
            (Some(cb), Some(cp)) if cb % p2 == parity => {
                if fired == cp {
                    self.cand_count += 1;
                    if self.cand_count >= INVERSION_REF_STABLE_FIRES {
                        // Proven stable: promote candidate -> reference.
                        self.ref_boundary = self.cand_boundary;
                        self.ref_phase = self.cand_phase;
                        self.cand_boundary = None;
                        self.cand_phase = None;
                        self.cand_count = 0;
                    }
                    false
                } else {
                    // Same vertex, different eye: the phase is still unsettled
                    // (startup re-anchor/freerun thrash, or a genuine blip
                    // before we've locked).  Restart the accumulator on this
                    // fire rather than locking a slot-off candidate.
                    self.cand_boundary = Some(target);
                    self.cand_phase = Some(fired);
                    self.cand_count = 1;
                    false
                }
            }
            // Different parity (normal alternation): does not disturb the
            // candidate, does not lock anything.
            _ => false,
        }
    }

    /// Drop the reference (and any in-progress candidate) so it is re-
    /// established from later real fires.  Used after a manual button
    /// correction, a re-anchor, or a game restart (the paired phase is
    /// intentionally re-committed).
    fn clear(&mut self) {
        self.ref_boundary = None;
        self.ref_phase = None;
        self.cand_boundary = None;
        self.cand_phase = None;
        self.cand_count = 0;
    }
}

/// The eye source, in two modes.
///
/// STAMPED (version-2 ring): the producer stamps each swap with a
/// CLOCK_MONOTONIC PRESENT time in `Swap::boundary_us`, which the stamp
/// diagnostics (`NVSTUSB_STAMP_DIAG=1`) used to measure the DLL-clock offset.
/// That stamp is NOT used for pinning: mapping it onto the grid either lands
/// the schedule a constant number of periods behind the armed grid (the old
/// epoch-offset failure) or, when resolved to `boundary_after`, shifts the
/// shuttering off the armed-service grid (both broke clean shuttering).
/// FIFO pinning -- the clean shutter/phase path -- is the one-and-only pinning
/// (see below).
/// ## FIFO pinning (the one-and-only path)
///
/// Each drained swap is pinned to the next unassigned boundary on the host's
/// anchored grid, advancing one period per swap in drain order.  The swap that
/// arrives is the next eye the display will show, so FIFO pinning to the next
/// free slot is correct and stable while the ring stays fed -- this is the
/// clean shutter/phase that demonstrably locks on hardware.
///
/// When an occasional (mid-session) inversion is detected, the automatic
/// detector toggles `FLAG_INVERT_EYES` so the RENDERER swaps which eye gets
/// which image -- the glasses/shutter timing is untouched, so the correction
/// cannot introduce the uncomfortable shuttering a shutter-timing shift would.
///
/// ## Residual start-phase lot (why a locked session can still be inverted)
///
/// FIFO pins the FIRST swap of a (re)started stream onto the currently-armed
/// boundary; whether that boundary is the swap's ACTUAL display slot depends on
/// the game's present->scanout buffer latency (1 vs. 2 vblanks -- an
/// unobservable, per-session pipeline property).  When the two disagree by one
/// slot the whole session is inverted from the first real fire onward, and the
/// automatic inversion detector cannot see it: once the phase proves stable the
/// detector locks it as the reference (see `INVERSION_REF_STABLE_FIRES`), so a
/// STATICALLY inverted session reads as "stable, matched, invert=0" and the
/// detector stays silent (`inversion detected` logs 0 for the entire session).
/// The detector can only catch a MID-SESSION slot shift -- which the
/// stabilization makes it reliably see even when the session started clean and
/// flipped late (the end-of-session drift re-anchor path).  The observed trace
/// #3 also showed a second FIFO-inversion path: a ~300ms startup stall froze
/// `last_vblank_epoch`, FIFO pinned the first real swaps ~36 periods behind the
/// armed grid, and the overdue-drop/re-anchor "recovery" committed a fresh
/// phase on that wrong basis -- with the same result (static inversion,
/// detector blind, `inversion detected` 0).  For those static-start cases,
/// pressing the emitter's 3D button once toggles `FLAG_INVERT_EYES` (renderer
/// swaps which eye it renders); that correction persists in the shared region
/// across restarts until it is deliberately changed.
///
/// One startup coin flip the host DOES control and now eliminates: before the
/// first real DLL eye fires, the schedule is fed only by startup backlog /
/// degraded-path FIFO whose parity is ARBITRARY relative to the game's real
/// alternation, and `next_eye` would FREERUN invented toggle eyes across that
/// gap.  Whatever the glasses lock to at that first arbitrary pulse is a
/// per-session 50/50 ("inverted at launch").  `first_dll_eye_fired` makes the
/// host stay silent until the game's first real eye, so the FIRST IR pulse --
/// and therefore the glasses' lock phase -- is a deterministic function of the
/// game's own first reported eye.  If the resulting launch phase is still
/// wrong, that is the content/glass-convention case only the button can fix.
///
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
///   * STUCK-AHEAD purge (the re-anchor storm): FIFO's `tail + period` floor
///     keeps extending the schedule as fast as the armed boundary advances, so
///     a queue that ever STOPS draining parks its front PAST every boundary --
///     nothing is ever due, every slot freeruns an invented eye, and the
///     `pending_reanchor` re-anchor keeps re-baselining onto the far-ahead
///     fresh eye (`reanchor boundary = SET boundary + ~4*period`; two storms in
///     trace.log: ~978 re-anchors with the front 2-4 periods ahead, then in
///     phase 3 ~1043 re-anchors with the front only ~0.86*period past the
///     frontier -- inside the old 2-period tolerance but still past the due
///     window).  That host-invented freerun has ARBITRARY parity vs. the
///     display, so the observed result is an uninvertible fake-stream flip
///     rather than the DLL-reported eye (`inversion detected` stays zero
///     because no real eye ever fires).  `ingest` therefore purges any schedule
///     whose front sits more than HALF a period past the frontier being armed
///     (`purge_stuck_ahead`; the due-window edge -- healthy flow is far inside
///     it), so the next drain re-anchors AT the frontier and real DLL eyes fire
///     on-time again -- leaving the automatic inversion detector its real-eye
///     observations to genuinely flip `FLAG_INVERT_EYES` when a true one-slot
///     phase shift happens.
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
    /// True when the MOST RECENTLY returned eye was a FREERUN (invented) rather
    /// than a real DLL-reported eye.  The automatic inversion detector (host.rs
    /// main loop) must only compare against real DLL eyes: a freerun toggles to
    /// an arbitrary opposite eye that carries no DLL phase information, so
    /// comparing it against the reference phase would fabricate spurious
    /// inversions during every dip.
    last_was_freerun: bool,
    /// True once a REAL DLL-reported eye has fired (scheduled pin, fallback
    /// pop, or present-driven pop) -- i.e. once the stream has a
    /// game-derived phase the glasses can lock to.  Before that, the host must
    /// stay SILENT (return `None`) rather than freerun: the startup backlog /
    /// degenerate-path freerun eyes have ARBITRARY parity vs. the game's real
    /// alternation, and whatever the glasses lock to at that first pulse is a
    /// per-session 50/50 -- the "inverted at launch" coin flip the automatic
    /// detector cannot see (it self-consistently locks the wrong phase as its
    /// reference).  Delaying emission until this flag drops makes the FIRST
    /// IR pulse (and hence the glasses' lock phase) a deterministic function of
    /// the game's first real eye.
    first_dll_eye_fired: bool,
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
            last_was_freerun: false,
            first_dll_eye_fired: false,
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
                    // DIAG: distinguish a genuine dip-length reanchor
                    // (freerun_since already crossed DIP_RESYNC_AFTER_FREERUN_SLOTS
                    // on its own) from one that only fired because
                    // `pending_reanchor` is set unconditionally by every single
                    // freerun/overdue-drop (see next_eye) -- i.e. the
                    // `>= DIP_RESYNC_AFTER_FREERUN_SLOTS` gate below is
                    // currently bypassed. If `bypass=1` shows up on what was
                    // actually just a 1-slot blip, that confirms the two-tier
                    // design has collapsed into "always reanchor".
                    let bypass = self.pending_reanchor
                        && self.freerun_since < DIP_RESYNC_AFTER_FREERUN_SLOTS;
                    eprintln!(
                        "nvstusb-host: DIAG t={} reanchor freerun_since={} bypass={} \
                         held_before={:?} -> right={} boundary={}",
                        diag_ts(),
                        self.freerun_since,
                        bypass as u8,
                        self.last_fired,
                        right,
                        t
                    );
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

    /// The stuck-ahead purge: drops the whole schedule when its oldest pinned
    /// eye points more than half a period PAST `guess` (the frontier the caller
    /// is arming).  Half a period is `next_eye`'s due-window edge (`boundary <=
    /// armed + period/2`), so such a front can never fire on the bound
    /// being armed -- and once the `pending_reanchor` re-anchor loop is active
    /// the same front is re-pinned forward on every drain, so it never ages
    /// into the window either (the trace.log freerun/re-anchor storm; see the
    /// `STUCK-AHEAD purge` docs).  Those pins are stale content whose later
    /// firing would bake a re-lotted phase into the stream.  Called from
    /// `ingest` for every drain.
    fn purge_stuck_ahead(&mut self, guess: u64, period: u64) {
        let limit = guess.saturating_add(period.saturating_div(2));
        if let Some(front) = self.scheduled.front().map(|&(b, _)| b) {
            if front > limit {
                self.scheduled.clear();
            }
        }
    }

    /// Ingests every swap from a drain, computing each eye's pinned boundary
    /// when the caller can convert its timestamp onto the anchored grid.
    /// Same-boundary collisions are resolved inside `enqueue`, against the
    /// live schedule, so a frame-sequential SIMPLE pair stays alternating even
    /// when its two eyes land in separate drains.
    ///
    /// Pin scheme: EVERY drain is pinned FIFO to the NEXT unassigned boundary
    /// on the host's anchored grid -- the clean shutter/phase path.  The
    /// `stamp_offset`/`use_stamp` stamped-pinning branch is RETIRED: the
    /// caller always supplies `None` (the `NVSTUSB_STAMP_PIN` flag is a no-op),
    /// so it never runs and a version-2 swap's submit stamp is intentionally
    /// NOT used for pinning.  Off-grid stamped pinning moved the shuttering a
    /// slot off the armed-service grid and broke clean shuttering on hardware;
    /// an inverted session is instead corrected by the automatic detector
    /// toggling `FLAG_INVERT_EYES` so the RENDERER swaps which eye gets which
    /// image, with glass/shutter timing untouched.  The epoch-offset failure
    /// that once shipped the whole schedule a constant number of periods behind
    /// the armed grid (dropping every eye OVERDUE) is what ruled out stamp
    /// pinning in the first place -- FIFO onto the host's own grid is immune to
    /// it.
    fn ingest(
        &mut self,
        swaps: &[shm::Swap],
        anchor: Option<&drm::DrmVblank>,
        period: u64,
        next_boundary: Option<u64>,
        ref_vb: Option<u64>,
        stamp_offset: Option<i64>,
    ) {
        // Stamped pinning (via `use_stamp`) is RETIRED -- see the module docs.
        // The caller supplies `stamp_offset = Some(..)` only through
        // `pin_mode()`, which is permanently `false`, so `use_stamp` is always
        // false at runtime and every swap takes the FIFO branch below.  The
        // branch is kept (rather than deleted) only so the retired pinned-slot
        // tests still type-check; it is never selected.
        let use_stamp = ref_vb.is_some() && stamp_offset.is_some();
        // Each drained swap is pinned to the NEXT unassigned boundary on the
        // display grid in FIFO order: the host knows the grid from the DRM
        // anchor and advances it by one period per swap, so the swap that
        // arrives becomes the next eye the display will show.  We do NOT pin
        // from the producer's per-slot absolute PRESENT time
        // (`Swap::boundary_us`): under Wine the DLL's QPC epoch does not
        // coincide with the host's CLOCK_MONOTONIC boot epoch that
        // `host_us_from_mono` subtracts, so mapping it onto the grid
        // translated the whole schedule a constant number of periods behind
        // the armed boundary -- every eye dropped OVERDUE, the host freeran
        // every slot, and the long-freerun hold let the glasses sit dark (see
        // the module docs).  FIFO onto the host's own grid is immune to that
        // epoch offset.  The existing `enqueue` re-spacing still handles
        // bursty same-boundary pairs (e.g. 30fps where 2 eyes arrive in one
        // game frame, often in separate top/late drains) by bumping the second
        // eye forward.
        let period = period.max(1);
        // Stuck-ahead storm guard: if the oldest pinned eye points more than
        // HALF a period PAST the frontier this drain is arming, it can never
        // fire on the boundary being armed (the due-window edge) -- and once
        // the `pending_reanchor` re-anchor loop is live, the same front is
        // re-pinned forward again on every drain, so it never ages into the
        // window either: nothing is ever due, every slot freeruns an invented
        // eye, and the automatic inversion detector starves for real DLL eyes.
        // Clear the queue so this batch re-anchors the front at the frontier
        // and the stream resumes with real, on-time DLL eyes.
        if let Some(guess) = next_boundary {
            self.purge_stuck_ahead(guess, period);
        }
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
            let assigned = if use_stamp {
                match (ref_vb, stamp_offset, s.t_us) {
                    (Some(rv), Some(off), Some(t_us)) => {
                        // Convert the DLL's stamp onto the host epoch, add the
                        // measured offset, and resolve the boundary the ARMED
                        // grid serves the swap on: one period AFTER the slot
                        // the present blocks to (`boundary_after` + period).
                        // The +0 form parked the eye one period behind the
                        // armed boundary, where `next_eye` drops it overdue and
                        // the host freeruns an invented eye (the
                        // dirty-shuttering failure observed with
                        // NVSTUSB_STAMP_PIN=1).
                        let naive = anchor.map_or(0, |a| a.host_us_from_mono(t_us));
                        let converted = (naive as i64 + off).max(0) as u64;
                        Some(stamped_pin(rv, period, converted))
                    }
                    // Unusable stamp: FIFO fallback from the frontier.
                    _ => {
                        let b = boundary;
                        boundary = boundary.map(|bb| bb.saturating_add(period));
                        b
                    }
                }
            } else {
                let b = boundary;
                boundary = boundary.map(|bb| bb.saturating_add(period));
                b
            };
            self.enqueue(s.eye == EYE_RIGHT, assigned, period);
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
                // WHY the front fell > half a period behind the slot being
                // armed decides the cure:
                //   * CONTENT STALL (trace.log's end-of-session inversion): the
                //     display kept vblanking while the game paused, so the armed
                //     grid jumped whole slots ahead of a schedule that is still
                //     ON the real display grid (its pins were stamped from real
                //     vblanks, and the display grid never moved during a pause).
                //     Dropping the scanned-out eyes is exactly right; RE-
                //     BASELINING / RE-ANCHORING here is what shifted the whole
                //     stream ~1 slot off (gap_from_armed -3260us, fire-gap
                //     churn, "will NOT lock") and latched the inversion.
                //   * GENUINE DRIFT: the pins really are off the grid
                //     (sub-period lag accumulated to ~half), so the remaining
                //     eyes MUST be re-baselined onto the armed grid or they fire
                //     on the wrong slot, and the re-anchor flag re-pins the held
                //     phase to the next fresh real eye.
                // Tell them apart by the next survivor's offset from the grid:
                // within +/-period/4 of a whole-slot multiple -> on-grid stall
                // (drop-only, phase stays authoritative) -- otherwise genuine
                // drift (rebaseline + re-anchor).
                let on_grid = self.scheduled.front().map_or(true, |&(b, _)| {
                    let d = (boundary as i64 - b as i64).rem_euclid(period as i64);
                    d.min(period as i64 - d) <= (period / 4) as i64
                });
                if on_grid {
                    eprintln!(
                        "nvstusb-host: DIAG t={} overdue drop-only (on-grid stall): \
                         front_next={} boundary={boundary} overdue_total={}",
                        diag_ts(),
                        self.scheduled
                            .front()
                            .map(|&(b, _)| b)
                            .map_or(-1, |b| b as i64),
                        self.drops.overdue
                    );
                } else {
                    self.rebaseline(boundary, period);
                    if !self.pending_reanchor {
                        eprintln!(
                            "nvstusb-host: DIAG t={} pending_reanchor SET (overdue) \
                             overdue_total={} boundary={boundary}",
                            diag_ts(),
                            self.drops.overdue
                        );
                    }
                    self.pending_reanchor = true;
                }
            }
            // Early drift re-sync (see `DRIFT_LAG_ALIGN_PERIOD_DIV`): the pin
            // chain's tiny systematic period error accumulates, so the front
            // slips progressively behind the armed grid (trace.log
            // `gap_from_armed` growing 0 -> ~half a period across a ~2min
            // session).  Re-baseline LONG before the half-period overdue point
            // so the shift stays small and pairing-preserving (never a whole
            // slot), the shutter always opens within ~period/8 of its true
            // slot, and the polarity-ambiguous overdue overhaul above never has
            // to fire on a healthy stream.  The automatic inversion detector
            // is structurally blind to this drift (it compares the fired-eye
            // SEQUENCE at arming-target parity, and the sequence is unchanged
            // while the targets themselves drift off the grid) -- so this is
            // the one reliable defense, and keeping the lag bounded is what
            // prevents the end-of-session wrong-depth the user sees.
            if !self.pending_reanchor {
                let align_limit = period / DRIFT_LAG_ALIGN_PERIOD_DIV;
                if let Some(front) = self.scheduled.front().map(|&(b, _)| b) {
                    let lag = boundary as i64 - front as i64;
                    if lag > align_limit as i64 {
                        self.rebaseline(boundary, period);
                        eprintln!(
                            "nvstusb-host: DIAG t={} drift re-sync: front {front} lagging \
                             armed {boundary} by > period/{} -> rebaselined onto grid",
                            diag_ts(),
                            DRIFT_LAG_ALIGN_PERIOD_DIV
                        );
                    } else if lag < -(align_limit as i64) {
                        // Mirror of the lagging drift re-sync: after a recovery
                        // (or a boundary that snapped back) the front can sit
                        // pinned MORE than period/8 AHEAD of the armed grid --
                        // firing it then opens the shutter EARLY onto the
                        // previous slot (the trace.log -3260us post-stall
                        // transient).  Re-baseline it back onto the armed grid:
                        // a small, pairing-preserving sub-period shift (`rebaseline`'s
                        // normalization handles either direction).
                        self.rebaseline(boundary, period);
                        eprintln!(
                            "nvstusb-host: DIAG t={} drift re-sync: front {front} LEADING \
                             armed {boundary} by > period/{} -> rebaselined onto grid",
                            diag_ts(),
                            DRIFT_LAG_ALIGN_PERIOD_DIV
                        );
                    }
                }
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
                self.last_was_freerun = false;
                if !self.first_dll_eye_fired {
                    self.first_dll_eye_fired = true;
                    if crate::stamp_diag::enabled() {
                        eprintln!(
                            "nvstusb-host: DIAG t={} first real eye {:?} fired on its stamped \
                             boundary {boundary} -- the glasses' lock phase is now a function of \
                             the game's eye, not startup freerun parity",
                            diag_ts(),
                            if e { "R" } else { "L" }
                        );
                    }
                }
                return Some(e);
            }
            // None due: freerun.  Toggle to the opposite eye so the fired
            // stream keeps shuttering/alternating (lock holds); the moment the
            // game's pinned eyes return they pop at their own boundaries and
            // the stream resyncs.  Past STALL_HOLD_AFTER_FREERUN_SLOTS this is
            // no longer a blip but a real stall -- stop toggling and hold the
            // last eye so a stuck/stale frame isn't strobed (see the constant's
            // doc comment).
            // Pre-first-eye: there is NO established phase to hold yet, and a
            // freerun-invented eye here would be the arbitrary-parity pulse
            // the glasses lock to (the "inverted at launch" 50/50).  Stay
            // SILENT instead -- the game's first real eye (whenever it arrives)
            // becomes the first IR pulse and the lock phase is deterministic.
            if !self.first_dll_eye_fired {
                return None;
            }
            self.freerun_since = self.freerun_since.saturating_add(1);
            self.drops.freerun += 1;
            // The freerun INVENTED this eye (the game never reported it for
            // this slot), so the held phase has left the game's true schedule.
            // Glue the next fresh DLL eye to a grid re-baseline so the resumed
            // stream cannot inherit this invented parity (which is the 50/50
            // "eyes swapped after a dip" source).
            //
            // FIX (was unconditional): only latch this on a genuine dip
            // (freerun_since >= DIP_RESYNC_AFTER_FREERUN_SLOTS), matching the
            // documented two-tier design. Setting it on every single-slot
            // blip (freerun_since==1) made `enqueue`'s own `>= 2` gate dead
            // code and forced a full reanchor on literally every incoming
            // swap during steady content -- trace.log showed 1431/1437
            // reanchors (99.6%) were this bypass, meaning the schedule was
            // never once settling into a clean on-time `due` fire, session-
            // long, before a real stall hit.
            if self.freerun_since >= DIP_RESYNC_AFTER_FREERUN_SLOTS {
                if !self.pending_reanchor {
                    eprintln!(
                        "nvstusb-host: DIAG t={} pending_reanchor SET (freerun, scheduled) \
                         freerun_since={} boundary={boundary}",
                        diag_ts(),
                        self.freerun_since
                    );
                }
                self.pending_reanchor = true;
            }
            if self.freerun_since > self.stall_hold_after_freerun {
                self.last_was_freerun = self.freerun_since > 0;
                return self.last_fired;
            }
            self.last_was_freerun = true;
            return self.last_fired.map(|r| freerun(&mut self.last_fired, r));
        }
        if let Some(e) = self.fallback.pop_front() {
            self.last_fired = Some(e);
            self.freerun_since = 0; // real DLL eye -> phase is authoritative
            self.last_was_freerun = false;
            if !self.first_dll_eye_fired {
                self.first_dll_eye_fired = true;
                if crate::stamp_diag::enabled() {
                    eprintln!(
                        "nvstusb-host: DIAG t={} first real eye {:?} fired from the FIFO \
                         backlog -- the glasses' lock phase is now a function of the game's eye, \
                         not startup freerun parity",
                        diag_ts(),
                        if e { "R" } else { "L" }
                    );
                }
            }
            Some(e)
        } else if let Some(r) = self.last_fired {
            // Scheduled empty AND fallback empty: a dip in the degraded path.
            // Pre-first-eye: stay silent (no invented pulse for the glasses to
            // lock onto) -- see next_eye's freerun guard above.
            if !self.first_dll_eye_fired {
                return None;
            }
            self.freerun_since = self.freerun_since.saturating_add(1);
            self.drops.freerun += 1;
            // FIX: same gate as above -- see the comment there.
            if self.freerun_since >= DIP_RESYNC_AFTER_FREERUN_SLOTS {
                if !self.pending_reanchor {
                    eprintln!(
                        "nvstusb-host: DIAG t={} pending_reanchor SET (freerun, fallback) \
                         freerun_since={}",
                        diag_ts(),
                        self.freerun_since
                    );
                }
                self.pending_reanchor = true;
            }
            if self.freerun_since > self.stall_hold_after_freerun {
                self.last_was_freerun = self.freerun_since > 0;
                return self.last_fired;
            }
            self.last_was_freerun = true;
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
            self.last_was_freerun = false;
            self.first_dll_eye_fired = true;
            Some(r)
        } else if let Some(e) = self.fallback.pop_front() {
            self.last_fired = Some(e);
            self.freerun_since = 0;
            self.last_was_freerun = false;
            self.first_dll_eye_fired = true;
            Some(e)
        } else {
            // Pre-first-eye: never repeat an invented/held phase before the
            // first real DLL eye (see the `first_dll_eye_fired` doc comment).
            let held = if self.first_dll_eye_fired {
                self.last_fired
            } else {
                None
            };
            if held.is_some() {
                self.last_was_freerun = true;
            }
            held
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

    /// TEMP DEBUG: tail scheduled boundary (pinned boundary of the newest eye).
    fn tail_boundary(&self) -> Option<u64> {
        self.scheduled.back().map(|&(b, _)| b)
    }

    /// Full reset (game stop / re-anchor): queue AND held eye cleared so the
    /// next stream starts from a clean phase reference.
    fn reset(&mut self) {
        self.fallback.clear();
        self.scheduled.clear();
        self.last_fired = None;
        self.freerun_since = 0;
        self.pending_reanchor = false;
        self.last_was_freerun = false;
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
/// header can carry either a concrete connector name (e.g. `DP-2`, as the nvstereo-calibrate
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
/// This is the host-side counterpart of the nvstereo-calibrate demo's `s`-key save: a
/// profile tuned and saved by the demo is loaded here too, so
/// `nvstereo3d` shuts the glasses with the same X/Y/W registers the demo
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
    eprintln!(
        "nvstusb-host: if depth looks INVERTED at launch (a wrong start-phase, which the \
         automatic detector cannot see), press the emitter's 3D button once to swap eyes. \
         The button fix is persistent (shared region), so after one press it stays \
         corrected for every later run.  Mid-session inversions are fixed automatically by \
         the detector asking the renderer to swap which eye gets which image (glass timing \
         untouched).  NVSTUSB_STAMP_DIAG=1 prints per-swap stamp diagnostics."
    );

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
    // exactly the mechanism nvstereo-calibrate relies on to shutter under a compositor.
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

    // --- Automatic inversion detection (phase-pinned mode) ------------------
    // The display alternates which eye it shows one frame per vblank boundary;
    // the shutter glasses lock to whatever alternating phase the host fired at
    // lock-in and KEEP that phase for the session.  The host fires exactly what
    // the DLL reports, so the fired↔display pairing is correct as long as the
    // two stay phase-locked.  But a discrete phase event -- a re-anchor after a
    // dip/freerun, a submit→scanout offset change, or a game that starts with a
    // different eye order -- can shift the pairing by one full slot: the same
    // eye now fires on the opposite boundary parity and the viewer sees
    // INVERTED depth for the rest of the session.
    //
    // Detection: the glasses' locked phase is constant.  The `InversionState`
    // records the reference eye that fired at its reference boundary once the
    // session PROVES a stable phase -- it requires `INVERSION_REF_STABLE_FIRES`
    // consistent same-parity real fires (skipping the startup re-anchor/freerun
    // thrash) before locking, so the reference reflects the glasses' true
    // locked phase, not an arbitrary early fire.  At any later same-parity
    // boundary, if the host fires a DIFFERENT eye, the pairing has shifted a
    // slot: set (or clear) FLAG_INVERT_EYES so the renderer swaps which eye it
    // renders.  The glasses' shutter timing is NEVER adjusted -- only the shm
    // flag is toggled, so there is no phase shift on the glasses side and no
    // uncomfortable dark/black period (the old approach re-pinned the schedule
    // by +/- one period, which held one eye for the extra slot -- visibly
    // uncomfortable).  Once the DLL swaps, the fired eye returns to the
    // reference and the detector goes quiet (self-correcting, no oscillation).
    //
    // Only real DLL eyes (not freerun toggles) are compared: a freerun invents
    // an arbitrary eye that carries no phase information.  Only stamped
    // (phase-pinned) mode runs the detector -- legacy (v1) mode has no grid and
    // keeps the manual button (manual_invert).  A manual button press clears
    // the reference so the detector can never undo the user's own correction.
    //
    // After a re-anchor (monitor switch, resync), the display's eye phase is
    // unchanged -- only the vblank clock phase shifts -- so the inversion flag
    // persists.  The reference boundary is re-established on the new monitor's
    // grid via the next real DLL eye.

    // Reference for automatic inversion detection: the boundary + fired eye
    // that defined the glasses' locked phase.  See `InversionState`.
    let mut inversion = InversionState::default();
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
    // Set when content RESUMES after a dead stretch.  The DRM prediction
    // (`next_present_us`) freezes while the host is idle (`frame_start`/
    // `frame_end` are gated on the schedule having an eye), so on resume it can
    // carry a stale phase relative to the display grid -- and `frame_start`
    // only ever moves it in WHOLE periods, so that stale phase survives every
    // snap.  Re-anchor it onto the next kernel-confirmed vblank's grid before
    // the first armed frame (see the acquisition-storm docs) so acquisition
    // cannot start half a period off the display grid.
    let mut force_grid_resync = false;

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

    // Instant of the most recent `send_eye`, used by the LOCK_MIN_FIRE_GAP_US
    // backstop to guarantee the inter-packet cadence never collapses below the
    // emitter's master-lock window (which would make the glasses "will NOT
    // lock").  Reset to None after each report so a long idle span (no swaps)
    // doesn't count as a huge gap and then mis-flag the next real fire.
    let mut last_fire: Option<std::time::Instant> = None;

    // Host-epoch of the most recent vblank we waited on (for the lead calc).
    let mut last_vblank_epoch: Option<u64> = None;

    // The fire target (host-epoch vblank to arm the IR for) is owned by the
    // DRM anchor itself (`DrmVblank::next_present_us`), advanced by the SAME
    // `frame_start`/`frame_end` pair the working nvstereo-calibrate demo drives the emitter
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
        // No DRM anchor: the anchor is required for proper shuttering, so this
        // loop paces on the vblank wait below.  Without one we degrade to
        // present-driven emission and poll the ring on a short timer instead of
        // busy-spinning (the old one-byte UDP wake datagram is gone; the ring is
        // the source of truth and was always re-checked within ~5 ms anyway).
        if drm_anchor.is_none() {
            std::thread::sleep(Duration::from_millis(5));
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
            eprintln!("nvstusb-host: DIAG t={} stamped_mode -> {}", diag_ts(), mode);
            eprintln!(
                "nvstusb-host: shared ring is {}",
                if mode {
                    "phase-pinned (version-2 stamped swaps)"
                } else {
                    "legacy (version-1 unstamped swaps; FIFO+hold fallback)"
                }
            );
            // Pinning is ALWAYS FIFO onto the anchored grid: the validated
            // silent path for the clean shutter/phase.  Start-phase inversions
            // are corrected by the automatic detector toggling FLAG_INVERT_EYES
            // (renderer swaps which eye gets which image -- glass timing
            // untouched), per the operator's directive to never shift the
            // shutter/phase to "fix" depth.
            eprintln!("nvstusb-host: pinning = FIFO (clean shutter/phase; pick which eye renders via the inversion detector)");
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
            // Producer-stamp diagnostics (NVSTUSB_STAMP_DIAG=1): compute the
            // exact FIFO pin sequence ingest will use, BEFORE ingest mutates
            // the schedule, so each swap can be correlated with the DLL's
            // submit stamp and the host grid afterwards (see stamp_diag.rs).
            let stamp_fifo_pins: Vec<Option<u64>> = if crate::stamp_diag::enabled() {
                if let Some(anchor) = drm_anchor.as_ref() {
                    let p = anchor.period_us().max(1);
                    let mut b =
                        next_boundary.map(|g| floored_next_boundary(pending.tail_boundary(), g, p));
                    swaps
                        .iter()
                        .map(|_| {
                            let r = b;
                            if let Some(x) = b.as_mut() {
                                *x = x.saturating_add(p);
                            }
                            r
                        })
                        .collect()
                } else {
                    vec![None; swaps.len()]
                }
            } else {
                Vec::new()
            };
            pending.ingest(
                &swaps,
                drm_anchor.as_ref(),
                drm_anchor.as_ref().map_or(0, |a| a.period_us().max(1)),
                next_boundary,
                last_vblank_epoch,
                if crate::stamp_diag::pin_mode() {
                    crate::stamp_diag::offset_est()
                } else {
                    None
                },
            );
            // Feed the DLL-clock offset estimator (stamp_diag no-ops unless a
            // stamp mode is active); the median age becomes the correction the
            // NEXT drain's stamped pins use.
            if let Some(anchor) = drm_anchor.as_ref() {
                for s in &swaps {
                    crate::stamp_diag::record_age_of(anchor, *s);
                }
            }
            if !stamp_fifo_pins.is_empty() {
                if let Some(anchor) = drm_anchor.as_ref() {
                    let p = anchor.period_us().max(1);
                    let ref_vb = last_vblank_epoch.unwrap_or(0);
                    for (s, fb) in swaps.iter().zip(&stamp_fifo_pins) {
                        crate::stamp_diag::log_swap(anchor, ref_vb, p, *s, *fb);
                    }
                }
            }
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
                    // The DISPLAY's eye phase is unchanged by a re-anchor (the
                    // monitor keeps alternating L/R at its own cadence), so the
                    // inversion flag persists.  But the reference boundary was
                    // on the old head's grid -- re-establish it on the new
                    // monitor's grid via the next real DLL eye.
                    inversion.clear();
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
                // Fresh stamp-diagnostic detail budget for the next launch /
                // resume burst (NVSTUSB_STAMP_DIAG=1).
                crate::stamp_diag::reset();
                // A game/exe that quits often restarts (or is followed by a
                // different one) with a FRESH DLL that may report a different
                // native eye phase than the session that just ended.  The
                // display's phase is unchanged, but the reference must be
                // re-established from the next launch's first real DLL eye --
                // the automatic detector re-compares and confirm/flips
                // FLAG_INVERT_EYES from there.
                inversion.clear();
            }
            if !was_alive && alive {
                // Content resumed after an idle stretch: re-anchor the DRM
                // prediction onto the display grid before the first armed
                // frame (see `force_grid_resync`'s declaration inside the main
                // loop).  The dead window froze the prediction with an
                // arbitrary phase vs. the marching display grid; without the
                // re-anchor, acquisition starts with the armed target half a
                // period off the pins and the overdue/re-anchor storm
                // throttles emission during the very window the glasses are
                // trying to lock (the pre-lock "will NOT lock" storm the
                // acquisition docs describe).
                force_grid_resync = true;
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
                        // A manual button press is the authoritative polarity
                        // correction: whatever the glasses now fire becomes the
                        // correct locked phase.  Re-establish the inversion
                        // reference from the next real DLL eye so the automatic
                        // detector compares against the newly-corrected phase
                        // instead of fighting the user's correction.
                        inversion.clear();
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
                            if first_grid {
                                // The FIRST kernel-confirmed vblank establishes
                                // the armed grid.  The anchor's prediction can
                                // carry a phase from open()/before content, and
                                // `frame_start` only advances it in WHOLE
                                // periods -- so a stale phase survives every
                                // snap.  Pin the prediction to this confirmed
                                // vblank's own grid (`vblank + period` = the
                                // real next slot) so the very first pins and
                                // arms agree with the display (the
                                // acquisition-storm root fix).  Applies in
                                // BOTH modes: even unstamped swaps are pinned
                                // FIFO onto this grid.
                                anchor.resync_to_vblank(vblank_us);
                                if stamped_mode {
                                    // Startup swaps that landed before this
                                    // first boundary were necessarily FIFO'd --
                                    // drop them so the stamped schedule (whose
                                    // targets are grid-exact) becomes the only
                                    // source.
                                    pending.clear_fallback();
                                }
                            }
                            if force_grid_resync {
                                // Content just resumed after a dead stretch
                                // (the prediction froze during it): re-anchor
                                // onto this confirmed vblank's grid before the
                                // first armed frame, exactly like `first_grid`.
                                anchor.resync_to_vblank(vblank_us);
                                force_grid_resync = false;
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
                                pending.ingest(
                                    &late,
                                    Some(anchor),
                                    anchor.period_us().max(1),
                                    Some(next_boundary),
                                    Some(vblank_us),
                                    if crate::stamp_diag::pin_mode() {
                                        crate::stamp_diag::offset_est()
                                    } else {
                                        None
                                    },
                                );
                                for s in &late {
                                    crate::stamp_diag::record_age_of(anchor, *s);
                                }
                            }
                            // Pace the fire target with the anchor's OWN
                            // `frame_start`/`frame_end` -- the exact pair the
                            // working nvstereo-calibrate demo uses to drive the emitter
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
                            // Snaps + computes the fire deadline.  Called on
                            // EVERY vblank -- even when the schedule is empty --
                            // so the prediction keeps marching with the display
                            // grid across sparse-content stretches instead of
                            // freezing with a stale phase (the acquisition-storm
                            // root cause: a frozen `next_present_us` whose
                            // sub-period phase no longer matches the vblank grid
                            // survives every `frame_start` snap, so the
                            // confirmed-grid pins fall just past the armed due
                            // window and the overdue/re-anchor loop storms right
                            // when the glasses try to lock).  Returns the proven
                            // fire Instant (target - alarm - lead); the actual
                            // send below is skipped when no eye is due.
                            let fire_at = anchor.frame_start(host_lead_us as u32);
                            let target = anchor.current_present_us();
                            if pending.has_eye() {
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
                                    // Harden the fire cadence against a drift /
                                    // re-anchor timing collapse.  The
                                    // `frame_start` deadline is normally ~1 slot
                                    // ahead, but the overdue-rebaseline path can
                                    // compress the next deadline inside the
                                    // emitter's master-lock window (trace:
                                    // period 6779us -> "will NOT lock").  Hold
                                    // the packet so two sends can never be
                                    // closer than LOCK_MIN_FIRE_GAP_US, keeping
                                    // the cadence inside [7600, 9000]us.  This
                                    // only ever holds on a pathological shorten
                                    // (never steady 120 Hz), costing at most a
                                    // few hundred us of lead on that one packet.
                                    if let (Some(prev), Some(fire_at)) =
                                        (last_fire, fire_at)
                                    {
                                        let floor = prev
                                            + std::time::Duration::from_micros(
                                                LOCK_MIN_FIRE_GAP_US,
                                            );
                                        if fire_at < floor {
                                            if crate::stamp_diag::enabled() {
                                                eprintln!(
                                                    "nvstusb-host: DIAG t={} fire-gap backstop: deadline {fire_at:?} within LOCK_MIN_FIRE_GAP_US of previous send; holding to {floor:?} (was a shorten)",
                                                    diag_ts()
                                                );
                                            }
                                            while Instant::now() < floor {
                                                std::hint::spin_loop();
                                            }
                                        } else {
                                            while Instant::now() < fire_at {
                                                std::hint::spin_loop();
                                            }
                                        }
                                    } else if let Some(fire_at) = fire_at {
                                        while Instant::now() < fire_at {
                                            std::hint::spin_loop();
                                        }
                                    }
                                    // manual_invert applies at the wire only
                                    // (and only matters in fallback mode).
                                    let out = eye != manual_invert;
                                    d.send_eye(out, rate_hz);
                                    // Capture AFTER the (USB) send so `last_fire`
                                    // aligns with `dbg.record`'s own Instant ->
                                    // the backstop gap matches the measured
                                    // `min_period` rather than understating it by
                                    // the send duration.
                                    last_fire = Some(std::time::Instant::now());
                                    dbg.record(out);

                                    // --- Automatic inversion detection ---
                                    // Only in phase-pinned (stamped) mode: that
                                    // is the mode where the DLL reads
                                    // FLAG_INVERT_EYES, and the only mode where
                                    // the display-phase pairing is trustworthy.
                                    // Legacy (v1) mode keeps the manual button.
                                    // Only real DLL eyes (not a freerun's
                                    // invented toggle) carry phase information
                                    // we can compare against the reference.
                                    if stamped_mode && !pending.last_was_freerun {
                                        // Observe this real fire.  A `true`
                                        // return means the fired↔display
                                        // pairing shifted a slot (inverted
                                        // depth).  Toggle FLAG_INVERT_EYES:
                                        // the DLL swaps which eye it renders
                                        // (glass timing untouched), so the
                                        // fired phase returns to the reference
                                        // and the view un-inverts.
                                        if inversion.observe(target, period, out) {
                                            let new_state = !shm.invert_eyes();
                                            if new_state {
                                                shm.set_invert_eyes();
                                            } else {
                                                shm.clear_invert_eyes();
                                            }
                                            eprintln!(
                                                "nvstusb-host: DIAG t={} inversion detected: fired {:?} != locked reference {:?} at boundary {} (ref {}); FLAG_INVERT_EYES {}",
                                                diag_ts(),
                                                if out { "R" } else { "L" },
                                                if inversion.ref_phase == Some(true) { "R" } else { "L" },
                                                target,
                                                inversion.ref_boundary.unwrap_or(0),
                                                if new_state { "SET" } else { "CLEARED" }
                                            );
                                        }
                                    }
                                }
                            }
                            // Advance the anchor's prediction EVERY vblank,
                            // exactly like the demo's post-swap `frame_end`
                            // (one period + sub-slot phase-lock) -- including
                            // empty-schedule frames.  Gating it (and
                            // `frame_start`) on the schedule having an eye
                            // froze `next_present_us` during idle/sparse
                            // stretches; that frozen prediction carries a
                            // stale phase into content start that survives
                            // every whole-period snap and ignites the
                            // acquisition overdue/re-anchor storm (this, and
                            // the `first_grid`/resume `resync_to_vblank`
                            // calls above, are the storm's root fix).
                            // Confirm against the kernel vblank this wait just
                            // returned: the armed boundary is exactly one
                            // period ahead, so the phase-preserving
                            // normalization folds that -period to ~0 (no
                            // spurious nudge) -- keeping the target pinned to
                            // the display grid.  Confirming against
                            // wall-clock `now` (mid-frame, `target - alarm -
                            // lead` before the boundary) would bake a
                            // -3250us error into every frame and drag the
                            // target off-grid.
                            anchor.frame_end(Some(vblank_us));
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
                // as the ring is polled on the timer above); fix the anchor
                // permissions instead of tuning here.  The eye still comes from
                // the game's ring.
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
    /// and latches INVERTED ~50/50 (the `freerun=7` eye-swap symptom).  A
    /// genuine dip (`freerun_since` >= the resync threshold) must therefore
    /// flag `pending_reanchor`, and the FIRST fresh, real DLL eye must consume
    /// it -- re-pinning the held phase to the game's true eye (so the resumed
    /// stream alternates off it deterministically).  Single-slot blips stay
    /// below the threshold and must NOT flag (see `short_blip_is_not_a_dip`).
    #[test]
    fn freerun_marks_pending_reanchor_consumed_by_next_real_eye() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        // Real L holds the phase.
        q.enqueue(false, Some(b), p);
        assert_eq!(q.next_eye(b, p), Some(false));
        assert!(!q.pending_reanchor);

        // A genuine dip: the game skips enough slots that the host invents
        // several freerun eyes, crossing the resync threshold.  That breaks
        // its trust in the held parity and MUST flag a re-anchor.
        assert_eq!(q.next_eye(b + p, p), Some(true)); // freerun -> R
        assert_eq!(q.freerun_since, 1);
        assert_eq!(q.next_eye(b + 2 * p, p), Some(false)); // freerun -> L
        assert_eq!(q.freerun_since, 2);
        assert_eq!(
            q.pending_reanchor, true,
            "a genuine dip (freerun_since >= threshold) must flag the phase as needing re-anchor"
        );

        // The game resumes with a fresh real eye.  It consumes the flag and
        // re-pins the held phase to the game's own eye -- NOT the freerun's
        // invented parity -- so post-dip parity is deterministic.
        assert!(q.pending_reanchor, "flag still pending before the fresh eye");
        q.enqueue(true, Some(b + 3 * p), p); // fresh real R
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

    /// The launch-phase coin flip: before the game's FIRST real DLL eye fires,
    /// the schedule is fed only by startup backlog whose parity is arbitrary
    /// relative to the game's alternation, and the freerun path would invent
    /// toggle eyes the glasses lock to (a per-session 50/50 "inverted at
    /// launch" -- the automatic detector is blind to it because it locks the
    /// wrong phase as its own reference).  The host must stay SILENT on those
    /// slots so the FIRST IR pulse is the game's first real eye -- making the
    /// glasses' lock phase a deterministic function of the game, not startup
    /// timing.
    #[test]
    fn startup_stays_silent_until_first_real_eye() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        // A held phase exists (set by the backlog re-anchor) but no real DLL
        // eye has ever fired -- the exact state that produced trace.log's
        // `freerun=36` at launch.
        q.last_fired = Some(true);
        assert!(!q.first_dll_eye_fired);
        let _ = q.fallback.len(); // keep fallback empty too
        // Scheduled and fallback both empty + no real eye yet: SILENT, never
        // an invented toggle pulse, and no stall/partial stats accumulate.
        assert_eq!(q.next_eye(b, p), None, "pre-first-eye dip must not freerun");
        assert_eq!(q.next_eye(b + p, p), None, "pre-first-eye dip must not freerun");
        assert_eq!(q.drops.freerun, 0, "no invented eyes before the first real one");
        assert_eq!(q.freerun_since, 0);
        assert!(!q.pending_reanchor, "silence is not a dip, so no re-anchor flag");
        // The game's first real eye fires on its own boundary and enables
        // normal dip-freerun afterwards.
        q.enqueue(true, Some(b + 2 * p), p);
        assert_eq!(q.next_eye(b + 2 * p, p), Some(true));
        assert!(q.first_dll_eye_fired, "first real eye flips the launch latch");
        assert_eq!(q.next_eye(b + 3 * p, p), Some(false), "dip freerun resumes after a real fire");
        assert_eq!(q.drops.freerun, 1);
    }

    /// Same guarantee on the degraded (no-anchor) present-driven path: never
    /// repeat a held/invented phase before the first real DLL eye.
    #[test]
    fn startup_present_driven_stays_silent_until_first_real_eye() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        q.last_fired = Some(false); // arbitrary held parity, nothing real fired
        assert!(!q.first_dll_eye_fired);
        assert_eq!(q.next_eye_present_driven(), None);
        assert_eq!(q.next_eye_present_driven(), None);
        // A real FIFO backlog eye pops (real DLL eye) and the held repeat
        // resumes only after that.
        q.enqueue(false, None, p);
        assert_eq!(q.next_eye_present_driven(), Some(false));
        assert!(q.first_dll_eye_fired);
        assert_eq!(q.next_eye_present_driven(), Some(false), "held eye repeats after launch");
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
        q.ingest(&[crate::shm::Swap { eye: EYE_LEFT, t_us: Some(105_000), seq: 1 }], None, p, None, None, None);
        assert_eq!(q.next_eye(b, p), Some(false)); // left
        // Legacy drain (t_us None) -> FIFO.
        q.ingest(&[crate::shm::Swap { eye: EYE_RIGHT, t_us: None, seq: 2 }], None, p, Some(b), None, None);
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
        let stale = crate::shm::Swap { eye: EYE_RIGHT, t_us: Some(confirmed - 13 * p), seq: 3 };
        q.ingest(&[stale], None, p, Some(armed), None, None);
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

    /// Stamped pinning (NVSTUSB_STAMP_PIN=1): with a grid origin and a
    /// measured DLL-clock offset, a version-2 swap pins to the boundary the
    /// ARMED grid serves its scanout on -- `boundary_after` (the slot the
    /// present blocks to) PLUS one period, never the caller's FIFO guess.
    /// (In this anchor-free test the `stamp_offset` IS the host-epoch-converted
    /// present time; in production it is `naive + stamp_diag::offset_est()`.)
    #[test]
    fn ingest_stamped_pin_wins_over_the_fifo_guess() {
        let p = 8_333u64;
        let ref_vb = 1_000_000u64;
        let mut q = EyeQueue::default();
        // Converted present = 1_004_000 (mid-period of B0): blocks to B1
        // (1_008_333), so the armed grid serves the swap on B2 (1_016_666 =
        // boundary_after + period) -- the same slot FIFO pinning lands on in
        // the healthy stream.
        let swap = crate::shm::Swap { eye: EYE_LEFT, t_us: Some(123), seq: 1 };
        // The caller's FIFO guess points at B1 -- a stale/behind frontier --
        // and the stamped pin must override it (it is one slot ahead, B2).
        q.ingest(&[swap], None, p, Some(1_008_333), Some(ref_vb), Some(1_004_000));
        assert_eq!(
            q.front_boundary(),
            Some(1_016_666),
            "the eye sits on the armed-served boundary (boundary_after + period)"
        );
        assert_eq!(q.drops.overdue, 0, "not dropped at ingest");
        assert_eq!(
            q.next_eye(1_016_666, p),
            Some(false),
            "fires DUE exactly when the armed grid reaches its boundary"
        );
        assert_eq!(q.drops.overdue, 0, "nothing dropped");
    }

    /// Stamped pinning pins a TRULY past slot where it is (never replays it on
    /// the tail); the eye has already scanned out, so `next_eye`'s overdue
    /// logic drops it instead -- the documented self-healing contract of the
    /// signed `boundary_after` mapping.
    #[test]
    fn ingest_stamped_keeps_a_true_past_slot_and_lets_it_drop_overdue() {
        let p = 8_333u64;
        let ref_vb = 1_000_000u64;
        let mut q = EyeQueue::default();
        // Converted present = 983_333 (two periods before the origin): the
        // present blocks to 983_334, so the armed grid would serve it at
        // 991_667 (= boundary_after + period) -- still long past, NOT bumped
        // forward onto the tail.
        let swap = crate::shm::Swap { eye: EYE_RIGHT, t_us: Some(1), seq: 1 };
        q.ingest(&[swap], None, p, Some(1_016_666), Some(ref_vb), Some(983_333));
        assert_eq!(
            q.front_boundary(),
            Some(991_667),
            "must keep the true (already-scanned) slot"
        );
        assert_eq!(q.drops.overdue, 0, "not dropped at ingest -- it is still pending");
        // When the armed grid passes it, the stale eye drops overdue...
        q.next_eye(1_016_666, p);
        assert!(q.drops.overdue >= 1, "a past slot must drop, never replay");
    }

    /// Two eyes of one 60fps frame submit back-to-back before the shared
    /// vblank; both stamps resolve to the SAME armed-served slot.  `enqueue`'s
    /// live-schedule collision walk re-spaces the second onto the following
    /// slot, so the pair still fires as a correct L/R alternation -- same
    /// behaviour as FIFO mode, now with the absolute slot from the stamps.
    #[test]
    fn ingest_stamped_pair_respaces_to_consecutive_slots() {
        let p = 8_333u64;
        let ref_vb = 1_000_000u64;
        let mut q = EyeQueue::default();
        let pair = [
            crate::shm::Swap { eye: EYE_LEFT, t_us: Some(1), seq: 1 },
            crate::shm::Swap { eye: EYE_RIGHT, t_us: Some(2), seq: 2 },
        ];
        // Both convert to 1_004_000 -> both block to B1 -> both would pin to
        // B2 (boundary_after + period); enqueue bumps R onto B3.
        q.ingest(&pair, None, p, None, Some(ref_vb), Some(1_004_000));
        assert_eq!(
            q.scheduled.iter().map(|&(b, _)| b).collect::<Vec<_>>(),
            vec![1_016_666, 1_024_999],
            "L keeps B2, R is re-spaced onto B3"
        );
        assert_eq!(q.next_eye(1_016_666, p), Some(false), "L on B2");
        assert_eq!(q.next_eye(1_024_999, p), Some(true), "R on B3");
        assert_eq!(q.drops.overdue, 0, "nothing dropped");
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

    // --- Automatic inversion detection (InversionState) --------------------

    /// Drives `slots` consecutive display slots of a STABLE alternating stream
    /// starting at boundary `b`: even slots (same parity as `b`) fire `r` on
    /// the even parity and `!r` on the odd parity -- i.e. a consistent
    /// alternation whose even-parity eye is `r`.  Returns the last observe()
    /// result (used to check for spurious flags while stabilizing).
    fn drive_stable(inv: &mut InversionState, b: u64, p: u64, r: bool, slots: u32) -> bool {
        let mut last = false;
        for i in 0..slots {
            let even = i % 2 == 0;
            let fired = if even { r } else { !r };
            last = inv.observe(b + i as u64 * p, p, fired);
        }
        last
    }

    /// The reference locks only after INVERSION_REF_STABLE_FIRES consistent
    /// same-parity real fires -- NOT on the arbitrary first fire (which can
    /// land during the startup re-anchor/freerun thrash and blind the detector
    /// to later phase flips).
    #[test]
    fn inversion_state_reference_locks_after_stable_run() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut inv = InversionState::default();
        // Stabilizing (less than the threshold of consistent even-parity R
        // fires) does NOT lock and never toggles.
        assert!(!drive_stable(&mut inv, b, p, true, 2 * (INVERSION_REF_STABLE_FIRES - 1)));
        assert_eq!(inv.ref_boundary, None, "reference must not lock prematurely");
        assert_eq!(inv.cand_phase, Some(true), "candidate tracks even-parity R");
        // Crossing the threshold promotes the proven candidate to the locked
        // reference (the even-parity R eye at the first candidate boundary).
        assert!(!drive_stable(&mut inv, b, p, true, 2));
        assert_eq!(inv.ref_boundary, Some(b));
        assert_eq!(inv.ref_phase, Some(true));
        assert_eq!(inv.cand_boundary, None, "candidate consumed by the lock");
    }

    /// A stable session never toggles: same-parity fires keep matching the
    /// reference, and the opposite eye at the other parity is normal
    /// alternation (never a false positive).
    #[test]
    fn inversion_state_stable_session_never_toggles() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut inv = InversionState::default();
        drive_stable(&mut inv, b, p, true, 2 * INVERSION_REF_STABLE_FIRES); // lock R at even parity
        // Keep firing the same stable alternation: no flags.
        assert!(!drive_stable(&mut inv, b, p, true, 40));
        assert_eq!(inv.ref_phase, Some(true));
    }

    /// A one-slot phase shift (re-anchor re-lot, DLL phase flip, submit→scanout
    /// offset change) makes same-parity boundaries fire the OPPOSITE eye -- the
    /// detector must flag it so FLAG_INVERT_EYES is toggled.
    #[test]
    fn inversion_state_detects_a_slot_shift() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut inv = InversionState::default();
        drive_stable(&mut inv, b, p, true, 2 * INVERSION_REF_STABLE_FIRES); // lock R at even parity
        // Phase shifts one slot: even boundaries now fire L (was R).
        assert!(
            inv.observe(b + 2 * p, p, false),
            "same-parity fire of the opposite eye must flag inversion"
        );
        // The OTHER parity (odd) firing L is still just alternation of the
        // shifted phase -- not relevant to the reference parity.
        assert!(!inv.observe(b + p, p, true));
    }

    /// After `clear()` (manual button / re-anchor / game restart) the next real
    /// fires re-establish the reference -- a fresh lock-in, no stale memory.
    #[test]
    fn inversion_state_clear_reestablishes_reference() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut inv = InversionState::default();
        drive_stable(&mut inv, b, p, true, 2 * INVERSION_REF_STABLE_FIRES);
        assert_eq!(inv.ref_phase, Some(true));
        inv.clear();
        assert_eq!(inv.ref_boundary, None);
        assert_eq!(inv.ref_phase, None);
        // Next stable run re-locks to whatever the glasses now show (L on the
        // even-relative-to-b3 parity this time).
        drive_stable(&mut inv, b + 3 * p, p, false, 2 * INVERSION_REF_STABLE_FIRES);
        assert_eq!(inv.ref_phase, Some(false));
        // Stable again: the SAME parity class as the new reference (b+5p, same
        // class as b+3p) fires L and matches -> quiet; the OPPOSITE parity
        // (b+6p) fires R -> normal alternation, quiet.
        assert!(!inv.observe(b + 5 * p, p, false));
        assert!(!inv.observe(b + 6 * p, p, true));
    }

    /// The full self-correcting cycle: a shift is flagged once, and once the
    /// renderer swaps (via FLAG_INVERT_EYES) the fired phase returns to the
    /// reference so the detector goes quiet -- no oscillation.
    #[test]
    fn inversion_state_self_corrects_without_oscillation() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut inv = InversionState::default();
        drive_stable(&mut inv, b, p, true, 2 * INVERSION_REF_STABLE_FIRES); // lock R at even parity

        // A slot shift: even boundaries fire L.  Flagged.
        assert!(inv.observe(b + 2 * p, p, false));

        // The renderer swaps; even boundaries fire R again (the swapped DLL
        // phase returns the fired eye to the reference).  Quiet.
        assert!(!inv.observe(b + 4 * p, p, true));
        assert!(!inv.observe(b + 6 * p, p, true));
        // Opposite parity is unchanged alternation.
        assert!(!inv.observe(b + p, p, false));
        assert!(!inv.observe(b + 3 * p, p, false));
    }

    /// The detector is purely phase-relative: different reference boundaries
    /// (odd vs even lock-in) behave symmetrically.
    #[test]
    fn inversion_state_symmetric_for_odd_lockin() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut inv = InversionState::default();
        // Lock-in at an ODD-parity reference boundary (b+p fires L there):
        // odd slots fire L, even slots fire R.
        let b0 = b + p;
        for i in 0..(2 * INVERSION_REF_STABLE_FIRES) {
            let even = i % 2 == 0;
            // At i=0 (slot b0, odd parity): L. At i=1 (b0+p = even): R.
            let fired = if even { false } else { true };
            inv.observe(b0 + i as u64 * p, p, fired);
        }
        assert_eq!(inv.ref_phase, Some(false), "odd-parity lock-in is L");
        // A shift to R at odd parity flags.
        assert!(inv.observe(b0 + 2 * p, p, true), "odd-parity shift detected");
        // Even boundaries firing R is normal alternation of the shifted phase.
        assert!(!inv.observe(b0 + p, p, false));
    }

    /// THE startup-thrash regression (trace.log): the opening seconds contain
    /// several re-anchors/freeruns and the FIFO polarity is still a coin-flip,
    /// so the OLD "first real fire locks the reference" design locked a slot-off
    /// phase and went permanently blind -- later same-parity fires "matched",
    /// and a genuine end-of-session flip read as `invert=0, no detection`.  The
    /// candidate accumulator must NOT lock while the even-parity eye oscillates;
    /// it locks only on the proven, stable phase -- which stays locked long
    /// enough to catch a real flip later (the user's "correct then flipped").
    #[test]
    fn inversion_state_does_not_lock_during_startup_thrash() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut inv = InversionState::default();
        // Startup thrash: even parity fires R,L,R,L,R,L... (the phase keeps
        // flipping as re-anchors re-commit).  The candidate must keep
        // resetting and never lock.
        for i in 0..(2 * INVERSION_REF_STABLE_FIRES) {
            let even = i % 2 == 0;
            let fired = if even { i % 4 < 2 } else { !(i % 4 < 2) };
            inv.observe(b + i as u64 * p, p, fired);
        }
        assert_eq!(
            inv.ref_boundary, None,
            "thrashing phase must never lock a reference"
        );
        // Session settles: even parity consistently R.  The accumulator now
        // builds a stable candidate and locks once proven.
        assert!(!drive_stable(&mut inv, b, p, true, 2 * INVERSION_REF_STABLE_FIRES));
        assert_eq!(inv.ref_boundary, Some(b));
        assert_eq!(inv.ref_phase, Some(true));
        // And once locked, a genuine late flip (the end-of-session re-anchor)
        // IS caught -- even boundaries firing L now flags it.
        assert!(
            inv.observe(b + 2 * p, p, false),
            "after stabilization the detector must catch a real late flip"
        );
        assert!(
            inv.observe(b + 4 * p, p, false),
            "still locked and still flagging until the renderer swaps"
        );
    }

    /// The user's exact scenario: a stable correct session (reference locks on
    /// the proven correct phase), then the end-of-session drift re-anchor
    /// commits the WRONG eye.  The very next same-parity real fire must flag it
    /// (this is what the old first-fire lock missed).
    #[test]
    fn inversion_state_detects_end_of_session_reanchor_flip() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut inv = InversionState::default();
        // Correct session: even parity fires R, stable, long enough to lock.
        drive_stable(&mut inv, b, p, true, 2 * INVERSION_REF_STABLE_FIRES + 40);
        assert_eq!(inv.ref_phase, Some(true), "correct phase locked");

        // Drift accumulates; at the overdue re-anchor the recovery commits the
        // WRONG eye: even boundaries now fire L (the flip the user saw).
        assert!(
            inv.observe(b + 2 * p, p, false),
            "the flip must be detected on the first same-parity real fire"
        );
        // The renderer swaps (FLAG_INVERT_EYES); even boundaries return to R.
        assert!(!inv.observe(b + 4 * p, p, true), "self-corrected, quiet");
        // Odd parity stays normal alternation.
        assert!(!inv.observe(b + p, p, false));
        assert!(!inv.observe(b + 3 * p, p, false));
    }

    // ------------------------------------------------------------------
    // Drift re-sync (DRIFT_LAG_ALIGN_PERIOD_DIV): the FIFO pin chain's tiny
    // systematic period error accumulates over a session, so the front slips
    // progressively behind the armed grid (trace.log `gap_from_armed` grows to
    // ~half a period before the overdue re-anchor).  `next_eye` now re-baselines
    // EARLY -- at ~period/8 lag -- so the shift is small and pairing-preserving,
    // the shutter never opens off-slot, and the overdue half-period (polarity-
    // ambiguous) overhaul never fires on a healthy stream.
    // ------------------------------------------------------------------

    /// A schedule whose front lags the armed boundary by just over period/8
    /// (but far below half a period) must be re-synced onto the grid by the
    /// early path: the front eye fires at the armed boundary, NO overdue drop
    /// happens, NO `pending_reanchor` is set, and the next front stays aligned
    /// to the following armed boundary (drift bounded).
    #[test]
    fn drift_resync_keeps_front_bounded_before_overdue() {
        let p = 8_333u64;
        let boundary = 100_000u64;
        let mut q = EyeQueue::default();
        // Drifted schedule: front just over period/8 behind the armed boundary
        // (the early-sync trigger), the rest of the schedule one period apart
        // behind it.  Well under half a period so the overdue path must NOT
        // run.
        let lag = p / DRIFT_LAG_ALIGN_PERIOD_DIV + 1; // ~period/8 + 1
        assert!(lag < p / 2, "test drift must stay below the overdue threshold");
        q.enqueue(false, Some(boundary - lag), p);
        q.enqueue(true, Some(boundary - lag + p), p);
        q.enqueue(false, Some(boundary - lag + 2 * p), p);

        let fired = q.next_eye(boundary, p);
        assert_eq!(fired, Some(false), "re-synced front eye fires at the armed boundary");
        assert!(!q.pending_reanchor, "early re-sync must not set pending_reanchor");
        assert_eq!(
            q.drops.overdue, 0,
            "early re-sync must not trigger the overdue drop path"
        );
        // The schedule is back on the grid: the next front sits within
        // period/8 of the next armed boundary (the drift stays bounded instead
        // of climbing toward the half-period ambiguity).
        let next_front = q.front_boundary().unwrap_or(boundary + p);
        let lag_limit = p / DRIFT_LAG_ALIGN_PERIOD_DIV;
        assert!(
            next_front + lag_limit >= boundary + p,
            "front {next_front} must sit within period/8 of the next armed boundary {}",
            boundary + p
        );
    }

    /// Re-syncing a drifted schedule must preserve the eye->slot pairing: the
    /// SAME eyes fire at the SAME boundaries as an identical on-grid schedule,
    /// so a corrected session keeps the exact left/right depth it had before
    /// the drift accumulated (no polarity flip from the correction).
    #[test]
    fn drift_resync_preserves_eye_slot_pairing() {
        let p = 8_333u64;
        let boundary = 100_000u64;
        let build = |q: &mut EyeQueue, off: u64| {
            q.enqueue(false, Some(boundary - off), p);
            q.enqueue(true, Some(boundary - off + p), p);
            q.enqueue(false, Some(boundary - off + 2 * p), p);
        };
        // On-grid schedule drives a clean L, R, L sequence.
        let mut clean = EyeQueue::default();
        build(&mut clean, 0);
        let mut clean_fired = Vec::new();
        for i in 0..3u64 {
            clean_fired.push(clean.next_eye(boundary + i * p, p));
        }
        assert_eq!(clean_fired, vec![Some(false), Some(true), Some(false)]);
        // The SAME schedule drifted by just over period/8 (the accumulated pin
        // error) drives to the identical eye sequence once the early re-sync
        // realigns each slot.
        let mut drifted = EyeQueue::default();
        build(&mut drifted, p / DRIFT_LAG_ALIGN_PERIOD_DIV + 1);
        let mut drifted_fired = Vec::new();
        for i in 0..3u64 {
            drifted_fired.push(drifted.next_eye(boundary + i * p, p));
        }
        assert_eq!(
            drifted_fired, clean_fired,
            "drift re-sync must preserve the eye->slot pairing exactly"
        );
    }

    /// A lag just inside the early-sync threshold (below period/8) does NOT
    /// re-sync (no churn on every fire): the healthy one-slot-ahead front stays
    /// put and fires normally.
    #[test]
    fn drift_resync_does_not_churn_on_small_lag() {
        let p = 8_333u64;
        let boundary = 100_000u64;
        let mut q = EyeQueue::default();
        // Healthy front exactly on the armed boundary (the normal steady-state
        // position) -- must fire without any re-baseline.
        q.enqueue(false, Some(boundary), p);
        q.enqueue(true, Some(boundary + p), p);
        let fired = q.next_eye(boundary, p);
        assert_eq!(fired, Some(false));
        // A subtle sub-threshold lag (a few us) must also not churn.
        let mut q = EyeQueue::default();
        q.enqueue(false, Some(boundary - 5), p);
        q.enqueue(true, Some(boundary - 5 + p), p);
        let fired = q.next_eye(boundary, p);
        assert_eq!(fired, Some(false));
        assert_eq!(
            q.front_boundary(),
            Some(boundary + p - 5),
            "sub-threshold lag leaves the schedule exactly where it is"
        );
    }

    /// trace.log's end-of-session failure was a CONTENT STALL, not drift: the
    /// display kept vblanking while the game paused, so the armed grid jumped
    /// whole slots ahead of a schedule that was still ON the display grid
    /// (`front=95159619, gap=2` right before `boundary=95176288`).  The old
    /// overdue path treated that as drift -- dropped the eye, re-baselined,
    /// flagged a re-anchor -- and the recovery shifted the stream ~1 slot off
    /// (`gap_from_armed -3260us`, fire-gap churn, "will NOT lock"), which
    /// latched the inversion the user saw.  An on-grid stall must be DROP-ONLY:
    /// the scanned-out eyes go, the survivors keep their ORIGINAL grid pins
    /// (the display grid never moved), and no re-anchor is flagged.
    #[test]
    fn stall_overdue_drops_passed_eye_without_rebaseline_or_reanchor() {
        let p = 8_333u64;
        let b = 100_000u64;
        let mut q = EyeQueue::default();
        // On-grid FIFO schedule: every eye sits on a real vblank multiple.
        q.enqueue(true, Some(b), p); // R
        q.enqueue(false, Some(b + p), p); // L
        q.enqueue(true, Some(b + 2 * p), p); // R
        q.enqueue(false, Some(b + 3 * p), p); // L

        assert_eq!(q.next_eye(b, p), Some(true), "R fires on b");
        assert!(!q.pending_reanchor);

        // Stall: the armed grid advances one whole slot with no present (slot
        // b+p scanned out).  The passed eye is dropped; the schedule is still
        // on the grid, so no shift and no re-anchor flag.
        assert_eq!(
            q.next_eye(b + 2 * p, p),
            Some(true),
            "grid-aligned R fires on b+2p"
        );
        assert_eq!(q.drops.overdue, 1, "only the passed slot is dropped");
        assert!(
            !q.pending_reanchor,
            "an on-grid stall must not flag a re-anchor (phase is authoritative)"
        );
        assert_eq!(
            q.front_boundary(),
            Some(b + 3 * p),
            "the survivor keeps its ORIGINAL on-grid pin; no whole-stream shift"
        );
        assert_eq!(
            q.next_eye(b + 3 * p, p),
            Some(false),
            "L fires on b+3p: the eye->slot pairing is exactly the pre-stall grid"
        );
    }

    /// The mirror of the drift re-sync: a recovery path (or a boundary that
    /// snapped back) can leave the front pinned MORE than period/8 AHEAD of the
    /// armed grid (the trace.log -3260us post-stall transient).  Firing it
    /// early opens the shutter onto the previous slot -> wrong image.  The same
    /// early path must pull it back onto the armed grid: small, pairing-
    /// preserving, no re-anchor flag.
    #[test]
    fn drift_resync_pulls_front_back_when_leading() {
        let p = 8_333u64;
        let boundary = 100_000u64;
        let mut q = EyeQueue::default();
        let lead = p / DRIFT_LAG_ALIGN_PERIOD_DIV + 1; // just over period/8 ahead
        assert!(lead < p / 2, "test lead must stay below the overdue threshold");
        q.enqueue(true, Some(boundary + lead), p);
        q.enqueue(false, Some(boundary + lead + p), p);
        q.enqueue(true, Some(boundary + lead + 2 * p), p);

        let fired = q.next_eye(boundary, p);
        assert_eq!(
            fired,
            Some(true),
            "the re-based front eye fires at the armed boundary, not its stale lead"
        );
        assert!(!q.pending_reanchor, "early re-sync must not set pending_reanchor");
        assert_eq!(q.drops.overdue, 0, "early re-sync must not trigger the overdue path");
        // The surviving eyes land back on the grid: the next front sits within
        // period/8 of the next armed boundary.
        let next_front = q.front_boundary().unwrap_or(boundary + p);
        let align_limit = p / DRIFT_LAG_ALIGN_PERIOD_DIV;
        assert!(
            (next_front as i64 - (boundary + p) as i64).abs() <= align_limit as i64,
            "front {next_front} must sit within period/8 of the next armed boundary {}",
            boundary + p
        );
    }

    // ------------------------------------------------------------------
    // Re-anchor-storm regression (trace.log): after ONE overdue drop the
    // schedule's front parked PAST every armed boundary, so no eye was ever
    // due, every slot freeran (~120/s of ~120 slots), and the
    // `pending_reanchor` re-anchor kept re-pinning the front forward for ~16
    // seconds.  The first trace storm sat 2-4 periods ahead (978 re-anchors);
    // phase 3 of the newest trace sat only ~0.86 period ahead at every drain
    // (1043 re-anchors) -- inside the old 2-period purge tolerance, which is
    // exactly why it survived.  `purge_stuck_ahead` now drops any schedule
    // whose front is beyond the due-window edge (armed + period/2) so the
    // schedule re-anchors AT the frontier and real eyes fire again.
    // ------------------------------------------------------------------

    // One boundary of the host main loop's drain/wait/arm cadence at 60fps
    // content on a 120Hz display: the game pushes one alternating L/R swap per
    // boundary in the top drain, pinned FIFO onto the anchored grid (ingest's
    // tail floor) preceded by the stuck-ahead storm guard.  Mirrors main.rs:
    // `next_boundary = last_vblank + period`; the wait confirms one period;
    // the arm fires one slot ahead of the just-confirmed vblank.  `stall_hold`
    // keeps the freerun hold below the test horizon so a sag never darkens.
    fn drive_stamped_boundary(
        q: &mut EyeQueue,
        p: u64,
        last_vblank: &mut u64,
        guard: bool,
    ) -> (u64, Option<bool>) {
        let guess = last_vblank.saturating_add(p.saturating_mul(2));
        if guard {
            q.purge_stuck_ahead(guess, p);
        }
        let boundary =
            floored_next_boundary(q.scheduled.back().map(|&(t, _)| t), guess, p);
        let eye = (*last_vblank / p) % 2 != 0; // deterministic alternation
        q.enqueue(eye, Some(boundary), p);
        let confirmed = *last_vblank + p;
        *last_vblank = confirmed;
        let armed = confirmed + p;
        (armed, q.next_eye(armed, p))
    }

    // trace.log PHASE-3 cadence, which the per-slot driver above cannot sustain:
    // a whole [L,R] frame lands in the drain every OTHER vblank (60fps content
    // on 120Hz), and the intermediate vblank drains NOTHING -- so ingest (and
    // the stuck-ahead purge) only runs on the drains that carry swaps, exactly
    // like the live loop's `if !swaps.is_empty()` gate.  The re-anchor then
    // re-pins the front forward by the same 2 periods per 2 vblanks the armed
    // frontier advances, keeping the front parked at a CONSTANT ~0.86*period
    // past the frontier forever (the storm the per-slot cadence drains by
    // itself).
    fn drive_phase3_boundary(
        q: &mut EyeQueue,
        p: u64,
        last_vblank: &mut u64,
        guard: bool,
        batch: bool,
    ) -> (u64, Option<bool>) {
        if batch {
            let guess = last_vblank.saturating_add(p.saturating_mul(2));
            if guard {
                q.purge_stuck_ahead(guess, p);
            }
            let boundary = floored_next_boundary(q.scheduled.back().map(|&(t, _)| t), guess, p);
            let eye = (*last_vblank / p) % 2 != 0; // deterministic L,R frame pair
            q.enqueue(eye, Some(boundary), p);
            q.enqueue(!eye, Some(boundary + p), p);
        }
        let confirmed = *last_vblank + p;
        *last_vblank = confirmed;
        let armed = confirmed + p;
        (armed, q.next_eye(armed, p))
    }

    #[test]
    fn stuck_ahead_guard_drops_only_a_front_beyond_the_due_window() {
        let p = 8_333u64;
        let guess = 100_000u64;
        // A front behind the frontier, or up to the due-window edge
        // (armed + period/2 -- that exact boundary is still fireable), is a
        // normal healthy in-flight eye.
        let mut q = EyeQueue::default();
        q.enqueue(false, Some(guess - 100), p);
        q.enqueue(true, Some(guess + p / 2), p);
        q.purge_stuck_ahead(guess, p);
        assert_eq!(
            q.scheduled.len(),
            2,
            "healthy schedule (front within the due window) must survive the guard"
        );
        // The trace.log PHASE-3 storm state: the re-anchor parks the front
        // ~0.86*period PAST the frontier at every drain -- far inside the old
        // 2-period purge tolerance, so the previous guard never fired -- yet
        // past the due-window edge, so nothing can ever be due and the
        // re-anchor loop keeps it there.  It must be dropped wholesale so the
        // next drain re-anchors at the frontier.
        let mut q = EyeQueue::default();
        q.enqueue(false, Some(guess + p), p);
        q.enqueue(true, Some(guess + 2 * p), p);
        q.purge_stuck_ahead(guess, p);
        assert!(
            q.scheduled.is_empty(),
            "stuck-ahead storm schedule must be purged"
        );
        // A front parked FOUR periods past the guess (trace.log's FIRST storm)
        // is also caught.
        let mut q = EyeQueue::default();
        q.enqueue(false, Some(guess + 4 * p), p);
        q.enqueue(true, Some(guess + 5 * p), p);
        q.purge_stuck_ahead(guess, p);
        assert!(q.scheduled.is_empty());
        // An empty queue is a no-op.
        let mut q = EyeQueue::default();
        q.purge_stuck_ahead(guess, p);
        assert!(q.scheduled.is_empty());
    }

    #[test]
    fn stuck_ahead_storm_recovers_to_on_time_real_fires() {
        let p = 8_333u64;
        // Seed the trace.log storm state: the failed re-anchor left the front
        // ~2 periods PAST the first armed boundary (which the re-anchor then
        // kept pushing to ~4 periods ahead -- the self-sustaining loop).
        let mut q = EyeQueue::default();
        let mut last_vblank = 100_000u64;
        let first_arm = last_vblank + 2 * p;
        q.enqueue(false, Some(first_arm + 2 * p), p);
        q.enqueue(true, Some(first_arm + 3 * p), p);

        let mut freeruns = 0u64;
        let mut reals = 0u64;
        for _ in 0..200u64 {
            let (_, fired) = drive_stamped_boundary(&mut q, p, &mut last_vblank, true);
            if let Some(_f) = fired {
                if q.last_was_freerun {
                    freeruns += 1;
                } else {
                    reals += 1;
                }
            }
        }
        assert!(
            freeruns <= 2,
            "storm must break within ~1 recovery slot, got {freeruns} freeruns"
        );
        assert!(
            reals >= 195,
            "the stream must re-anchor and fire real DLL eyes, got {reals} real fires"
        );
        assert_eq!(
            q.freerun_since, 0,
            "no ongoing freerun once the schedule is on-time"
        );
        assert!(
            !q.pending_reanchor,
            "no pending re-anchor once the schedule is on-time"
        );
        // The schedule drains each boundary: the front tracks the armed grid
        // (empty, or the next boundary's eye) instead of parking ahead.
        let arm_after = last_vblank + p;
        match q.front_boundary() {
            None => {}
            Some(f) => assert!(
                f <= arm_after + p / 2,
                "front {f} must sit within the due window of armed {arm_after}"
            ),
        }
    }

    #[test]
    fn stuck_ahead_storm_control_storms_without_the_guard() {
        // Same seed and cadence WITHOUT the guard: the storm must reproduce
        // (front ~4p ahead of every arm, one re-anchor per 2 slots, ~no real
        // fires) -- proving the guard is the mechanism that fixes the storm.
        let p = 8_333u64;
        let mut q = EyeQueue::default();
        let mut last_vblank = 100_000u64;
        let first_arm = last_vblank + 2 * p;
        q.enqueue(false, Some(first_arm + 2 * p), p);
        q.enqueue(true, Some(first_arm + 3 * p), p);
        // trace.log's storm is a MID-session failure (after an overdue drop),
        // so a real eye has already fired and startup-silence no longer
        // applies -- otherwise the pre-first-eye guard drains the queue on its
        // own and the "without the stuck-ahead guard" storm never forms.
        q.first_dll_eye_fired = true;

        let mut freeruns = 0u64;
        let mut reals = 0u64;
        for _ in 0..200u64 {
            let (_, fired) = drive_stamped_boundary(&mut q, p, &mut last_vblank, false);
            if let Some(_f) = fired {
                if q.last_was_freerun {
                    freeruns += 1;
                } else {
                    reals += 1;
                }
            }
        }
        assert!(
            reals <= 10 && freeruns >= 180,
            "control: without the guard the stream must storm \
             (reals={reals} freeruns={freeruns})"
        );
    }

    #[test]
    fn healthy_pipeline_never_trips_the_stuck_ahead_guard() {
        // A steady 60fps stream from a cold start (empty schedule): every slot
        // fires a real DLL eye and the guard stays dormant -- the front hovers
        // within roughly one period of the frontier.
        let p = 8_333u64;
        let mut q = EyeQueue::default();
        let mut last_vblank = 100_000u64;
        let mut freeruns = 0u64;
        for _ in 0..200u64 {
            let (_, fired) = drive_stamped_boundary(&mut q, p, &mut last_vblank, true);
            if let Some(_f) = fired {
                if q.last_was_freerun {
                    freeruns += 1;
                }
            }
        }
        assert_eq!(freeruns, 0, "healthy stream must fire real eyes every slot");
        assert!(!q.pending_reanchor, "guard must never trip on a healthy stream");
    }

    #[test]
    fn stuck_ahead_guard_breaks_phase3_due_window_storm() {
        // trace.log phase 3: after an overdue re-anchor the front parked
        // ~0.86*period past the frontier at EVERY drain -- just inside the old
        // 2-period purge tolerance, so the previous guard never fired and the
        // session stormed (~1043 freerun re-anchors, ~0 real fires; the stream
        // stayed inverted and the detector starved).  The due-window-edge guard
        // must break it: the first drain after the purge re-anchors at the
        // frontier and every (L,R) frame fires real.
        let p = 8_333u64;
        let mut q = EyeQueue::default();
        let mut last_vblank = 100_000u64;
        let first_arm = last_vblank + 2 * p;
        // The phase-3 steady state the failed re-anchor left behind: front just
        // past the due window, tail one period further, mid freerun/re-anchor.
        q.enqueue(false, Some(first_arm + (7 * p) / 8), p);
        q.enqueue(true, Some(first_arm + (7 * p) / 8 + p), p);
        q.freerun_since = DIP_RESYNC_AFTER_FREERUN_SLOTS;
        q.pending_reanchor = true;

        let mut freeruns = 0u64;
        let mut reals = 0u64;
        for i in 0..400u64 {
            let (_, fired) = drive_phase3_boundary(&mut q, p, &mut last_vblank, true, i % 2 == 0);
            if let Some(_f) = fired {
                if q.last_was_freerun {
                    freeruns += 1;
                } else {
                    reals += 1;
                }
            }
        }
        assert!(
            freeruns <= 4,
            "due-window guard must break the phase-3 storm, got {freeruns} freeruns"
        );
        assert!(
            reals >= 390,
            "schedule must re-anchor at the frontier and fire real eyes, got {reals}"
        );
        assert_eq!(q.freerun_since, 0, "no ongoing freerun after recovery");
        assert!(!q.pending_reanchor, "no pending re-anchor after recovery");
    }

    #[test]
    fn stuck_ahead_guard_phase3_control_storms_without_the_guard() {
        // Control: the SAME phase-3 seed and cadence WITHOUT the guard
        // reproduces the trace storm -- ~zero real fires, endless freeruns and
        // re-anchors -- proving the due-window purge is the mechanism that
        // breaks it.
        let p = 8_333u64;
        let mut q = EyeQueue::default();
        let mut last_vblank = 100_000u64;
        let first_arm = last_vblank + 2 * p;
        q.enqueue(false, Some(first_arm + (7 * p) / 8), p);
        q.enqueue(true, Some(first_arm + (7 * p) / 8 + p), p);
        q.freerun_since = DIP_RESYNC_AFTER_FREERUN_SLOTS;
        q.pending_reanchor = true;
        // Same mid-session seeding as the other control: phase-3 storms happen
        // after real eyes have fired, so the startup-silence latch is off.
        q.first_dll_eye_fired = true;

        let mut freeruns = 0u64;
        let mut reals = 0u64;
        for i in 0..400u64 {
            let (_, fired) = drive_phase3_boundary(&mut q, p, &mut last_vblank, false, i % 2 == 0);
            if let Some(_f) = fired {
                if q.last_was_freerun {
                    freeruns += 1;
                } else {
                    reals += 1;
                }
            }
        }
        assert!(
            reals <= 10 && freeruns >= 380,
            "control: without the guard phase 3 must storm (reals={reals} freeruns={freeruns})"
        );
    }
}
