//! Producer-stamp diagnostics for the eye-inversion diagnosis.
//!
//! Enable with `NVSTUSB_STAMP_DIAG=1`.  For every drained version-2 swap this
//! prints (once per 120Hz slot, summarised per ~1s window):
//!
//! ```text
//! [stamp] seq=... eye=L t_us=... host_now=... age=...us stamped_b=... fifo_b=... delta=...us
//! ```
//!
//! * `t_us`   — the producer-stamped submit time (version-2 ring contract says
//!   host CLOCK_MONOTONIC microseconds).
//! * `now`    — host-epoch microseconds at drain time (same clock as the DRM
//!   vblank timestamps).
//! * `age`    — `now - host_us_from_mono(t_us)`: the DLL's stamp converted to
//!   the host epoch, compared to the host clock NOW.  If the DLL's clock
//!   really were host CLOCK_MONOTONIC, `age` is the present->drain latency
//!   (small, positive, stable).  A large constant means the DLL's clock has a
//!   fixed epoch offset (usable, measured by the offset estimator below);
//!   wildly varying means the stamp carries no reliable slot information at
//!   all.
//! * `stamped_b`/`fifo_b`/`delta` — the display boundary the stamp *would*
//!   assign (if stamped pinning were enabled: measured offset + `stamped_pin`
//!   = `boundary_after` + one period, the armed-grid service slot) vs. the
//!   FIFO pin the host actually used, normalized to [-half, half] period.
//!   `delta`==0 means both schemes agree; non-zero means the stamps land one
//!   slot off the FIFO frontier -- useful for understanding why stamped
//!   pinning shifted the shuttering.  The host ALWAYS uses FIFO; `stamped_b`
//!   is now purely a diagnostic (see `host.rs` module docs).
//!
//! This was originally used to decide the inversion root cause (start-phase
//! parity lot vs. mid-session shift); that diagnostic remains useful even
//! though stamped pinning is retired.
//!
//! # Result on hardware
//!
//! The first session recorded with the diagnostics proved the stamps ARE
//! usable: `age` (the stamped submit converted with the naive CLOCK_MONOTONIC
//! assumption, compared to host now) is CONSTANT to ~±1.3ms across 900+
//! samples (~258ms — Wine's QPC does not start at the host boot epoch, but it
//! is an affine constant, not garbage).  The naive conversion was therefore
//! the whole reason the original stamped pinning shipped the schedule a
//! constant number of periods behind the armed grid.  With that offset
//! measured (a streaming median, below) the estimator could support stamped
//! pinning, but it is RETIRED: on hardware it moved the shuttering off the
//! armed-service grid and broke clean shuttering.  Inversion is instead
//! corrected via the automatic detector toggling `FLAG_INVERT_EYES` (renderer
//! swaps which eye gets which image -- glass timing untouched).  The
//! diagnostics remain useful for tuning and understanding the clock.
//!
//! `NVSTUSB_STAMP_PIN` is RETIRED (no-op; see `host.rs` module docs).

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::nvstusb::drm::DrmVblank;
use crate::shm::Swap;

/// `NVSTUSB_STAMP_DIAG=1` turns the diagnostics on.
pub fn enabled() -> bool {
    std::env::var_os("NVSTUSB_STAMP_DIAG").is_some()
}

/// Deterministic stamped pinning is RETIRED.  It pinned each swap via
/// `boundary_after` (+ one period) off the submit stamp, but on hardware it
/// moved the shuttering off the armed-service grid (a slot too early in its
/// +0 form) and broke clean shuttering -- while FIFO pinning, which the
/// working shutter/phase already uses, is the validated path.  The start-phase
/// inversion it was meant to make deterministic is instead corrected by the
/// automatic inversion detector toggling `FLAG_INVERT_EYES`, which has the
/// RENDERER swap which image goes to which eye (glass/shutter timing
/// untouched), so an inversion is fixed without any uncomfortable shuttering.
///
/// This always returns `false`: the `NVSTUSB_STAMP_PIN` flag is ignored (a
/// leftover no-op kept so an old script setting it cannot silently change
/// behavior).  FIFO pinning is the one-and-only path.
pub fn pin_mode() -> bool {
    false
}

fn active() -> bool {
    enabled() || pin_mode()
}

/// Samples the age estimator must accumulate before `offset_est()` returns
/// `Some` (~50ms of 120Hz content).  Below this the offset is unreliable and
/// `ingest` stays on FIFO pinning.
const MIN_OFFSET_SAMPLES: usize = 6;
/// Rolling age window; the median over this many samples rejects individual
/// drain-latency outliers.
const OFFSET_WINDOW: usize = 24;

/// Rolling ages (`now - naive`) feeding the epoch-offset estimator.  The
/// recording and read functions themselves are ungated (unit-testable); the
/// ACTIVE gating (which modes feed the estimator / ask it for pins) lives at
/// the call sites and in [`record_age_of`].
static AGES: Mutex<VecDeque<i64>> = Mutex::new(VecDeque::new());

/// Feed the epoch-offset estimator one sample: `age` = host-epoch NOW minus
/// the (`host_us_from_mono`-converted) submit stamp.  The streaming median of
/// these is the DLL-clock->host-epoch offset: `age = present_to_drain + offset`,
/// and the constant part IS the offset (`present_to_drain` is sub-period, so
/// the pinning slot is insensitive to it -- any conversion within half a
/// period lands on the same display slot).
pub fn record_age(age: i64) {
    if let Ok(mut q) = AGES.lock() {
        q.push_back(age);
        while q.len() > OFFSET_WINDOW {
            q.pop_front();
        }
    }
}

/// Best current estimate of the DLL-clock->host-epoch offset (median age), or
/// `None` until enough samples exist.  `ingest` converts each stamp with this
/// and pins via `stamped_pin`.
pub fn offset_est() -> Option<i64> {
    let Ok(q) = AGES.lock() else {
        return None;
    };
    if q.len() < MIN_OFFSET_SAMPLES {
        return None;
    }
    let mut v: Vec<i64> = q.iter().copied().collect();
    v.sort_unstable();
    Some(v[v.len() / 2])
}

/// Convenience for the drain path (which already holds the anchor): convert +
/// record the age for one stamped swap.  Gated on a stamp mode being active so
/// the estimator is untouched (and the mutex uncontended) on the legacy path.
pub fn record_age_of(anchor: &DrmVblank, swap: Swap) {
    if !active() {
        return;
    }
    if let Some(t) = swap.t_us {
        let naive = anchor.host_us_from_mono(t);
        let now = anchor.host_epoch_us();
        record_age(now as i64 - naive as i64);
    }
}

/// Per-reset per-swap detail rows (then only periodic summaries).
const ROWS_PER_RESET: u64 = 30;
/// One summary line per ~1s of 120 Hz content.
const SUMMARY_EVERY: u64 = 120;

/// One summary window's accumulators.
#[derive(Default, Clone)]
struct Window {
    ages: Vec<i64>,
    age_min: Option<i64>,
    age_max: Option<i64>,
    deltas: Vec<i64>,
    delta_min: Option<i64>,
    delta_max: Option<i64>,
}

struct StampDiag {
    n: u64,
    rows_left: u64,
    win: Window,
}

static DIAG: Mutex<Option<StampDiag>> = Mutex::new(None);

/// Start a fresh detail-budget + summary window (call on every producer
/// idle/dead edge, i.e. alongside `EyeQueue::reset`).
pub fn reset() {
    // Always clear the offset estimator (also makes the estimator
    // deterministically testable); the DLL-clock offset is a boot constant so
    // a fresh stream simply re-measures it in ~6 slots.
    if let Ok(mut a) = AGES.lock() {
        a.clear();
    }
    if !active() {
        return;
    }
    if let Ok(mut g) = DIAG.lock() {
        *g = Some(StampDiag {
            n: 0,
            rows_left: ROWS_PER_RESET,
            win: Window::default(),
        });
    }
}

/// Log one drained swap against the FIFO boundary the host pinned it to.
pub fn log_swap(
    anchor: &DrmVblank,
    ref_vb: u64,
    period: u64,
    swap: Swap,
    fifo_b: Option<u64>,
) {
    if swap.t_us.is_none() || !enabled() {
        return;
    }
    let Ok(mut g) = DIAG.lock() else {
        return;
    };
    let Some(d) = g.as_mut() else {
        return;
    };
    let per = period.max(1);
    let now = anchor.host_epoch_us();
    let naive = anchor.host_us_from_mono(swap.t_us.unwrap_or(0));
    let age = now as i64 - naive as i64;

    let per_i = per as i64;
    // The display boundary the armed grid serves the swap on once the
    // measured DLL-clock offset is applied -- i.e. the pin stamped mode would
    // assign (`stamped_b`, via `host::stamped_pin` = `boundary_after` + one
    // period), vs. the FIFO pin ingest used (`fifo_b`).  Before enough offset
    // samples exist the raw epoch-naive boundary (same +period convention) is
    // shown.
    let naive_b = ref_vb as i64
        + ((swap.t_us.unwrap_or(0) as i64 - ref_vb as i64).div_euclid(per_i) + 1) * per_i
        + per_i;
    let stamped_b = match offset_est() {
        Some(off) => {
            let converted = (naive as i64 + off).max(0) as u64;
            crate::host::stamped_pin(ref_vb, per, converted) as i64
        }
        None => naive_b,
    };

    let delta = match fifo_b {
        Some(fb) => {
            let mut dl = (stamped_b - fb as i64).rem_euclid(per_i);
            let half = per_i / 2;
            if dl > half {
                dl -= per_i;
            }
            dl
        }
        None => 0,
    };

    if d.rows_left > 0 {
        d.rows_left -= 1;
        let eye = if swap.eye == crate::shm::EYE_RIGHT { "R" } else { "L" };
        println!(
            "[stamp] seq={} eye={} t_us={} host_now={} age={}us stamped_b={} fifo_b={} delta={}us",
            swap.seq,
            eye,
            swap.t_us.map_or(0, |t| t),
            now,
            age,
            stamped_b,
            fifo_b.map_or(0, |b| b as i64),
            delta
        );
    }

    let w = &mut d.win;
    w.ages.push(age);
    w.age_min = Some(w.age_min.map_or(age, |m| m.min(age)));
    w.age_max = Some(w.age_max.map_or(age, |m| m.max(age)));
    w.deltas.push(delta);
    w.delta_min = Some(w.delta_min.map_or(delta, |m| m.min(delta)));
    w.delta_max = Some(w.delta_max.map_or(delta, |m| m.max(delta)));

    d.n += 1;
    if d.n % SUMMARY_EVERY == 0 {
        d.win.ages.sort_unstable();
        d.win.deltas.sort_unstable();
        let med = |v: &Vec<i64>| -> i64 { v[v.len() / 2] };
        println!(
            "[stamp] summary n={} age min/med/max = {}/{}/{} us | delta min/med/max = {}/{}/{} us (of {}us)",
            d.n,
            d.win.age_min.unwrap_or(0),
            med(&d.win.ages),
            d.win.age_max.unwrap_or(0),
            d.win.delta_min.unwrap_or(0),
            med(&d.win.deltas),
            d.win.delta_max.unwrap_or(0),
            per
        );
        d.win = Window::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_estimator_needs_minimum_samples() {
        reset();
        assert_eq!(offset_est(), None, "no samples -> no estimate");
        for i in 0..MIN_OFFSET_SAMPLES - 1 {
            record_age(250_000 + i as i64);
        }
        assert_eq!(
            offset_est(),
            None,
            "below MIN_OFFSET_SAMPLES the estimate must stay None (ingest stays FIFO)"
        );
    }

    #[test]
    fn offset_estimator_returns_streaming_median() {
        reset();
        // A stable ~258ms Wine QPC offset with one outlier (a drain hiccup).
        for _ in 0..20 {
            record_age(258_000);
            record_age(258_010);
            record_age(258_005);
        }
        record_age(500_000); // outlier
        let est = offset_est().expect("enough samples");
        assert!(
            (258_000..=258_010).contains(&est),
            "median must reject the outlier, got {est}"
        );
    }
}