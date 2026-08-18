//! Kernel vblank anchor via the DRM/KMS ioctl interface.
//!
//! The compositor's swap return wobbles by hundreds of microseconds on
//! Hyprland/Wayland, which makes the host-phase sleep (vblank method 1) a
//! jittery anchor for the IR packet.  This module instead anchors the IR
//! packet to the display engine's real vblank clock, which is stable to a
//! microsecond, by opening /dev/dri/card* and waiting on
//! `DRM_IOCTL_WAIT_VBLANK`.
//!
//! Design: the frame loop predicts the next presentation vblank (previous
//! kernel-confirmed vblank + measured period), busy-waits until 3000us before
//! it (the emitter's fixed packet->IR alarm delay), then sends the eye packet
//! and lets `swap_buffers` block to that same vblank.  Every
//! [`DrmVblank::resync_every`] frames the prediction is re-anchored to cancel
//! CPU/GPU clock drift.
//!
//! The re-anchor must NOT block on a fresh `DRM_IOCTL_WAIT_VBLANK`: in the KMS
//! path the swap already blocks until `DRM_EVENT_FLIP_COMPLETE`, i.e. the swap
//! return time *is* a kernel-confirmed vblank.  Issuing a second blocking wait
//! would stall the frame loop for up to a full period on every re-sync (a
//! visible 2-frame freeze every ~1.25s) and, while stalled, the emitter's
//! stale-repeat keeps firing the previously-sent eye against the newly flipped
//! opposite-eye content - a wrong-eye ghost frame on every re-sync.  So the
//! re-anchor simply samples the swap-return host epoch (the flip's vblank
//! time) and re-measures the period over the full re-sync interval.

use std::fs::File;
use std::mem::size_of;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::raw::{c_int, c_long, c_uint, c_ulong};
use std::time::{Duration, Instant};

// --- Minimal DRM ABI (drm.h, stable for decades). ---

// The kernel exposes drm_wait_vblank as a *union* (linux/drm.h UAPI, stable
// for decades): a 16-byte request and a 24-byte reply that OVERLAP at offset
// 0.  On 64-bit Linux the request's `signal` is `unsigned long` (8 bytes) and
// the reply is `{ type, sequence, long tval_sec, long tval_usec }` (24 bytes),
// so the union is 24 bytes and DRM_IOCTL_WAIT_VBLANK (= _IOWR('d', 0x3a,
// sizeof)) carries size 24.  Modelling request/reply as struct members laid out
// one after the other misplaces the reply: the kernel writes it at offset 0
// (overlapping the request), so the reply fields read back as garbage -- a
// bogus 0 us or "vblank wait failed" on an otherwise live CRTC.
#[repr(C)]
#[derive(Clone, Copy)]
struct DrmWaitVblankRequest {
    r#type: c_uint,
    sequence: c_uint,
    signal: c_ulong,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmWaitVblankReply {
    r#type: c_uint,
    sequence: c_uint,
    tval_sec: c_long,
    tval_usec: c_long,
}

#[repr(C)]
#[derive(Clone, Copy)]
union DrmWaitVblank {
    request: DrmWaitVblankRequest,
    reply: DrmWaitVblankReply,
}

const DRM_VBLANK_RELATIVE: c_int = 0x1;
const DRM_VBLANK_NEXTONMISS: c_int = 0x10000000;
/// Pipe (CRTC index) selection bits, as used by drmWaitVBlank.
const DRM_VBLANK_CRTC_SHIFT: u32 = 24;

const fn ioc(dir: u32, ty: u8, nr: u8, size: usize) -> c_ulong {
    ((dir as c_ulong) << 30)
        | ((size as c_ulong) << 16)
        | ((ty as c_ulong) << 8)
        | (nr as c_ulong)
}
const IOC_READ_WRITE: u32 = 3;
const DRM_IOCTL_WAIT_VBLANK: c_ulong =
    ioc(IOC_READ_WRITE, b'd', 0x3a, size_of::<DrmWaitVblank>());

unsafe extern "C" {
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
}

#[repr(C)]
struct Timespec {
    tv_sec: c_long,
    tv_nsec: c_long,
}
// The kernel fills drm_wait_vblank_reply from drm_vblank_count_and_time(),
// which reports CLOCK_MONOTONIC since linux-4.15 (drm_vblank.c,
// drm_wait_vblank_reply).  The same clock must be sampled on our side so the
// host-epoch conversion is consistent.
const CLOCK_MONOTONIC: c_int = 1;

unsafe extern "C" {
    fn clock_gettime(clk_id: c_int, tp: *mut Timespec) -> c_int;
}

fn clock_us() -> u64 {
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { clock_gettime(CLOCK_MONOTONIC, &mut ts) };
    let secs = ts.tv_sec.max(0) as u64;
    let nsecs = ts.tv_nsec.max(0) as u64;
    secs.saturating_mul(1_000_000).saturating_add(nsecs / 1000)
}

// --- DrmVblank ---

const PERIOD_MIN_US: u64 = 7_600;
const PERIOD_MAX_US: u64 = 9_000;
/// The emitter's fixed packet -> IR alarm delay (FRAME_ALARM_DELAY_US).
const ALARM_DELAY_US: u64 = 3_000;
/// Default re-sync interval in frames (~1.25 s at 120 Hz).
const DEFAULT_RESYNC_EVERY: u32 = 150;

pub struct DrmVblank {
    fd: OwnedFd,
    /// Pipe (CRTC index) whose vblank clock we wait on.
    pipe: u32,
    /// Measured vblank period in microseconds.
    period_us: u64,
    /// Host-epoch microseconds of the vblank at which the current frame
    /// presents (predicted, re-anchored every `resync_every` frames).
    next_present_us: u64,
    /// Host-epoch microseconds of the last kernel-confirmed vblank.
    confirmed_us: u64,
    frames_since_resync: u32,
    resync_every: u32,
    /// Host epoch (CLOCK_MONOTONIC) of the swap-return sample that last
    /// re-anchored the prediction (kernel-confirmed via FLIP_COMPLETE).
    last_resync_us: u64,
    /// Frames between the previous and current re-anchor samples; used to
    /// re-measure the period over the full re-sync interval.
    last_resync_frames: u32,
    /// `Instant` corresponding to host-epoch 0.
    epoch_instant: Instant,
    /// Monotonic value at `epoch_instant`, to convert the kernel's monotonic
    /// vblank timestamps into host-epoch microseconds.
    real_epoch_offset_us: u64,
    /// Set once a computed deadline was in the future (schedule is live).
    synced: bool,
    /// Set once an ioctl fails; callers fall back to method 1 behavior.
    broken: bool,
    /// Re-anchor diagnostics: how far the predicted present was from the
    /// kernel-confirmed vblank (count, running sum, max abs).
    resync_count: u64,
    resync_err_total: i64,
    resync_err_max_abs: i64,
}

impl DrmVblank {
    /// Opens the first /dev/dri/card* whose vblank counter advances. A
    /// specific card can be forced with NVSTUSB_DRM_CARD=/dev/dri/cardN
    /// (probed first; the scan continues if it is unusable).
    pub fn open() -> Option<DrmVblank> {
        let mut reasons: Vec<String> = Vec::new();
        let mut order: Vec<String> = (0..4).map(|i| format!("/dev/dri/card{i}")).collect();
        if let Some(p) = std::env::var_os("NVSTUSB_DRM_CARD") {
            if let Ok(s) = p.into_string() {
                let s = s.trim().to_string();
                if !s.is_empty() {
                    order.retain(|p| p != &s);
                    order.insert(0, s);
                }
            }
        }
        for path in order {
            let file = match File::options().read(true).write(true).open(&path) {
                Ok(f) => f,
                Err(e) => {
                    reasons.push(format!("{path}: {e}"));
                    continue;
                }
            };
            let epoch_instant = Instant::now();
            let real_epoch_offset_us = clock_us();
            let mut chosen: Option<DrmVblank> = None;
            // Probe each CRTC pipe; the active one advances the master counter
            // with a stable period, inactive ones fail or hang.
            let mut pipe_diag: Vec<String> = Vec::new();
            for pipe in 0..8u32 {
                let fd = match file.try_clone() {
                    Ok(fd) => fd,
                    Err(e) => {
                        pipe_diag.push(format!("pipe{pipe}: {e}"));
                        continue;
                    }
                };
                let mut d = match Self::new(fd, epoch_instant, real_epoch_offset_us, pipe) {
                    Ok(d) => d,
                    Err(e) => {
                        pipe_diag.push(format!("pipe{pipe}: {e}"));
                        continue;
                    }
                };
                // Measure the period with two blocking waits (NEXTONMISS can
                // return the same vblank back-to-back, giving a 0 us period).
                match (d.wait_vblank_blocking(), d.wait_vblank_blocking()) {
                    (Some(t0), Some(t1)) => {
                        let p = t1.saturating_sub(t0);
                        d.period_us = p;
                        d.confirmed_us = t1;
                        d.next_present_us = t1.saturating_add(p);
                    }
                    _ => {
                        pipe_diag.push(format!("pipe{pipe}: vblank wait failed"));
                        continue;
                    }
                }
                pipe_diag.push(format!("pipe{pipe}: {} us", d.period_us));
                if (PERIOD_MIN_US..=PERIOD_MAX_US).contains(&d.period_us) {
                    chosen = Some(d);
                    break;
                }
            }
            let d = match chosen {
                Some(d) => d,
                None => {
                    reasons.push(format!(
                        "{path}: no usable pipe ({})",
                        pipe_diag.join("; ")
                    ));
                    continue;
                }
            };
            if let Ok(rs) = std::env::var("NVSTUSB_DRM_RESYNC") {
                if let Ok(v) = rs.parse::<u32>() {
                    let _ = v; // resync handled by the paced anchor
                }
            }
            eprintln!(
                "nvstusb: DRM vblank anchor on {path} pipe {} (period {} us)",
                d.pipe, d.period_us
            );
            return Some(d);
        }
        eprintln!("nvstusb: DRM vblank unavailable: {}", reasons.join("; "));
        None
    }

    fn new(
        file: File,
        epoch_instant: Instant,
        real_epoch_offset_us: u64,
        pipe: u32,
    ) -> Result<DrmVblank, std::io::Error> {
        let fd = OwnedFd::from(file);
        let mut v = DrmWaitVblank {
            request: DrmWaitVblankRequest {
                r#type: (DRM_VBLANK_RELATIVE
                    | DRM_VBLANK_NEXTONMISS
                    | (pipe << DRM_VBLANK_CRTC_SHIFT) as c_int) as c_uint,
                sequence: 1,
                signal: 0,
            },
        };
        let r = unsafe { ioctl(fd.as_raw_fd(), DRM_IOCTL_WAIT_VBLANK, &mut v) };
        if r != 0 {
            return Err(std::io::Error::last_os_error());
        }
        // Safe: a successful WAIT_VBLANK fills the reply union member.
        if unsafe { v.reply }.sequence == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "vblank sequence 0",
            ));
        }
        Ok(DrmVblank {
            fd,
            pipe,
            period_us: 0,
            next_present_us: 0,
            confirmed_us: 0,
            frames_since_resync: 0,
            resync_every: DEFAULT_RESYNC_EVERY,
            last_resync_us: 0,
            last_resync_frames: 0,
            epoch_instant,
            real_epoch_offset_us,
            synced: false,
            broken: false,
            resync_count: 0,
            resync_err_total: 0,
            resync_err_max_abs: 0,
        })
    }

    /// Blocking wait for the next vblank on the configured pipe; returns its
    /// timestamp in host-epoch microseconds.
    pub fn wait_vblank(&mut self) -> Option<u64> {
        self.wait_vblank_flags(DRM_VBLANK_RELATIVE | DRM_VBLANK_NEXTONMISS)
    }

    /// Same as [`wait_vblank`] but without `NEXTONMISS`, so it always blocks
    /// until the next vblank actually occurs.  Used for period measurement,
    /// where `NEXTONMISS` can return the same (current) vblank timestamp on
    /// back-to-back calls.
    pub fn wait_vblank_blocking(&mut self) -> Option<u64> {
        self.wait_vblank_flags(DRM_VBLANK_RELATIVE)
    }

    fn wait_vblank_flags(&mut self, flags: c_int) -> Option<u64> {
        let mut v = DrmWaitVblank {
            request: DrmWaitVblankRequest {
                r#type: (flags | (self.pipe << DRM_VBLANK_CRTC_SHIFT) as c_int) as c_uint,
                sequence: 1,
                signal: 0,
            },
        };
        let r = unsafe { ioctl(self.fd.as_raw_fd(), DRM_IOCTL_WAIT_VBLANK, &mut v) };
        if r != 0 {
            self.broken = true;
            return None;
        }
        // Safe: a successful WAIT_VBLANK fills the reply union member.
        let reply = unsafe { v.reply };
        let sec = reply.tval_sec;
        let usec = reply.tval_usec;
        if sec == 0 && usec == 0 {
            return None;
        }
        let mono_us = (sec.max(0) as u64)
            .saturating_mul(1_000_000)
            .saturating_add(usec.max(0) as u64);
        Some(mono_us.saturating_sub(self.real_epoch_offset_us))
    }

    /// Returns the `Instant` to busy-wait to before sending the eye packet:
    /// `next present vblank - alarm delay - host lead`.  If we are already
    /// late for this frame, returns now (frame_end will re-anchor).
    pub fn frame_start(&mut self, lead_us: u32) -> Option<Instant> {
        if self.broken || self.period_us == 0 {
            return None;
        }
        let target = self.next_present_us;
        let deadline_epoch = target.saturating_sub(ALARM_DELAY_US + lead_us as u64);
        let now = self.epoch_instant.elapsed().as_micros() as u64;
        if deadline_epoch <= now {
            // We're late: send immediately; frame_end will re-anchor.
            return Some(Instant::now());
        }
        self.synced = true;
        Some(self.epoch_instant + Duration::from_micros(deadline_epoch))
    }

    /// Called after the swap.  Advances the predicted presentation vblank,
    /// nudges it every frame toward the just-observed flip time, and
    /// periodically does a full re-anchor + period re-measurement on a
    /// kernel-confirmed vblank.  Until the first future deadline proves the
    /// schedule is live, full-resyncs every frame (the startup anchor from
    /// `open()` is stale by then).
    ///
    /// `precise_us` should be the DRM page-flip event's own hardware
    /// timestamp (see [`super::kms::KmsDisplay::present`]) converted via
    /// [`host_us_from_mono`], when available - it has no scheduler/wakeup
    /// jitter, unlike a post-syscall `Instant::now()` sample. Pass `None` to
    /// fall back to [`host_epoch_us`].
    ///
    /// The previous version only ever corrected the prediction once every
    /// `resync_every` frames (~1.25s), dead-reckoning with the last measured
    /// `period_us` in between. Any small clock-drift or one-off scheduling
    /// hiccup during that gap stayed uncorrected for up to 1.25s, which
    /// showed up as shutter/frame-boundary misalignment (visible as edge
    /// crosstalk) that would build up and then snap away at the next
    /// resync. The per-frame nudge below now cancels that drift every
    /// frame instead of leaving it to accumulate.
    pub fn frame_end(&mut self, precise_us: Option<u64>) {
        if self.broken {
            return;
        }
        let this_present = self.next_present_us;
        self.next_present_us = this_present.saturating_add(self.period_us);
        self.frames_since_resync += 1;

        let t = precise_us.unwrap_or_else(|| self.host_epoch_us());

        // Normalize the raw flip-time offset against the predicted present
        // into +/-half period so the reported value is the true prediction
        // error (positive = prediction early).
        let raw = t as i64 - this_present as i64;
        let half = (self.period_us / 2) as i64;
        let mut err = raw;
        if err > half {
            err -= self.period_us as i64;
        } else if err < -half {
            err += self.period_us as i64;
        }

        // Gentle per-frame phase lock: fold 1/8th of this frame's error into
        // the next prediction. Small enough not to chase normal sample
        // jitter, fast enough to cancel real drift well inside one full
        // resync interval. Skipped pre-sync (the schedule isn't live yet -
        // `frame_start` already re-anchors every frame in that case) and on
        // an out-of-range sample (a genuine multi-vblank miss, which the
        // full resync path below handles by re-measuring the period).
        if self.synced && err.abs() < half {
            self.next_present_us = (self.next_present_us as i64 + err / 8).max(0) as u64;
        }

        if self.synced && self.frames_since_resync < self.resync_every {
            return;
        }
        self.frames_since_resync = 0;
        self.resync_count += 1;
        self.resync_err_total += err;
        self.resync_err_max_abs = self.resync_err_max_abs.max(err.abs());
        if t > self.confirmed_us {
            let p = t - self.confirmed_us;
            if (PERIOD_MIN_US..=PERIOD_MAX_US).contains(&p) {
                self.period_us = p;
            }
        }
        self.confirmed_us = t;
        // Re-measure the period over the full re-sync interval when enough
        // frames elapsed between samples, then re-anchor the prediction on the
        // just-confirmed swap-return vblank.
        if self.last_resync_frames >= 2 && self.last_resync_us > 0 && t > self.last_resync_us {
            let p = (t - self.last_resync_us) / self.last_resync_frames as u64;
            if (PERIOD_MIN_US..=PERIOD_MAX_US).contains(&p) {
                self.period_us = p;
            }
        }
        self.last_resync_us = t;
        self.last_resync_frames = self.resync_every;
        self.next_present_us = t.saturating_add(self.period_us);
    }

    /// Predicted host-epoch present of the frame currently being sent (valid
    /// between `frame_start` and `frame_end`).
    pub fn current_present_us(&self) -> u64 {
        self.next_present_us
    }

    /// Measured vblank period in microseconds.
    pub fn period_us(&self) -> u64 {
        self.period_us
    }

    /// Converts a host-epoch vblank timestamp (as returned by [`wait_vblank`])
    /// into an `Instant` on the same clock the paced stream uses for its
    /// sleep/spin deadline.
    pub fn instant_of(&self, host_epoch_us: u64) -> Instant {
        self.epoch_instant + Duration::from_micros(host_epoch_us)
    }

    /// Host-epoch microseconds now, on the same clock as the vblank
    /// timestamps (both relative to `epoch_instant`).
    pub fn host_epoch_us(&self) -> u64 {
        self.epoch_instant.elapsed().as_micros() as u64
    }

    /// Converts a raw CLOCK_MONOTONIC microsecond timestamp (e.g. from a DRM
    /// page-flip event) into the same host-epoch microseconds as
    /// [`host_epoch_us`]/the vblank timestamps.
    pub fn host_us_from_mono(&self, mono_us: u64) -> u64 {
        mono_us.saturating_sub(self.real_epoch_offset_us)
    }

    /// Re-anchor diagnostics: (count, running sum of prediction error us,
    /// max abs error us).
    pub fn resync_stats(&self) -> (u64, i64, i64) {
        (
            self.resync_count,
            self.resync_err_total,
            self.resync_err_max_abs,
        )
    }

    /// Resets the re-anchor diagnostics accumulators (see
    /// `NvstusbContext::reset_drm_resync_stats`).
    pub fn reset_resync_stats(&mut self) {
        self.resync_count = 0;
        self.resync_err_total = 0;
        self.resync_err_max_abs = 0;
    }

    /// Forces the next `frame_end` to re-anchor on a kernel-confirmed vblank.
    /// Used after a KMS mode-set, which can shift the vblank phase.
    pub fn force_resync(&mut self) {
        self.frames_since_resync = self.resync_every;
    }
}
