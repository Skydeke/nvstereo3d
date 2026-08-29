//! Kernel vblank anchor via the DRM/KMS ioctl interface.
//!
//! The compositor's swap return wobbles by hundreds of microseconds on
//! Hyprland/Wayland, which makes the swap return a jittery anchor for the IR
//! packet.  This module instead anchors the IR packet to the display engine's
//! real vblank clock, which is stable to a microsecond, by opening
//! /dev/dri/card* and waiting on the vblank.
//!
//! The preferred mechanism is the modern, driver-agnostic CRTC-sequence
//! ioctls, `DRM_IOCTL_CRTC_GET_SEQUENCE` / `DRM_IOCTL_CRTC_QUEUE_SEQUENCE`
//! (see the ABI notes below): they are CORE-table ioctls served on every KMS
//! driver - i915, amdgpu and nvidia-drm alike - and address a CRTC by object
//! id.  The legacy `DRM_IOCTL_WAIT_VBLANK` is retained only as a fallback for
//! the case where no CRTC id can be resolved because a driver (nvidia-drm)
//! denies enumeration to a non-master client.
//!
//! Design: the frame loop predicts the next presentation vblank (previous
//! kernel-confirmed vblank + measured period), busy-waits until 3000us before
//! it (the emitter's fixed packet->IR alarm delay), then sends the eye packet
//! and lets `swap_buffers` block to that same vblank.  Every
//! [`DrmVblank::resync_every`] frames the prediction is re-anchored to cancel
//! CPU/GPU clock drift.
//!
//! The re-anchor must NOT block on a fresh vblank wait: in the KMS
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
use std::os::raw::{c_int, c_long, c_short, c_uint, c_ulong};
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
/// CRTC selection for WAIT_VBLANK, per the UAPI (include/uapi/drm/drm.h):
/// pipes >= 1 are encoded in the HIGH_CRTC field, type bits 1..5. The kernel
/// decodes `(type & 0x3e) >> 1` (drm_vblank.c: drm_wait_vblank_ioctl); pipe 0
/// needs no bits at all. NOTE: the "CRTC_SHIFT 24" encoding found in old
/// snippets is NOT part of the UAPI - bit 24 is undefined, and the kernel's
/// validation mask (`TYPES | FLAGS | HIGH_CRTC`) rejects it with EINVAL, so
/// every wait on pipe >= 1 silently failed until this was fixed.
const DRM_VBLANK_HIGH_CRTC_MASK: u32 = 0x0000_003e;
const DRM_VBLANK_HIGH_CRTC_SHIFT: u32 = 1;

/// Encodes a CRTC pipe index into WAIT_VBLANK `request.type` bits.
const fn vblank_pipe_bits(pipe: u32) -> c_int {
    if pipe == 0 {
        0
    } else {
        ((pipe << DRM_VBLANK_HIGH_CRTC_SHIFT) & DRM_VBLANK_HIGH_CRTC_MASK) as c_int
    }
}

const fn ioc(dir: u32, ty: u8, nr: u8, size: usize) -> c_ulong {
    ((dir as c_ulong) << 30)
        | ((size as c_ulong) << 16)
        | ((ty as c_ulong) << 8)
        | (nr as c_ulong)
}
const IOC_READ_WRITE: u32 = 3;
const DRM_IOCTL_WAIT_VBLANK: c_ulong =
    ioc(IOC_READ_WRITE, b'd', 0x3a, size_of::<DrmWaitVblank>());

// --- Modern CRTC-sequence vblank ABI (drm.h UAPI, since linux-4.13). ---
//
// DRM_IOCTL_CRTC_GET_SEQUENCE (0x3b) and DRM_IOCTL_CRTC_QUEUE_SEQUENCE
// (0x3c) are the modern, driver-agnostic replacement for the legacy
// DRM_IOCTL_WAIT_VBLANK.  WAIT_VBLANK is a *driver-table* ioctl: a driver
// that does not install it in its own `ioctls` table (nvidia-drm notably
// does not) answers it with EOPNOTSUPP ("Inappropriate/Operation not
// supported").  The CRTC sequence ioctls, by contrast, are *core-table*
// ioctls: their numbers sit below DRM_COMMAND_BASE (0x40), so the DRM core
// dispatches them from the fixed global `drm_ioctls[]` table in
// drm_ioctl.c - engineered on top of the generic drm_vblank.c machinery -
// and they therefore work on EVERY KMS driver (i915, amdgpu, nvidia-drm,
// ...) provided the driver initialized vblank via `drm_vblank_init`.
//
// Two differences from WAIT_VBLANK matter to the caller:
//   * They address a CRTC by its object ID (`crtc_id`), not a pipe index.
//   * They report timestamps in CLOCK_MONOTONIC nanoseconds (`sequence_ns`
//     / `time_ns`), the same clock the rest of this module converts to
//     host-epoch microseconds via `real_epoch_offset_us`.
//
// On nvidia-drm, `drm_vblank_init` is gated behind the `vblank=1` module
// parameter on the 600/610-series drivers (see README).  On 595 and older
// there is no such option and vblank cannot be enabled on a >= 4.19 kernel
// at all.  Until vblank is initialized on branches that can,
// BOTH the legacy and the modern ioctls return EOPNOTSUPP there.

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmCrtcGetSequence {
    crtc_id: u32,
    active: u32,
    /// Monotonic-frame counter of the last vblank on this CRTC.
    sequence: u64,
    /// CLOCK_MONOTONIC nanoseconds of that vblank (the field the kernel
    /// fills in-place when the `ioctl` succeeds).
    sequence_ns: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmCrtcQueueSequence {
    crtc_id: u32,
    flags: u32,
    sequence: u64,
    user_data: u64,
}

/// `DRM_CRTC_SEQUENCE_RELATIVE`: `sequence` is an offset added to the current
/// counter, rather than an absolute sequence number (drm.h UAPI).
const DRM_CRTC_SEQUENCE_RELATIVE: u32 = 0x0000_0001;

/// A queued CRTC sequence is delivered as a pollable/readable frame event on
/// the same fd (like a page-flip event) - `drm_event` header plus one of
/// these.  `DRM_EVENT_CRTC_SEQUENCE` == 0x03, and `DRM_EVENT_FLIP_COMPLETE`
/// == 0x02 (drm.h UAPI).
const DRM_EVENT_CRTC_SEQUENCE: u32 = 0x03;

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmEventBase {
    length: u32,
    r#type: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmEventCrtcSequence {
    base: DrmEventBase,
    user_data: u64,
    /// CLOCK_MONOTONIC nanoseconds of the vblank that matched the request
    /// (signed in the UAPI).
    time_ns: i64,
    sequence: u64,
}

const DRM_IOCTL_CRTC_GET_SEQUENCE: c_ulong =
    ioc(IOC_READ_WRITE, b'd', 0x3b, size_of::<DrmCrtcGetSequence>());
const DRM_IOCTL_CRTC_QUEUE_SEQUENCE: c_ulong =
    ioc(IOC_READ_WRITE, b'd', 0x3c, size_of::<DrmCrtcQueueSequence>());

const _: () = {
    assert!(size_of::<DrmCrtcGetSequence>() == 24);
    assert!(size_of::<DrmCrtcQueueSequence>() == 24);
    assert!(size_of::<DrmEventBase>() == 8);
    assert!(size_of::<DrmEventCrtcSequence>() == 32);
    assert!(DRM_IOCTL_CRTC_GET_SEQUENCE as u64 == 0xc018_643b);
    assert!(DRM_IOCTL_CRTC_QUEUE_SEQUENCE as u64 == 0xc018_643c);
};

// --- Minimal KMS connector/CRTC ABI (drm_mode.h, stable UAPI). ---
//
// Read-only mode queries (GETRESOURCES/GETCONNECTOR/GETENCODER) are permitted
// without DRM master and even on render nodes, which lets a plain client
// discover which CRTC (pipe) scans out which connector - the missing link
// between "the window is on output DP-1" and "WAIT_VBLANK pipe N".
//
// EXCEPTION to its simplicity: GETRESOURCES uses the two-call pattern, and
// pass 1 writes back ALL four list counts. Pass 2 must then supply buffers -
// or explicitly re-zeroed counts - for EVERY list, because the kernel copies
// any list whose user-supplied count covers the real one. A stale non-zero
// count over a NULL pointer is copy_to_user(0x0) = EFAULT ("Bad address"),
// on every driver; this exact mistake once made enumeration look "denied".
// libdrm allocates all four arrays; we zero the two we don't need.
//
// The struct layouts below are counter-intuitive in several places (see the
// field comments) - they were verified against /usr/include/drm/drm_mode.h
// with offsetof(), and pinned by the const asserts underneath so any drift
// breaks the build instead of silently corrupting kernel ABI.

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmModeCardRes {
    framebuffer_id_ptr: u64,
    crtc_id_ptr: u64,
    connector_id_ptr: u64,
    encoder_id_ptr: u64,
    count_fbs: u32,
    count_crtcs: u32,
    count_connectors: u32,
    count_encoders: u32,
    min_width: u32,
    max_width: u32,
    min_height: u32,
    max_height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmModeGetConnector {
    encoders_ptr: u64,
    modes_ptr: u64,
    props_ptr: u64,
    prop_values_ptr: u64,
    count_modes: u32,
    /// NOTE: props comes BEFORE encoders in the UAPI struct.
    count_props: u32,
    count_encoders: u32,
    /// Object ID of the CURRENTLY-BOUND encoder (0 = none). Filled directly
    /// by the kernel - no need to fetch the encoders_ptr array just for it.
    encoder_id: u32,
    connector_id: u32,
    connector_type: u32,
    connector_type_id: u32,
    connection: u32,
    mm_width: u32,
    mm_height: u32,
    subpixel: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmModeGetEncoder {
    encoder_id: u32,
    encoder_type: u32,
    crtc_id: u32,
    possible_crtcs: u32,
    possible_clones: u32,
}

/// drm_mode_connector_status: only `Connected` (1) has an active encoder.
const DRM_MODE_CONNECTED: u32 = 1;

const DRM_IOCTL_MODE_GETRESOURCES: c_ulong =
    ioc(IOC_READ_WRITE, b'd', 0xA0, size_of::<DrmModeCardRes>());
const DRM_IOCTL_MODE_GETENCODER: c_ulong =
    ioc(IOC_READ_WRITE, b'd', 0xA6, size_of::<DrmModeGetEncoder>());
const DRM_IOCTL_MODE_GETCONNECTOR: c_ulong =
    ioc(IOC_READ_WRITE, b'd', 0xA7, size_of::<DrmModeGetConnector>());

// Pin the ABI: sizes AND the fully-encoded ioctl numbers (which encode the
// struct size), cross-checked against the values drm_mode.h computes on
// x86_64. A struct edit that changes layout fails here at compile time
// instead of corrupting memory in the kernel at runtime.
const _: () = {
    assert!(size_of::<DrmModeCardRes>() == 64);
    assert!(size_of::<DrmModeGetConnector>() == 80);
    assert!(size_of::<DrmModeGetEncoder>() == 20);
    assert!(DRM_IOCTL_MODE_GETRESOURCES as u64 == 0xc040_64a0);
    assert!(DRM_IOCTL_MODE_GETENCODER as u64 == 0xc014_64a6);
    assert!(DRM_IOCTL_MODE_GETCONNECTOR as u64 == 0xc050_64a7);
};

/// Kernel display-name for a `DRM_MODE_CONNECTOR_*` type value (the strings
/// from `drm_connector_enum_list` in drivers/gpu/drm/drm_connector.c). A
/// connector's userspace name is `<type>-<index>`, e.g. `DP-1`, `HDMI-A-2`.
pub(crate) fn connector_type_name(ty: u32) -> &'static str {
    match ty {
        1 => "VGA",
        2 => "DVI-I",
        3 => "DVI-D",
        4 => "DVI-A",
        5 => "Composite",
        6 => "SVIDEO",
        7 => "LVDS",
        8 => "Component",
        9 => "DIN",
        10 => "DP",
        11 => "HDMI-A",
        12 => "HDMI-B",
        13 => "TV",
        14 => "eDP",
        15 => "Virtual",
        16 => "DSI",
        17 => "DPI",
        _ => "Unknown",
    }
}

/// Normalizes a display/connector name for comparison across compositors and
/// the kernel: lower-case, alphanumerics + dashes only, with the common
/// spellings mapped onto the kernel's names (`DisplayPort-1` -> `dp-1`,
/// `HDMI-1` -> `hdmi-a-1`). Compositors report `wl_output` names that usually
/// equal the kernel connector name (wlroots/KWin/Hyprland do), but e.g. some
/// GNOME builds spell it `DisplayPort-1`.
pub(crate) fn normalize_connector_name(name: &str) -> String {
    let mut t: String = name.trim().to_ascii_lowercase();
    t.retain(|c| c.is_ascii_alphanumeric() || c == '-');
    if let Some(pos) = t.find("displayport") {
        t.replace_range(pos..pos + "displayport".len(), "dp");
    }
    // "hdmi-1"/"hdmi1" -> kernel "hdmi-a-1"; leaves real "hdmi-a-*" alone.
    if let Some(rest) = t.strip_prefix("hdmi") {
        let rest = rest.trim_start_matches('-');
        if !rest.is_empty() && rest.as_bytes()[0].is_ascii_digit() {
            t = format!("hdmi-a-{rest}");
        }
    }
    t
}

/// One enabled/disabled connector on a card, resolved to the CRTC pipe index
/// (the same number WAIT_VBLANK takes in the high bits of `request.type`) and
/// the DRM CRTC object ID (what the modern CRTC-sequence ioctls address).
#[derive(Clone, Debug)]
pub struct ConnectorInfo {
    /// Kernel-style name, e.g. `DP-1`.
    pub name: String,
    /// drmModeConnector.connection == DRM_MODE_CONNECTED.
    pub connected: bool,
    /// CRTC pipe index serving this connector right now, when it is active.
    pub pipe: Option<u32>,
    /// The DRM object ID of the CRTC serving this connector (the id passed
    /// to `CRTC_GET_SEQUENCE`/`CRTC_QUEUE_SEQUENCE`), when it is active.
    pub crtc_id: Option<u32>,
}

/// Result of a connector scan: the list, the CRTC id list (pipe-index order,
/// for mapping a forced pipe back to its id), plus why it might be empty.
#[derive(Default)]
struct ConnectorScan {
    infos: Vec<ConnectorInfo>,
    /// Every CRTC object id on the card, in pipe-index order (index 0 = pipe
    /// 0 ...).  Populated whenever GETRESOURCES succeeds - even if the
    /// per-connector lookups below are denied - so a forced pipe can still be
    /// resolved to its CRTC id for the modern sequence ioctls.
    crtc_ids: Vec<u32>,
    /// `Some(errno text + stage)` when GETRESOURCES itself failed; empty
    /// otherwise (a successful query with zero connectors is reported via
    /// an empty list and no error).
    error: Option<String>,
}

/// Lists every connector on the open DRM fd with its current CRTC pipe.
/// Fails soft: returns an empty list plus a stage-tagged errno string instead
/// of panicking (the stage matters: nvidia-drm faults GETRESOURCES for
/// non-master clients while Mesa answers it happily).
unsafe fn enumerate_connectors(fd: c_int) -> ConnectorScan {
    let mut out = ConnectorScan::default();

    // Pass 1: counts. Pass 2: fill the id arrays (standard two-call pattern).
    let mut res: DrmModeCardRes = std::mem::zeroed();
    if ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) != 0 {
        out.error = Some(format!(
            "GETRESOURCES (pass 1): {}",
            std::io::Error::last_os_error()
        ));
        return out;
    }
    let mut crtcs: Vec<u32> = vec![0; res.count_crtcs as usize];
    let mut conns: Vec<u32> = vec![0; res.count_connectors as usize];
    // Pass 2 inherits every count the kernel wrote back in pass 1 - including
    // count_fbs/count_encoders for the lists we deliberately skip (connector
    // lookups give us the bound encoder directly, and this file owns no FBs).
    // The kernel copies any list whose user count covers the real one, so a
    // stale count paired with a NULL pointer means copy_to_user(0x0) ->
    // EFAULT. Zero both unused lists explicitly before re-calling.
    res.framebuffer_id_ptr = 0;
    res.count_fbs = 0;
    res.encoder_id_ptr = 0;
    res.count_encoders = 0;
    res.crtc_id_ptr = crtcs.as_mut_ptr() as u64;
    res.connector_id_ptr = conns.as_mut_ptr() as u64;
    if ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) != 0 {
        out.error = Some(format!(
            "GETRESOURCES (pass 2): {}",
            std::io::Error::last_os_error()
        ));
        return out;
    }
    crtcs.truncate(res.count_crtcs as usize);
    conns.truncate(res.count_connectors as usize);
    // Keep the CRTC id list (pipe-index order) even if every per-connector
    // lookup below is denied - a forced pipe still needs its id for the
    // modern CRTC-sequence ioctls.
    out.crtc_ids = crtcs.clone();

    for &id in &conns {
        // One call suffices: the kernel reports the currently-bound encoder
        // in `encoder_id` directly; with all counts left at 0 it copies no
        // arrays at all.
        let mut c: DrmModeGetConnector = std::mem::zeroed();
        c.connector_id = id;
        if ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &mut c) != 0 {
            continue;
        }

        let mut pipe = None;
        let mut crtc_id = None;
        // A connected, actively-scanned-out connector has its encoder bound;
        // GETENCODER yields the serving CRTC, whose index in the resources'
        // crtc list IS the WAIT_VBLANK pipe number.
        if c.encoder_id != 0 && c.connection == DRM_MODE_CONNECTED {
            let mut e: DrmModeGetEncoder = std::mem::zeroed();
            e.encoder_id = c.encoder_id;
            if ioctl(fd, DRM_IOCTL_MODE_GETENCODER, &mut e) == 0 && e.crtc_id != 0 {
                pipe = crtcs.iter().position(|&x| x == e.crtc_id).map(|p| p as u32);
                crtc_id = Some(e.crtc_id);
            }
        }

        out.infos.push(ConnectorInfo {
            name: format!("{}-{}", connector_type_name(c.connector_type), c.connector_type_id),
            connected: c.connection == DRM_MODE_CONNECTED,
            pipe,
            crtc_id,
        });
    }
    out
}

/// Resolves the DRM CRTC object IDs on `fd`, in pipe-index order, using only
/// `DRM_IOCTL_MODE_GETRESOURCES` (the standard two-call pattern).  Unlike
/// [`enumerate_connectors`] this touches no per-connector ioctls, so it works
/// even where GETCONNECTOR/GETENCODER are denied.  Returns an empty list when
/// GETRESOURCES itself is denied (e.g. nvidia-drm for a non-master client),
/// in which case no CRTC id can be resolved and the caller falls back to the
/// legacy pipe-index `WAIT_VBLANK` ioctl.
fn resolve_crtc_ids(fd: c_int) -> Vec<u32> {
    let mut res: DrmModeCardRes = unsafe { std::mem::zeroed() };
    if unsafe { ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) } != 0 {
        return Vec::new();
    }
    if res.count_crtcs == 0 {
        return Vec::new();
    }
    let mut crtcs: Vec<u32> = vec![0; res.count_crtcs as usize];
    res.framebuffer_id_ptr = 0;
    res.count_fbs = 0;
    res.encoder_id_ptr = 0;
    res.count_encoders = 0;
    res.crtc_id_ptr = crtcs.as_mut_ptr() as u64;
    if unsafe { ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) } != 0 {
        return Vec::new();
    }
    crtcs.truncate(res.count_crtcs as usize);
    crtcs
}

/// Counts ACTIVE heads via sysfs (`/sys/class/drm/cardN-*/enabled`), readable
/// without any DRM permissions at all. Used to tell "one display - any valid
/// pipe is THE pipe" apart from "several displays and no way to name them".
/// Returns 0 when sysfs says nothing useful.
fn sysfs_active_heads(card_dev_path: &str) -> usize {
    let base = match std::path::Path::new(card_dev_path).file_name().and_then(|s| s.to_str()) {
        Some(b) => b.to_string(),
        None => return 0,
    };
    let dir = match std::fs::read_dir("/sys/class/drm") {
        Ok(d) => d,
        Err(_) => return 0,
    };
    let mut n = 0;
    for entry in dir.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(connector) = name.strip_prefix(&base).and_then(|r| r.strip_prefix('-')) else {
            continue;
        };
        if connector.is_empty() {
            continue;
        }
        if std::fs::read_to_string(entry.path().join("enabled"))
            .map(|s| s.trim() == "enabled")
            .unwrap_or(false)
        {
            n += 1;
        }
    }
    n
}

/// True when the `nvidia_drm` kernel module is loaded (checked via sysfs, no
/// DRM permissions needed).  Used to tailor the vblank-unsupported hint to
/// nvidia-drm, whose vblank is gated behind the `vblank=1` option on the
/// 600/610+ branch.  (On a 595/390 kernel the `vblank` parameter file does
/// not exist, so the diagnostics below naturally stop short of suggesting a
/// fix that would not apply.)
fn nvidia_drm_present() -> bool {
    std::path::Path::new("/sys/module/nvidia_drm").exists()
}

unsafe extern "C" {
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    /// `poll(2)` for the DRM fd: waits until a queued CRTC-sequence event is
    /// readable (`POLLIN`).  Declared here rather than pulled from libc so
    /// this module stays dependency-free for the raw syscall surface.
    fn poll(fds: *mut PollFd, nfds: c_ulong, timeout: c_int) -> c_int;
    /// `read(2)`; pulls a `drm_event_crtc_sequence` off the fd.  Declared
    /// with a `*mut c_void` buffer to match the crate-root `read_tty` binding
    /// of the same symbol (avoiding a clashing-extern-declaration warning).
    fn read(fd: c_int, buf: *mut std::ffi::c_void, count: usize) -> isize;
}

/// `struct pollfd` for the hand-rolled `poll(2)` above.
#[repr(C)]
struct PollFd {
    fd: c_int,
    events: c_short,
    revents: c_short,
}

const POLLIN: c_short = 0x0001;

/// `errno 95` == EOPNOTSUPP/ENOTSUP ("Operation not supported").  A KMS
/// driver returns this from *every* vblank ioctl - legacy WAIT_VBLANK AND the
/// modern CRTC-sequence ones - when it never initialized its vblank
/// infrastructure.  nvidia-drm does exactly that unless the `vblank=1`
/// module option is set (available on 600/610+ only), so we use this code to
/// give a targeted hint (see [`nvidia_drm_present`]).
const ENOTSUP: i32 = 95;

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

/// Converts a CLOCK_MONOTONIC nanosecond timestamp (the modern CRTC-sequence
/// ioctls' clock) into host-epoch microseconds relative to the `offset` the
/// module sampled at open time.  Split out so it is unit-testable without a
/// live `DrmVblank`.
fn mono_to_host_us(mono_ns: u64, offset_us: u64) -> u64 {
    (mono_ns / 1000).saturating_sub(offset_us)
}

/// Decides, from one poll sample, whether the next vblank has elapsed.
///
/// Given the CRTC sequence observed at the previous sample and a fresh sample
/// `(sequence, timestamp_us)`, returns `Some(timestamp_us)` when the sequence
/// counter has advanced (a vblank has occurred -- the next one we were waiting
/// on), else `None` (same vblank still current, keep polling).  The `!=`
/// comparison also detects counter wraparound, since any change is a new
/// vblank.  Extracted as a pure function so the universal (non-master,
/// all-vendor) `wait_crtc_sequence_poll` wait has a unit test.
fn next_vblank_on_advance(prev_seq: u64, sample: (u64, u64)) -> Option<u64> {
    let (seq, ts) = sample;
    if seq != prev_seq {
        Some(ts)
    } else {
        None
    }
}

/// Phase-preserving correction for re-anchoring a prediction on a freshly
/// confirmed vblank.  `current` is the dead-reckoned prediction; `desired` is
/// the naive re-anchor (`confirmed + period`).  Returns the delta to add,
/// normalized into +/-half `period`.
///
/// PRECONDITION: `current` must already sit on the slot AFTER the boundary
/// `confirmed` is on -- the caller must have advanced its dead-reckoned
/// prediction by one `period` first (as [`DrmVblank::frame_end`] does before
/// folding).  Without that advance a healthy schedule's one-slot progression
/// (`current == confirmed`, `desired == confirmed + period`) is folded to a
/// ZERO step and the target freezes on the just-confirmed vblank -- every eye
/// then fires late.  A whole-period-ahead `desired` is therefore a symptom of
/// a MISSING advance in the caller, not of a vblank report error.
///
/// A naive `next_present = confirmed + period` is wrong because the confirmed
/// vblank is reported ±a WHOLE period relative to the target slot depending on
/// kernel/driver timing; applying it directly would sometimes step the
/// prediction one display slot off, firing that frame's eye too late (shutter
/// window missed -> glasses go dark -> a flicker/inversion once per resync).
/// Normalizing into +/-half period corrects only sub-slot drift and never moves
/// across a whole slot boundary, so the eye-to-slot phase is preserved.
pub(crate) fn phase_preserving_step(current: u64, desired: u64, period: u64) -> i64 {
    let mut step = desired as i64 - current as i64;
    let half = (period / 2) as i64;
    if step > half {
        step -= period as i64;
    } else if step < -half {
        step += period as i64;
    }
    step
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
    /// DRM object ID of the CRTC this anchor is bound to; the id passed to
    /// the modern CRTC-sequence ioctls.  Always set when `modern` is true.
    crtc_id: u32,
    /// True when the anchor uses the modern `CRTC_GET/QUEUE_SEQUENCE`
    /// ioctls; false when it had to fall back to the legacy
    /// `DRM_IOCTL_WAIT_VBLANK` (only when no CRTC id could be resolved,
    /// e.g. nvidia-drm denying enumeration to a non-master client).
    modern: bool,
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
    /// Set once an ioctl fails; the frame loop then fires eyes on presents
    /// directly (best-effort, without anchor timing).
    broken: bool,
    /// Kernel connector name this pipe was resolved from (`DP-1`), when the
    /// anchor was bound by output name; `None` for blind pipe probing.
    connector: Option<String>,
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
        Self::open_preferring(None, None)
    }

    /// Like [`open`], but when `pref_connector` names an output (e.g. the
    /// wl_output name of the monitor showing our window - it matches the
    /// kernel connector name on wlroots/KWin/Hyprland compositors), the anchor
    /// is bound to the CRTC that actually scans out THAT connector.
    ///
    /// On a multi-head GPU every CRTC free-runs with an arbitrary fixed phase
    /// offset to its siblings, so pacing shutter packets off the wrong head's
    /// vblank grid leaves a constant 0..1-frame timing error - sometimes small
    /// enough to look fine, sometimes mid-scanout (the top/bottom colour split).
    ///
    /// `force_pipe` (runtime 'o' key) bypasses naming and probes exactly that
    /// pipe index: the escape hatch for drivers that deny connector
    /// enumeration to non-master clients (nvidia-drm), where name binding is
    /// impossible and only the user's eyes can pick the right grid.  The
    /// forced pipe is still mapped to its CRTC object id via GETRESOURCES so
    /// the modern CRTC-sequence ioctls are used; only if even GETRESOURCES is
    /// denied (so no CRTC id exists) does the anchor fall back to the legacy
    /// pipe-indexing `WAIT_VBLANK` for that pipe.
    ///
    /// Falls back to the first-usable-pipe scan if the connector cannot be
    /// found or its CRTC does not run at ~120 Hz (loudly logged either way).
    pub fn open_preferring(
        pref_connector: Option<&str>,
        force_pipe: Option<u32>,
    ) -> Option<DrmVblank> {
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
            // Resolve connectors -> pipes so probing follows real displays and
            // we can bind to the one the window is on. Read-only ioctls, no
            // master needed on Mesa - but nvidia-drm denies them to plain
            // clients; that lands in `scan.error` and we degrade gracefully.
            let scan = if force_pipe.is_some() {
                ConnectorScan::default()
            } else {
                unsafe { enumerate_connectors(file.as_raw_fd()) }
            };
            let infos = &scan.infos;
            if !infos.is_empty() {
                let summary = infos
                    .iter()
                    .map(|i| {
                        format!(
                            "{}({}{})",
                            i.name,
                            i.pipe
                                .map(|p| format!("pipe{p}"))
                                .unwrap_or_else(|| "no-crtc".into()),
                            if i.connected { "" } else { ", disconnected" }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                eprintln!("nvstusb: {path} outputs: {summary}");
            }

            // Probe order: explicit pipe override, then preferred connector,
            // then every other connected+active connector, then - only if
            // enumeration failed entirely - the old blind 0..8 pipe scan.
            // Probing live CRTCs first avoids WAIT_VBLANK stalls on disabled
            // heads.
            let pref_norm = pref_connector.map(normalize_connector_name);
            let mut probe_order: Vec<(u32, String)> = Vec::new();
            if let Some(p) = force_pipe {
                probe_order.push((p, format!("pipe{p}")));
            } else {
                if let Some(want) = &pref_norm {
                    match infos.iter().find(|i| {
                        i.connected && i.pipe.is_some() && normalize_connector_name(&i.name) == *want
                    }) {
                        Some(i) => probe_order.push((i.pipe.unwrap(), i.name.clone())),
                        None => {
                            if scan.error.is_some() || infos.is_empty() {
                                // Distinguish "driver refused the query" from
                                // "output genuinely absent" so the user is not
                                // told their display does not exist.
                                eprintln!(
                                    "nvstusb: {path}: connector enumeration unavailable ({}) - \
                                     cannot bind anchor by output name",
                                    scan.error
                                        .as_deref()
                                        .unwrap_or("no connectors reported"),
                                );
                                let heads = sysfs_active_heads(&path);
                                match heads {
                                    1 => eprintln!(
                                        "nvstusb: 1 active head via sysfs - any valid pipe IS \
                                         that head; anchor should be correct"
                                    ),
                                    0 => eprintln!(
                                        "nvstusb: sysfs lists no enabled head on this card - \
                                         assuming a single head"
                                    ),
                                    _ => eprintln!(
                                        "nvstusb: NOTE {heads} active heads and no way to tell \
                                         them apart here: press 'o' in the demo to cycle the \
                                         anchor pipe until the blue/red scene separates cleanly"
                                    ),
                                }
                            } else {
                                eprintln!(
                                    "nvstusb: requested output {pref_connector:?} not found on {path}; \
                                     using the first active display"
                                );
                            }
                        }
                    }
                }
                for i in infos.iter() {
                    if i.connected
                        && i.pipe.is_some()
                        && !probe_order.iter().any(|(p, _)| Some(*p) == i.pipe)
                    {
                        probe_order.push((i.pipe.unwrap(), i.name.clone()));
                    }
                }
            }
            if probe_order.is_empty() {
                probe_order.extend((0..8u32).map(|p| (p, format!("pipe{p}"))));
            }

            // Probe each candidate CRTC; the active one advances the master
            // counter with a stable period.
            let mut pipe_diag: Vec<String> = Vec::new();
            for (pipe, label) in probe_order {
                let fd = match file.try_clone() {
                    Ok(fd) => fd,
                    Err(e) => {
                        eprintln!("nvstusb: probe {path}/{label}: clone fd: {e}");
                        reasons.push(format!("{path}/{label}: {e}"));
                        continue;
                    }
                };
                // Resolve the CRTC object ID for this pipe: prefer the
                // connector-bound id, else the GETRESOURCES crtc list in
                // pipe-index order (covers the forced/blind pipe scan, where
                // we may not have run full connector enumeration).  `None`
                // when enumeration is denied outright (nvidia non-master);
                // `new` then falls back to the legacy WAIT_VBLANK ioctl.
                let crtc_id = infos
                    .iter()
                    .find(|i| i.pipe == Some(pipe))
                    .and_then(|i| i.crtc_id)
                    .or_else(|| scan.crtc_ids.get(pipe as usize).copied())
                    .or_else(|| resolve_crtc_ids(file.as_raw_fd()).get(pipe as usize).copied());
                let mut d = match Self::new(
                    fd,
                    epoch_instant,
                    real_epoch_offset_us,
                    pipe,
                    crtc_id,
                ) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("nvstusb: probe {path}/{label}: open: {e}");
                        // A vblank ioctl answered EOPNOTSUPP - the driver
                        // never initialized vblank.  On nvidia-drm that is
                        // branch-dependent (see README): 600/610+ gate it
                        // behind the `vblank=1` module option; 595/390 have
                        // no option and cannot init it on a >= 4.19 kernel.
                        // No amount of pipe/connector retrying fixes it.
                        if e.raw_os_error() == Some(ENOTSUP) && nvidia_drm_present() {
                            // The `vblank` parameter file exists only on the
                            // 600/610+ branch; its absence means vblank can
                            // never be enabled here.
                            let param = std::fs::read_to_string(
                                "/sys/module/nvidia_drm/parameters/vblank",
                            )
                            .unwrap_or_default();
                            if param.is_empty() {
                                eprintln!(
                                    "nvstusb:   this nvidia-drm branch (595/390-class) has no \
                                     `vblank` module option, so DRM vblank cannot be enabled on \
                                     a >= 4.19 kernel; use the swap-return anchor or upgrade to \
                                     600/610+ (see README)"
                                );
                            } else {
                                let on = param.trim() == "Y" || param.trim() == "1";
                                eprintln!(
                                    "nvstusb:   nvidia-drm vblank is DISABLED - every vblank \
                                     ioctl returns EOPNOTSUPP.  Enable it (see README):\n\
                                     \x20 sudo cp 98-nvidia-drm.conf /etc/modprobe.d/\n\
                                     \x20 sudo modprobe -r nvidia_drm && sudo modprobe nvidia_drm\n\
                                     \x20 then verify: cat \
                                     /sys/module/nvidia_drm/parameters/vblank  # should print Y"
                                );
                                if !on {
                                    eprintln!(
                                        "nvstusb:   (currently: /sys/module/nvidia_drm/parameters/\
                                         vblank = {}{} - the option must be the bare name `vblank`, \
                                         not `nvidia_drm_vblank`)",
                                        param.trim(),
                                        "",
                                    );
                                }
                            }
                        }
                        reasons.push(format!("{path}/{label}: {e}"));
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
                        // Loud: a candidate that fails its waits while OTHER
                        // candidates succeed would otherwise vanish from the
                        // log entirely (reasons only print on total failure).
                        eprintln!(
                            "nvstusb: probe {path}/{label}: vblank wait failed \
                             ({errno})",
                            errno = std::io::Error::last_os_error()
                        );
                        reasons.push(format!("{path}/{label}: vblank wait failed"));
                        continue;
                    }
                }
                eprintln!("nvstusb: {path}/{} vblank period {} us", label, d.period_us);
                if (PERIOD_MIN_US..=PERIOD_MAX_US).contains(&d.period_us) {
                    d.connector = Some(label);
                    chosen = Some(d);
                    break;
                }
                pipe_diag.push(format!("{label}: {} us", d.period_us));
            }
            let d = match chosen {
                Some(d) => d,
                None => {
                    reasons.push(format!(
                        "{path}: no usable ~120 Hz display ({})",
                        if pipe_diag.is_empty() {
                            "no probes ran".to_string()
                        } else {
                            pipe_diag.join("; ")
                        }
                    ));
                    continue;
                }
            };
            if force_pipe.is_none()
                && !infos.is_empty()
                && pref_norm.is_some()
                && d.connector.as_deref().map(normalize_connector_name).as_deref()
                    != pref_norm.as_deref()
            {
                // The requested output was skipped because its CRTC did not
                // measure as ~120 Hz; say so loudly - sync will be phase-shifted.
                eprintln!(
                    "nvstusb: NOTE anchor is NOT on the requested output \
                     (its period was out of range); sync may be de-phased"
                );
            }
            if let Ok(rs) = std::env::var("NVSTUSB_DRM_RESYNC") {
                if let Ok(v) = rs.parse::<u32>() {
                    let _ = v; // resync handled by the paced anchor
                }
            }
            eprintln!(
                "nvstusb: DRM vblank anchor on {path}/{} (pipe {} / crtc {} , period {} us, {})",
                d.connector.as_deref().unwrap_or("?"),
                d.pipe,
                d.crtc_id,
                d.period_us,
                if d.modern {
                    "CRTC-sequence"
                } else {
                    "legacy WAIT_VBLANK"
                }
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
        crtc_id: Option<u32>,
    ) -> Result<DrmVblank, std::io::Error> {
        let fd = OwnedFd::from(file);

        // Mechanism selection.  We have two ways to read the vblank clock:
        //
        //   * Legacy `DRM_IOCTL_WAIT_VBLANK` — blocking, returns its vblank
        //     timestamp DIRECTLY in the ioctl reply (no event delivery), and
        //     works for NON-MASTER clients.  This is exactly what a windowed /
        //     composited desktop is: the anchor reads the display engine's
        //     vblank clock without holding DRM master.
        //
        //   * Modern `CRTC_GET_SEQUENCE` / `CRTC_QUEUE_SEQUENCE` — core-table
        //     ioctls served by every KMS driver, but `CRTC_QUEUE_SEQUENCE`
        //     delivers its result as a DRM frame EVENT, which the kernel only
        //     queues for the DRM MASTER.  A non-master (composited) client
        //     therefore never receives the event, so the modern blocking wait
        //     cannot work there.
        //
        // So we PREFER the legacy ioctl (it is what made the anchor work on
        // composited desktops) and fall back to the modern path ONLY when
        // WAIT_VBLANK is unavailable for the channel — some drivers
        // (nvidia-drm) never install WAIT_VBLANK and answer it with
        // EOPNOTSUPP, leaving the modern core-table ioctls as the sole
        // option.  The modern path needs a CRTC object id.
        let (modern, crtc_id) = if let Some(id) = crtc_id {
            let legacy_ok = {
                let mut v = DrmWaitVblank {
                    request: DrmWaitVblankRequest {
                        r#type: (DRM_VBLANK_RELATIVE
                            | DRM_VBLANK_NEXTONMISS
                            | vblank_pipe_bits(pipe)) as c_uint,
                        sequence: 1,
                        signal: 0,
                    },
                };
                let r = unsafe { ioctl(fd.as_raw_fd(), DRM_IOCTL_WAIT_VBLANK, &mut v) };
                // Safe: a successful WAIT_VBLANK fills the reply union member;
                // sequence 0 means the counter is not ticking, treat as legacy
                // failure so we still try the modern path when we have an id.
                r == 0 && unsafe { v.reply }.sequence != 0
            };
            if legacy_ok {
                (false, 0)
            } else {
                // WAIT_VBLANK is unavailable (e.g. nvidia-drm EOPNOTSUPP):
                // try the modern CRTC-sequence path for this CRTC id.  If it
                // too fails, return the LEGACY error (its errno carries the
                // EOPNOTSUPP that the caller's nvidia hint keys off).
                let mut v = DrmWaitVblank {
                    request: DrmWaitVblankRequest {
                        r#type: (DRM_VBLANK_RELATIVE
                            | DRM_VBLANK_NEXTONMISS
                            | vblank_pipe_bits(pipe)) as c_uint,
                        sequence: 1,
                        signal: 0,
                    },
                };
                let legacy_err = {
                    let r = unsafe { ioctl(fd.as_raw_fd(), DRM_IOCTL_WAIT_VBLANK, &mut v) };
                    if r != 0 {
                        std::io::Error::last_os_error()
                    } else {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "vblank sequence 0",
                        )
                    }
                };
                let mut g = DrmCrtcGetSequence {
                    crtc_id: id,
                    active: 0,
                    sequence: 0,
                    sequence_ns: 0,
                };
                let r = unsafe { ioctl(fd.as_raw_fd(), DRM_IOCTL_CRTC_GET_SEQUENCE, &mut g) };
                if r != 0 || g.sequence == 0 {
                    return Err(legacy_err);
                }
                (true, id)
            }
        } else {
            // No CRTC id resolvable (e.g. nvidia-drm denying enumeration to a
            // non-master client): only the legacy pipe-indexing WAIT_VBLANK
            // can be used for this pipe (the original pre-CRTC-sequence path).
            let mut v = DrmWaitVblank {
                request: DrmWaitVblankRequest {
                    r#type: (DRM_VBLANK_RELATIVE
                        | DRM_VBLANK_NEXTONMISS
                        | vblank_pipe_bits(pipe)) as c_uint,
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
            (false, 0)
        };
        Ok(DrmVblank {
            fd,
            pipe,
            crtc_id,
            modern,
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
            connector: None,
            resync_count: 0,
            resync_err_total: 0,
            resync_err_max_abs: 0,
        })
    }

    /// Blocking wait for the next vblank on the configured pipe; returns its
    /// timestamp in host-epoch microseconds.
    pub fn wait_vblank(&mut self) -> Option<u64> {
        if self.modern {
            self.wait_crtc_sequence_poll()
        } else {
            self.wait_vblank_flags(DRM_VBLANK_RELATIVE | DRM_VBLANK_NEXTONMISS)
        }
    }

    /// Same as [`wait_vblank`] but without `NEXTONMISS`, so it always blocks
    /// until the next vblank actually occurs.  Used for period measurement,
    /// where `NEXTONMISS` can return the same (current) vblank timestamp on
    /// back-to-back calls.
    pub fn wait_vblank_blocking(&mut self) -> Option<u64> {
        if self.modern {
            self.wait_crtc_sequence_poll()
        } else {
            self.wait_vblank_flags(DRM_VBLANK_RELATIVE)
        }
    }

    /// Non-blocking query: returns the host-epoch microseconds of the most
    /// recent vblank on this pipe.  On the modern path a single
    /// `CRTC_GET_SEQUENCE` returns the CLOCK_MONOTONIC timestamp of the last
    /// vblank directly.  On the legacy path an absolute WAIT_VBLANK with
    /// target sequence 0 completes immediately (the counter has long passed
    /// it) and reports the last vblank's kernel timestamp - the standard way
    /// to read MSC without waiting.  Transient failures return `None` without
    /// poisoning [`Self::broken`]; callers fall back for that frame.
    pub fn query_vblank(&mut self) -> Option<u64> {
        if self.modern {
            let mut g = DrmCrtcGetSequence {
                crtc_id: self.crtc_id,
                active: 0,
                sequence: 0,
                sequence_ns: 0,
            };
            let r = unsafe { ioctl(self.fd.as_raw_fd(), DRM_IOCTL_CRTC_GET_SEQUENCE, &mut g) };
            if r != 0 {
                return None;
            }
            let ns = g.sequence_ns;
            if ns == 0 {
                return None;
            }
            Some(self.mono_ns_to_host_us(ns as u64))
        } else {
            let mut v = DrmWaitVblank {
                request: DrmWaitVblankRequest {
                    r#type: vblank_pipe_bits(self.pipe) as c_uint,
                    sequence: 0,
                    signal: 0,
                },
            };
            let r = unsafe { ioctl(self.fd.as_raw_fd(), DRM_IOCTL_WAIT_VBLANK, &mut v) };
            if r != 0 {
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
    }

    /// Non-master-safe blocking wait for the next vblank on the configured
    /// pipe, returning its timestamp in host-epoch microseconds.  This is the
    /// one wait that works in BOTH windowed modes (the demo AND the host) on
    /// EVERY vendor:
    ///
    ///   * the legacy `WAIT_VBLANK` blocking wait is only installed by
    ///     amdgpu/i915 -- nvidia-drm answers it `EOPNOTSUPP`;
    ///   * `CRTC_QUEUE_SEQUENCE` (the other blocking mechanism) delivers its
    ///     event ONLY to the DRM master, which a windowed/host client is not.
    ///
    /// `CRTC_GET_SEQUENCE` is a core-table query served to ANY client on every
    /// KMS driver (the anchor only needs a CRTC object id), so polling it until
    /// the sequence counter advances is the universal vblank listen.  The
    /// returned timestamp is the vblank's own kernel `sequence_ns`
    /// (CLOCK_MONOTONIC, via `mono_ns_to_host_us`), so it is hardware-accurate
    /// regardless of poll granularity -- polling only sets how soon AFTER the
    /// vblank we observe it, never the value we return.
    fn wait_crtc_sequence_poll(&mut self) -> Option<u64> {
        let fd = self.fd.as_raw_fd();
        // Query the current sequence + last-vblank timestamp.
        let get = |this: &mut Self| -> Option<(u64, u64)> {
            let mut g = DrmCrtcGetSequence {
                crtc_id: this.crtc_id,
                active: 0,
                sequence: 0,
                sequence_ns: 0,
            };
            if unsafe { ioctl(fd, DRM_IOCTL_CRTC_GET_SEQUENCE, &mut g) } != 0 {
                return None;
            }
            let ns = g.sequence_ns;
            if ns == 0 {
                return None;
            }
            Some((g.sequence, this.mono_ns_to_host_us(ns as u64)))
        };
        let (last_seq, _) = get(self)?;
        // Poll until the counter advances (the NEXT vblank), with a timeout far
        // above any sane period so a dead/exhausted CRTC surfaces as `None`
        // (and `broken`) instead of hanging the emitter loop.
        let deadline = Instant::now() + Duration::from_millis(1000);
        loop {
            // Tiny sleep to avoid hammering the ioctl; the returned timestamp is
            // still the exact vblank time, so this does not add timing error.
            std::thread::sleep(Duration::from_micros(100));
            if let Some(sample) = get(self) {
                if let Some(ts) = next_vblank_on_advance(last_seq, sample) {
                    // Counter advanced: the NEXT vblank has occurred.  Its
                    // timestamp is hardware-accurate (`sequence_ns`), so this
                    // is the vblank we were waiting for.
                    return Some(ts);
                }
            }
            if Instant::now() >= deadline {
                self.broken = true;
                return None;
            }
        }
    }

    /// Converts a CLOCK_MONOTONIC nanosecond timestamp (as returned by the
    /// modern CRTC-sequence ioctls) into host-epoch microseconds.
    fn mono_ns_to_host_us(&self, ns: u64) -> u64 {
        mono_to_host_us(ns, self.real_epoch_offset_us)
    }

    fn wait_vblank_flags(&mut self, flags: c_int) -> Option<u64> {
        let mut v = DrmWaitVblank {
            request: DrmWaitVblankRequest {
                r#type: (flags | vblank_pipe_bits(self.pipe)) as c_uint,
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
    ///
    /// Before computing the deadline the predicted present is SNAPPED onto
    /// the vblank grid: it is advanced in whole periods until the boundary
    /// is still reachable (alarm delay + lead + margin in the future).
    /// Without this, a pipeline that lands every flip N periods after the
    /// prediction - the driver throttling inside `eglSwapBuffers` followed
    /// by our own `MODE_PAGE_FLIP` - fed `flip + period` back as the next
    /// prediction via `frame_end`, advancing it three periods per frame and
    /// trapping the whole loop at one frame per three vblanks (~40 fps at
    /// 120 Hz) with a full re-anchor every frame.  In a healthy steady
    /// state the loop runs early and the snap below never triggers.
    pub fn frame_start(&mut self, lead_us: u32) -> Option<Instant> {
        if self.broken || self.period_us == 0 {
            return None;
        }
        const SNAP_MARGIN_US: u64 = 250;
        let min_lead = ALARM_DELAY_US + lead_us as u64 + SNAP_MARGIN_US;
        let now = self.epoch_instant.elapsed().as_micros() as u64;
        while self.next_present_us.saturating_sub(min_lead) <= now {
            self.next_present_us = self.next_present_us.saturating_add(self.period_us);
        }
        let target = self.next_present_us;
        let deadline_epoch = target.saturating_sub(ALARM_DELAY_US + lead_us as u64);
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
    /// `precise_us` should be a DRM page-flip event's own hardware
    /// timestamp converted via [`host_us_from_mono`], when available - it has
    /// no scheduler/wakeup jitter, unlike a post-syscall `Instant::now()`
    /// sample. Pass `None` to fall back to [`host_epoch_us`].
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
        // Frames ACTUALLY elapsed since the previous re-anchor - the period
        // re-measurement below divides by this.  It equals `resync_every` in
        // steady state but is smaller after a forced/early resync (mode-set,
        // `force_resync`), where dividing by the nominal interval would
        // underestimate the period and skew every later prediction.
        let frames_elapsed = self.frames_since_resync.max(1);
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
        self.last_resync_frames = frames_elapsed;
        // Re-anchor the prediction on the just-confirmed vblank, but PHASE-
        // PRESERVING: nudge to the slot nearest the per-frame dead-reckoned
        // prediction and only correct the sub-slot residual.  A naive
        // `next_present_us = t + period` is wrong: `t` (the last COMPLETED
        // vblank) is reported ±a whole period relative to the target slot
        // depending on kernel/driver timing at this instant, so a full-period
        // re-anchor sometimes jumps the prediction one display slot off --
        // firing that frame's eye too late (shutter window missed, glasses go
        // dark) and flickering once each resync (~every 1.25 s).  Normalize the
        // re-anchor delta into +/-half period so we correct only drift, never
        // step a whole slot (which is what a phase flip looks like).
        let desired = t.saturating_add(self.period_us);
        let step = phase_preserving_step(self.next_present_us, desired, self.period_us);
        self.next_present_us = (self.next_present_us as i64 + step).max(0) as u64;
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

    /// Human-readable identity of the anchored display, e.g. `DP-1/pipe1`.
    /// `pipeN` alone when the anchor was opened blind (no connector match).
    pub fn label(&self) -> String {
        let conn = self.connector.as_deref().unwrap_or("?");
        format!("{conn}/pipe{}", self.pipe)
    }

    /// The kernel connector name this pipe was resolved from (e.g. `DP-1`),
    /// when a display match was possible.  Used for monitor identity (EDID).
    pub fn connector_name(&self) -> Option<&str> {
        self.connector.as_deref()
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
    /// Used after a mode-set or a head change, which can shift the vblank phase.
    pub fn force_resync(&mut self) {
        self.frames_since_resync = self.resync_every;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        next_vblank_on_advance, normalize_connector_name, phase_preserving_step, vblank_pipe_bits,
        DRM_VBLANK_HIGH_CRTC_MASK,
    };

    /// Pins the universal (non-master, every-vendor) vblank wait: the sequence
    /// counter advancing is what reports "the next vblank has occurred".  A
    /// sample at the same sequence means the vblank has not yet elapsed; any
    /// advance (including wraparound) yields the vblank's own timestamp.
    #[test]
    fn next_vblank_detected_only_on_sequence_advance() {
        // Same sequence -> not a new vblank yet, keep polling.
        assert_eq!(next_vblank_on_advance(42, (42, 1_000_000)), None);
        assert_eq!(next_vblank_on_advance(42, (42, 1_001_000)), None);
        // Counter advanced by one -> the next vblank elapsed; its timestamp is
        // returned verbatim (hardware-accurate, not the poll time).
        assert_eq!(next_vblank_on_advance(42, (43, 1_009_000)), Some(1_009_000));
        // Multi-slot advance (we slept through several vblanks) also counts.
        assert_eq!(next_vblank_on_advance(42, (48, 1_050_000)), Some(1_050_000));
        // Wraparound: u64 counter rolls 0 before 1; a change is still an
        // advance, so the wait must not hang forever.
        assert_eq!(next_vblank_on_advance(u64::MAX, (0, 9_999_999)), Some(9_999_999));
    }

    /// Pins the periodic-flicker fix: the resync re-anchor MUST NOT step the
    /// prediction a whole display slot.  With a 120 Hz period (8333us), if the
    /// freshly confirmed vblank reports one slot behind the dead-reckoned
    /// prediction, the naive `confirmed + period` is a full period off -- a
    /// naive `next_present = current + step` (step = -8333) would jump a whole
    /// slot: that frame's eye fires late, misses the shutter window, glasses go
    /// dark -> the flicker once per resync.  `phase_preserving_step` clamps the
    /// correction into +/-half period so it only nudges sub-slot drift (0 here),
    /// never stepping a full slot.
    #[test]
    fn resync_reanchor_never_steps_a_whole_slot() {
        let period = 8333u64;
        // Dead-reckoned prediction points at the next slot (slot 5, ts 41665).
        let current = 41_665;
        // Confirmed vblank reported ONE slot behind (slot 3, ts 24999):
        // naive desired = 24999 + 8333 = 33332 = a full period behind current.
        let desired = 24_999 + period; // 33332, exactly `period` behind current
        // The proper correction preserves phase: it must be ~0, not -period.
        assert_eq!(phase_preserving_step(current, desired, period), 0);
        // Boundary: exactly one half-period off is still clamped (not stepped).
        let just_under_half = current.saturating_sub(period / 2 - 1);
        let _ = phase_preserving_step(current, just_under_half, period);
        // A genuine sub-slot residual (say a +3us real drift) IS corrected.
        assert_eq!(phase_preserving_step(current, current + 3, period), 3);
    }

    #[test]
    fn connector_names_normalize_across_spellings() {
        // Kernel name == compositor name (wlroots/KWin/Hyprland).
        assert_eq!(normalize_connector_name("DP-1"), normalize_connector_name("dp-1"));
        assert_eq!(normalize_connector_name("HDMI-A-1"), "hdmi-a-1");
        // GNOME-style spelling maps onto the kernel name.
        assert_eq!(normalize_connector_name("DisplayPort-1"), "dp-1");
        assert_eq!(normalize_connector_name("DisplayPort-2"), "dp-2");
        // Plain "HDMI-1" (some compositors) -> kernel "HDMI-A-1".
        assert_eq!(normalize_connector_name("HDMI-1"), "hdmi-a-1");
        assert_eq!(normalize_connector_name("eDP-1"), "edp-1");
        // Distinct outputs must stay distinct.
        assert_ne!(normalize_connector_name("DP-1"), normalize_connector_name("DP-2"));
    }

    /// Pins the WAIT_VBLANK pipe encoding against the kernel's decoder
    /// (drm_vblank.c): `(type & 0x3e) >> 1`, no other bits set. Guards
    /// against a regression to the bogus `pipe << 24` encoding, which the
    /// kernel rejects with EINVAL for every pipe >= 1.
    #[test]
    fn vblank_pipe_encoding_matches_kernel_decoder() {
        for pipe in 0u32..8 {
            let bits = vblank_pipe_bits(pipe) as u32;
            if pipe == 0 {
                assert_eq!(bits, 0, "pipe 0 must encode to no bits");
            }
            // Only RELATIVE-compatible low bits (HIGH_CRTC field) may be set.
            assert_eq!(bits & !DRM_VBLANK_HIGH_CRTC_MASK, 0);
            // Round-trip through the kernel's decode formula.
            let decoded = (bits & DRM_VBLANK_HIGH_CRTC_MASK) >> 1;
            assert_eq!(decoded, pipe, "pipe {pipe} must survive kernel decode");
        }
    }

    /// Pins the modern CRTC-sequence ioctl numbers and the `DRM_EVENT_*`
    /// type codes against the UAPI (drm.h) so the event parse never silently
    /// checks the wrong type.
    #[test]
    fn crtc_sequence_abi_matches_uapi() {
        use super::{
            DRM_CRTC_SEQUENCE_RELATIVE, DRM_EVENT_CRTC_SEQUENCE, DRM_IOCTL_CRTC_GET_SEQUENCE,
            DRM_IOCTL_CRTC_QUEUE_SEQUENCE, DrmCrtcGetSequence, DrmCrtcQueueSequence,
            DrmEventCrtcSequence, size_of,
        };
        let _ = size_of::<DrmCrtcGetSequence>;
        assert_eq!(size_of::<DrmCrtcGetSequence>(), 24);
        assert_eq!(size_of::<DrmCrtcQueueSequence>(), 24);
        assert_eq!(size_of::<DrmEventCrtcSequence>(), 32);
        // _IOWR('d', 0x3b/0x3c) as computed by drm.h on x86_64.
        assert_eq!(DRM_IOCTL_CRTC_GET_SEQUENCE, 0xc018_643b);
        assert_eq!(DRM_IOCTL_CRTC_QUEUE_SEQUENCE, 0xc018_643c);
        // drm.h: DRM_CRTC_SEQUENCE_RELATIVE == 1, DRM_EVENT_CRTC_SEQUENCE == 3.
        assert_eq!(DRM_CRTC_SEQUENCE_RELATIVE, 0x0000_0001);
        assert_eq!(DRM_EVENT_CRTC_SEQUENCE, 0x03);
    }

    /// The modern ioctls report CLOCK_MONOTONIC nanoseconds; the module
    /// converts them to the same host-epoch microseconds the legacy path
    /// returns, by subtracting the same `real_epoch_offset_us` the whole
    /// module samples at open time.
    #[test]
    fn mono_ns_converts_to_host_epoch_us() {
        use super::mono_to_host_us;
        // 1234.567 ms monotonic with the epoch sampled at 1s exactly.
        assert_eq!(mono_to_host_us(1_234_567_000, 1_000_000), 234_567);
        assert_eq!(mono_to_host_us(1_500_000_000, 1_000_000), 500_000);
    }
}
