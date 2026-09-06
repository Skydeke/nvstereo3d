//! Shared-memory IPC between wiz3D's `Nvidia3DOutput.dll` (running under
//! Wine/Proton) and this host helper.
//!
//! The DLL maps the same backing file (host `/tmp/nvstusb.shm`, which Wine
//! exposes as `Z:\tmp\nvstusb.shm`) and pushes one eye-swap command per frame
//! into a lock-free single-producer/single-consumer ring.  This helper
//! consumes the ring and fires the eye packet at the USB emitter.
//!
//! The layout below is the *exact* binary contract shared with
//! `OutputMethods/Nvidia3DOutput/shm.h` in the wiz3D repo — keep them in sync.
//! All fields are fixed-width and pointer-free, so the DLL (Win32 or x64) and
//! this helper (x86_64) agree on offsets regardless of word size.

#![allow(dead_code)]

use memmap2::MmapMut;
use std::sync::atomic::{AtomicU32, Ordering};

/// `"NVSF"` in little-endian.
pub const SHM_MAGIC: u32 = 0x4E56_5346;
/// Version 1 = legacy 8-byte slots (eye only, no timestamp).
pub const SHM_VERSION: u32 = 1;
/// Version 2 = 16-byte slots with a producer-stamped submit time (`t_us`).
///
/// The DLL bumps the region from 1 to 2 when it starts stamping; this host
/// accepts both, switching scheduling modes on the header version (stamped
/// slots pin each eye to its own vblank boundary, legacy slots use the
/// FIFO+hold fallback -- see `host.rs`).  Must match `NVSTUSB_SHM_VERSION`
/// / `NVSTUSB_SHM_VERSION_STAMPED` in `shm.h`.
pub const SHM_VERSION_STAMPED: u32 = 2;
/// Ring capacity in slots (must be a power of two).
pub const RING_CAP: u32 = 1024;
/// Legacy slot stride (version 1).
pub const SLOT_SIZE: usize = 8;
/// Stamped slot stride (version 2; also `size_of::<RingSlot>()`).
pub const SLOT_SIZE_STAMPED: usize = 16;
/// Total file size (header + ring).
pub const SHM_FILE_SIZE: usize = 64 * 1024;

/// Header offset/size (must match `shm.h`).
pub const HDR_SIZE: usize = 64;
pub const OFF_MAGIC: usize = 0;
pub const OFF_VERSION: usize = 4;
pub const OFF_CAP: usize = 8;
pub const OFF_HEAD: usize = 12;
pub const OFF_TAIL: usize = 16;
pub const OFF_STATUS: usize = 20;
pub const OFF_SEQ: usize = 24;
pub const OFF_RATE_HZ: usize = 28;
pub const OFF_ALARM_DELAY_US: usize = 32;
pub const OFF_FLAGS: usize = 36;

/// Window-owner -> helper target display.  A fixed 12-byte, null-terminated
/// ASCII field in the header's spare bytes (52..=63) naming the DRM connector
/// (e.g. `DP-2`) the game window is actually on.  The window owner (the 3dv3d
/// demo, via winit's `current_monitor().name()`, or the wiz3D DLL via the
/// target monitor's EDID) writes it whenever the window changes heads; the
/// helper re-anchors its vblank clock to that connector.  Empty = no signal.
/// Must match `NVSTUSB_SHM_OFF_CONNECTOR` / `NVSTUSB_SHM_CONNECTOR_LEN` in
/// `shm.h`.
pub const OFF_CONNECTOR: usize = 52;
pub const CONNECTOR_LEN: usize = 12;

/// Emitter state reported by the helper (helper -> DLL).
pub const STATUS_NONE: u32 = 0;
pub const STATUS_OPENING: u32 = 1;
pub const STATUS_READY: u32 = 2;
pub const STATUS_ERROR: u32 = 3;
pub const STATUS_CLOSED: u32 = 4;

/// `flags` bits (both directions on the same field; set via `fetch_or`).
pub const FLAG_FIRMWARE_LOADED: u32 = 1 << 0;
pub const FLAG_EMITTER_PRESENT: u32 = 1 << 1;
/// Set by the producer (DLL) when it stamps `t_us` into version-2 slots.
/// A legacy producer leaves it clear and this host falls back to FIFO+hold.
/// Mirrors `NVSTUSB_SHM_FLAG_SLOT_TIME_US` in `shm.h`.
pub const FLAG_SLOT_TIME_US: u32 = 1 << 2;
/// Host -> DLL control bit: "swap the left/right eye.  Do NOT shift the
/// shutter phase (which makes one eye hold -> a longer dark/black period);
/// instead the DLL inverts which eye it renders AND reports to the ring, so
/// the glasses keep shuttering at their natural cadence.  The host sets this
/// when a polarity fix is required (e.g. the emitter button, or after a drop
/// in phase-pinned mode) and clears it to return to the game's native mapping.
/// Mirrors `NVSTUSB_SHM_FLAG_INVERT_EYES` in `shm.h`.
pub const FLAG_INVERT_EYES: u32 = 1 << 3;

/// Eye index carried in a ring slot (DLL -> helper).
pub const EYE_LEFT: u8 = 0;
pub const EYE_RIGHT: u8 = 1;

/// Backing file path on the host.  The DLL opens the same file through the
/// Wine `Z:` drive as `Z:\tmp\nvstusb.shm`.
pub const DEFAULT_SHM_PATH: &str = "/tmp/nvstusb.shm";

/// A single eye-swap command slot (version-2, 16-byte layout).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RingSlot {
    /// Producer sequence number.  Used for diagnostics and to disambiguate a
    /// slot that still holds a stale command from a freshly written one.
    pub seq: u32,
    /// `EYE_LEFT` or `EYE_RIGHT` (kept at offset 4 -- the same offset the
    /// legacy 8-byte layout used, so the strided reader can share the raw-byte
    /// read path for the eye byte).
    pub eye: u8,
    /// Reserved, zero.
    pub flags: u8,
    pub pad: [u8; 2],
    /// Producer-stamped submit time of this present, as host CLOCK_MONOTONIC
    /// microseconds (meaningful when the producer set [`FLAG_SLOT_TIME_US`]).
    /// The host pins the eye to the vblank boundary its present actually scans
    /// out on via this timestamp.
    pub t_us: u64,
}

/// One drained eye-swap: the eye plus (version-2 regions) the producer's
/// CLOCK_MONOTONIC submit timestamp.  `t_us` is `None` on legacy regions.
/// `seq` is the producer sequence number (used by the stamp diagnostics to
/// notice producer restarts / backlog gaps); it is 0 on legacy layouts where
/// the 8-byte slot carries no sequence.
#[derive(Clone, Copy, Debug)]
pub struct Swap {
    pub eye: u8,
    pub t_us: Option<u64>,
    pub seq: u32,
}

// Pin the wire layout: the stamped slot must be exactly the 16-byte stride and
// the legacy 8-byte layout must remain valid (slots are indexed by the header
// version, so both strides must fit the shared file).
const _: () = {
    assert!(std::mem::size_of::<RingSlot>() == SLOT_SIZE_STAMPED);
    assert!(std::mem::align_of::<RingSlot>() == 8);
    assert!(HDR_SIZE + RING_CAP as usize * SLOT_SIZE_STAMPED <= SHM_FILE_SIZE);
    assert!(HDR_SIZE + RING_CAP as usize * SLOT_SIZE <= SHM_FILE_SIZE);
};

/// Returns a shared reference to an atomic view of a raw u32 at a byte offset.
fn atom(base: *mut u8, off: usize) -> &'static AtomicU32 {
    // AtomicU32 has the same size/alignment as u32; the header slots are all
    // naturally aligned 4-byte fields, so this cast is sound.
    unsafe { &*(base.add(off) as *const AtomicU32) }
}

/// The mapped shared region.
pub struct Shm {
    pub mmap: MmapMut,
    base: *mut u8,
    pub head: &'static AtomicU32,
    pub tail: &'static AtomicU32,
    pub status: &'static AtomicU32,
    pub seq: &'static AtomicU32,
    pub rate_hz: &'static AtomicU32,
    pub alarm_delay_us: &'static AtomicU32,
    pub flags: &'static AtomicU32,
    pub slots: *mut RingSlot,
    /// Raw pointer to the 12-byte connector-name field (offset 52).  Written
    /// by the window owner, read by the helper; see `OFF_CONNECTOR`.
    connector: *mut u8,
    pub cap: u32,
}

impl Shm {
    /// Opens the backing file and maps it.  If the file is missing or too
    /// small, it is created/truncated to [`SHM_FILE_SIZE`].  The DLL may
    /// already have initialized it, in which case the existing header is
    /// preserved (see [`Shm::cap`]).
    pub fn open(path: &str) -> std::io::Result<Shm> {
        use std::fs::OpenOptions;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        let len = file.metadata()?.len() as usize;
        if len < SHM_FILE_SIZE {
            file.set_len(SHM_FILE_SIZE as u64)?;
        }
        // SAFETY: the file is opened read/write and stays alive in `Shm`.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        let base = mmap.as_ptr() as *mut u8;
        let mut shm = Shm {
            head: atom(base, OFF_HEAD),
            tail: atom(base, OFF_TAIL),
            status: atom(base, OFF_STATUS),
            seq: atom(base, OFF_SEQ),
            rate_hz: atom(base, OFF_RATE_HZ),
            alarm_delay_us: atom(base, OFF_ALARM_DELAY_US),
            flags: atom(base, OFF_FLAGS),
            slots: unsafe { base.add(HDR_SIZE) } as *mut RingSlot,
            connector: unsafe { base.add(OFF_CONNECTOR) },
            cap: 0,
            mmap,
            base,
        };
        shm.cap = shm.cap();
        Ok(shm)
    }

    /// Sets the magic/version/cap and (re)initializes head/tail.  Only called
    /// when this process owns a freshly created region.
    pub fn init_region(&mut self) {
        atom(self.base, OFF_MAGIC).store(SHM_MAGIC, Ordering::Relaxed);
        atom(self.base, OFF_VERSION).store(SHM_VERSION, Ordering::Relaxed);
        atom(self.base, OFF_CAP).store(RING_CAP, Ordering::Relaxed);
        self.head.store(0, Ordering::Release);
        self.tail.store(0, Ordering::Release);
        self.cap = RING_CAP;
    }

    /// Validates the header; returns the ring capacity (0 if invalid).  Both
    /// wire versions are accepted: v1 (legacy 8-byte slots), v2 (stamped
    /// 16-byte slots).
    pub fn cap(&self) -> u32 {
        let magic = atom(self.base, OFF_MAGIC).load(Ordering::Relaxed);
        let version = atom(self.base, OFF_VERSION).load(Ordering::Relaxed);
        let cap = atom(self.base, OFF_CAP).load(Ordering::Relaxed);
        if magic != SHM_MAGIC
            || (version != SHM_VERSION && version != SHM_VERSION_STAMPED)
            || cap == 0
            || !cap.is_power_of_two()
        {
            0
        } else {
            cap
        }
    }

    /// The header's wire version: [`SHM_VERSION`] (legacy) or
    /// [`SHM_VERSION_STAMPED`].
    pub fn version(&self) -> u32 {
        atom(self.base, OFF_VERSION).load(Ordering::Relaxed)
    }

    /// True when the producer is stamping submit times: the region is at
    /// version 2 AND the producer set [`FLAG_SLOT_TIME_US`].  Gates phase-pinned
    /// scheduling (a stamped region is unambiguous -- an 8-byte legacy slot has
    /// no timestamp field to read).
    pub fn stamped(&self) -> bool {
        self.cap != 0
            && self.version() == SHM_VERSION_STAMPED
            && self.flags() & FLAG_SLOT_TIME_US != 0
    }

    /// Reads the DLL's requested refresh rate in Hz (f32 bit pattern).
    pub fn rate_hz(&self) -> f32 {
        f32::from_bits(self.rate_hz.load(Ordering::Relaxed))
    }

    /// Reads the DLL's requested IR alarm delay (us); 0 = leave default.
    pub fn alarm_delay_us(&self) -> u32 {
        self.alarm_delay_us.load(Ordering::Relaxed)
    }

    /// Reports emitter state to the DLL.
    pub fn set_status(&self, status: u32) {
        self.status.store(status, Ordering::Relaxed);
    }

    /// Reports capability/state flags to the DLL.  ORs the bits in so a
    /// concurrent writer is never clobbered.
    pub fn set_flags(&self, flags: u32) {
        self.flags.fetch_or(flags, Ordering::Relaxed);
    }

    /// Current header flags (both directions).
    pub fn flags(&self) -> u32 {
        self.flags.load(Ordering::Relaxed)
    }

    /// Set the host -> DLL "swap eyes" control bit (see [`FLAG_INVERT_EYES`]).
    /// Forces the DLL to invert its rendered/reported eye; the glasses' shutter
    /// phase is untouched.
    pub fn set_invert_eyes(&self) {
        self.flags.fetch_or(FLAG_INVERT_EYES, Ordering::Relaxed);
    }

    /// Clear the host -> DLL "swap eyes" control bit, returning polarity to the
    /// game's native mapping.
    pub fn clear_invert_eyes(&self) {
        self.flags.fetch_and(!FLAG_INVERT_EYES, Ordering::Relaxed);
    }

    /// True when the host is requesting the DLL to swap its rendered eyes.
    pub fn invert_eyes(&self) -> bool {
        self.flags() & FLAG_INVERT_EYES != 0
    }

    /// Reads the window owner's published target connector name (e.g. `DP-2`),
    /// or an empty string if none is published (or the region is not ready).
    /// The window owner writes this whenever the window moves heads; the helper
    /// re-anchors when the value changes.  Byte reads are relaxed/best-effort:
    /// a torn read only yields a bogus name that connector resolution rejects,
    /// never a crash.
    pub fn connector_name(&self) -> String {
        if self.cap == 0 || self.connector.is_null() {
            return String::new();
        }
        let mut buf = [0u8; CONNECTOR_LEN];
        unsafe {
            std::ptr::copy_nonoverlapping(self.connector, buf.as_mut_ptr(), CONNECTOR_LEN);
        }
        std::sync::atomic::fence(Ordering::Acquire);
        let end = buf.iter().position(|&b| b == 0).unwrap_or(CONNECTOR_LEN);
        String::from_utf8_lossy(&buf[..end]).trim().to_string()
    }

    /// Publishes the target connector name (window owner side).  Truncated to
    /// the field width and null-terminated.  Empty clears the field.  The plain
    /// byte store is published to the helper by the release fence (mirrors the
    /// helper's acquire fence on read).
    pub fn set_connector_name(&self, name: &str) {
        if self.cap == 0 || self.connector.is_null() {
            return;
        }
        let bytes = name.as_bytes();
        let n = bytes.len().min(CONNECTOR_LEN - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.connector, n);
            // Null-terminate (and zero any trailing bytes so a stale longer
            // name can't leak into a later read).
            std::ptr::write_bytes(self.connector.add(n), 0, CONNECTOR_LEN - n);
        }
        std::sync::atomic::fence(Ordering::Release);
    }

    /// Producer (DLL) side: pushes one eye-swap command.  Returns `false` if
    /// the ring is full (the DLL's caller then just drops the pulse, which the
    /// next frame re-syncs anyway).  The DLL mirrors this exact algorithm.
    ///
    /// The slot is written at the region's NEGOTIATED stride, keyed on the SAME
    /// gate [`Self::stamped`] the consumer's [`Self::drain`] uses to size its
    /// reads: 16-byte `RingSlot`s on a stamped version-2 region, 8-byte
    /// `{seq, eye, pad}` on a legacy one.  A producer must advance by the stride
    /// the header advertises -- never blindly by `size_of::<RingSlot>()` -- or
    /// its `eye` bytes would land at a different offset than the strided
    /// consumer reads, scrambling the L/R order.
    pub fn enqueue(&self, eye: u8) -> bool {
        let cap = self.cap;
        if cap == 0 {
            return false;
        }
        let mask = cap - 1;
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= cap {
            return false; // full
        }
        let stamping = self.stamped();
        let stride = if stamping {
            SLOT_SIZE_STAMPED
        } else {
            SLOT_SIZE
        };
        let idx = (head & mask) as usize;
        let seq = head.wrapping_add(1);
        // Byte address of this slot at the negotiated stride.  `slots` is
        // `base + HDR_SIZE`; we advance by `idx * stride` from there (NOT
        // `slots.add(idx)`, which would stride by `size_of::<RingSlot>()` and
        // land the `eye` byte at the wrong offset on a legacy region).  The
        // base is page-aligned and both strides are multiples of 8, so every
        // stamped (16-byte) slot and every even legacy (8-byte) slot is 8-byte
        // aligned; an odd legacy slot is 8 bytes off that, so its 4-byte `seq`
        // is unaligned -- hence `write_unaligned` below.
        let slot = (self.slots as usize) + idx * stride;
        // Slot stores are plain; the Release store of `head` below publishes
        // them to the consumer's Acquire load.
        unsafe {
            if stamping {
                let s = slot as *mut RingSlot;
                (*s).seq = seq;
                (*s).eye = eye;
                (*s).flags = 0;
                (*s).pad = [0, 0];
                (*s).t_us = 0;
            } else {
                // Legacy 8-byte slot: {seq(4), eye(1), pad(3)}.
                let p = slot as *mut u8;
                (p as *mut u32).write_unaligned(seq);
                p.add(4).write(eye);
                p.add(5).write_bytes(0u8, 3);
            }
        }
        self.head.store(seq, Ordering::Release);
        true
    }

    /// Consumes every pending eye-swap command and returns them, in order, with
    /// (version-2 regions) the producer's submit-time stamp.  Reads the wire
    /// version fresh on every call so a producer bump (1 -> 2) or downgrade is
    /// picked up immediately.  The producer (DLL) releases `head`; this
    /// consumer owns `tail`.
    pub fn drain(&self) -> Vec<Swap> {
        let cap = self.cap;
        if cap == 0 {
            return Vec::new();
        }
        let stamped = self.stamped();
        let stride = if stamped {
            SLOT_SIZE_STAMPED
        } else {
            SLOT_SIZE
        };
        let mask = cap - 1;
        let head = self.head.load(Ordering::Acquire);
        let mut tail = self.tail.load(Ordering::Relaxed);
        let mut swaps = Vec::with_capacity(head.wrapping_sub(tail) as usize);
        while tail != head {
            // SAFETY: both strides keep slot index `tail & mask` inside the
            // mapped region (HDR_SIZE + RING_CAP*16 <= SHM_FILE_SIZE, asserted
            // above), and stamped slots are 8-byte aligned (base is page
            // aligned, HDR_SIZE and the stride are multiples of 8).
            let addr = (self.slots as usize) + ((tail & mask) as usize) * stride;
            let swap = unsafe {
                if stamped {
                    let s = &*(addr as *const RingSlot);
                    Swap {
                        eye: s.eye,
                        t_us: Some(s.t_us),
                        seq: s.seq,
                    }
                } else {
                    // Legacy 8-byte slot; the eye byte sits at the same offset
                    // 4 in both layouts.
                    Swap {
                        eye: *(addr as *const u8).add(4),
                        t_us: None,
                        seq: 0,
                    }
                }
            };
            swaps.push(swap);
            tail = tail.wrapping_add(1);
            // Release the slot before we do the (comparatively slow) USB write
            // so the producer can keep enqueueing without waiting on us.
            self.tail.store(tail, Ordering::Release);
        }
        swaps
    }
}

/// Sets the free-running head/tail to the same value to clear the ring.
pub fn reset_ring(shm: &Shm) {
    shm.head.store(0, Ordering::Release);
    shm.tail.store(0, Ordering::Release);
}
