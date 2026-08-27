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
pub const SHM_VERSION: u32 = 1;
/// Ring capacity in slots (must be a power of two).
pub const RING_CAP: u32 = 1024;
pub const SLOT_SIZE: usize = 8;
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

/// Emitter state reported by the helper (helper -> DLL).
pub const STATUS_NONE: u32 = 0;
pub const STATUS_OPENING: u32 = 1;
pub const STATUS_READY: u32 = 2;
pub const STATUS_ERROR: u32 = 3;
pub const STATUS_CLOSED: u32 = 4;

/// `flags` bits (helper -> DLL).
pub const FLAG_FIRMWARE_LOADED: u32 = 1 << 0;
pub const FLAG_EMITTER_PRESENT: u32 = 1 << 1;

/// Eye index carried in a ring slot (DLL -> helper).
pub const EYE_LEFT: u8 = 0;
pub const EYE_RIGHT: u8 = 1;

/// Backing file path on the host.  The DLL opens the same file through the
/// Wine `Z:` drive as `Z:\tmp\nvstusb.shm`.
pub const DEFAULT_SHM_PATH: &str = "/tmp/nvstusb.shm";

/// A single eye-swap command slot.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RingSlot {
    /// Producer sequence number.  Used for diagnostics and to disambiguate a
    /// slot that still holds a stale command from a freshly written one.
    pub seq: u32,
    /// `EYE_LEFT` or `EYE_RIGHT`.
    pub eye: u8,
    pub pad: [u8; 3],
}

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

    /// Validates the header; returns the ring capacity (0 if invalid).
    pub fn cap(&self) -> u32 {
        let magic = atom(self.base, OFF_MAGIC).load(Ordering::Relaxed);
        let version = atom(self.base, OFF_VERSION).load(Ordering::Relaxed);
        let cap = atom(self.base, OFF_CAP).load(Ordering::Relaxed);
        if magic != SHM_MAGIC || version != SHM_VERSION || cap == 0 || !cap.is_power_of_two() {
            0
        } else {
            cap
        }
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

    /// Producer (DLL) side: pushes one eye-swap command.  Returns `false` if
    /// the ring is full (the DLL's caller then just drops the pulse, which the
    /// next frame re-syncs anyway).  The DLL mirrors this exact algorithm.
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
        let idx = (head & mask) as usize;
        let seq = head.wrapping_add(1);
        // Slot stores are plain; the Release store of `head` below publishes
        // them to the consumer's Acquire load.
        unsafe {
            (*self.slots.add(idx)).seq = seq;
            (*self.slots.add(idx)).eye = eye;
        }
        self.head.store(seq, Ordering::Release);
        true
    }

    /// Consumes every pending eye-swap command and returns the eyes, in
    /// order.  The producer (DLL) releases `head`; this consumer owns `tail`.
    pub fn drain(&self) -> Vec<u8> {
        let cap = self.cap;
        let mask = cap - 1;
        let head = self.head.load(Ordering::Acquire);
        let mut tail = self.tail.load(Ordering::Relaxed);
        let mut eyes = Vec::with_capacity(head.wrapping_sub(tail) as usize);
        while tail != head {
            let slot = unsafe { &*self.slots.add((tail & mask) as usize) };
            eyes.push(slot.eye);
            tail = tail.wrapping_add(1);
            // Release the slot before we do the (comparatively slow) USB write
            // so the producer can keep enqueueing without waiting on us.
            self.tail.store(tail, Ordering::Release);
        }
        eyes
    }
}

/// Sets the free-running head/tail to the same value to clear the ring.
pub fn reset_ring(shm: &Shm) {
    shm.head.store(0, Ordering::Release);
    shm.tail.store(0, Ordering::Release);
}
