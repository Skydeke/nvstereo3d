//! Direct KMS/GBM/EGL rendering backend (no Wayland compositor).
//!
//! Renders the scene straight to the display engine via DRM page flips, so
//! the present vblank is exactly the vblank the [`super::drm::DrmVblank`]
//! anchor predicts.  This removes the compositor from the shutter-timing
//! equation entirely: with a windowed/Wayland surface the swap return is a
//! commit ack that can lead the real present by several vblanks (observed
//! present-err max ~2.4 periods), which no host-side phase can compensate.
//!
//! Enable with `NVSTUSB_KMS=1` together with `NVSTUSB_DRM=1`.  Requires
//! running on a VT where nothing else holds DRM master (switch to a spare
//! tty, or stop the compositor on that VT), plus Mesa EGL built with the
//! GBM platform (standard on Arch).  All of libgbm / libEGL are resolved at
//! runtime through `libloading`, so the crate builds and links without any
//! system dev packages.

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_short, c_ulong, c_void};
use std::fs::File;
use std::mem;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// DRM ABI (linux/drm.h + drm_mode.h).  Sizes are static-asserted below.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DrmModeModeInfo {
    clock: u32,
    hdisplay: u16,
    hsync_start: u16,
    hsync_end: u16,
    htotal: u16,
    hskew: u16,
    vdisplay: u16,
    vsync_start: u16,
    vsync_end: u16,
    vtotal: u16,
    vscan: u16,
    /// Vertical refresh in Hz for the chosen mode.
    pub vrefresh: u32,
    flags: u32,
    r#type: u32,
    name: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DrmModeCardRes {
    fb_id_ptr: u64,
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
#[derive(Clone, Copy, Default)]
struct DrmModeGetConnector {
    encoders_ptr: u64,
    modes_ptr: u64,
    props_ptr: u64,
    prop_values_ptr: u64,
    count_modes: u32,
    count_props: u32,
    count_encoders: u32,
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
#[derive(Clone, Copy, Default)]
struct DrmModeCrtc {
    set_connectors_ptr: u64,
    count_connectors: u32,
    crtc_id: u32,
    fb_id: u32,
    x: u32,
    y: u32,
    gamma_size: u32,
    mode_valid: u32,
    mode: DrmModeModeInfo,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmModeFbCmd2 {
    fb_id: u32,
    width: u32,
    height: u32,
    pixel_format: u32,
    flags: u32,
    handles: [u32; 4],
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: [u64; 4],
    reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmModeCrtcPageFlip {
    crtc_id: u32,
    fb_id: u32,
    flags: u32,
    reserved: u32,
    user_data: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmEvent {
    r#type: u32,
    length: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrmEventVblank {
    base: DrmEvent,
    user_data: u64,
    tv_sec: u32,
    tv_usec: u32,
    sequence: u32,
    crtc_id: u32,
}

#[repr(C)]
struct PollFd {
    fd: c_int,
    events: c_short,
    revents: c_short,
}

const _: () = assert!(std::mem::size_of::<DrmModeModeInfo>() == 68);
const _: () = assert!(std::mem::size_of::<DrmModeCardRes>() == 64);
const _: () = assert!(std::mem::size_of::<DrmModeGetConnector>() == 80);
const _: () = assert!(std::mem::size_of::<DrmModeCrtc>() == 104);
const _: () = assert!(std::mem::size_of::<DrmModeFbCmd2>() == 120);
const _: () = assert!(std::mem::size_of::<DrmModeCrtcPageFlip>() == 24);
const _: () = assert!(std::mem::size_of::<DrmModeEncoder>() == 24);
const _: () = assert!(std::mem::size_of::<DrmEventVblank>() == 32);

/// `_IOC(dir, 'd', nr, size)` helper, mirroring the kernel's encoding.
const fn drm_ioc(dir: u32, nr: u32, size: usize) -> c_ulong {
    const IOC_NONE: u32 = 0;
    const IOC_READ: u32 = 2;
    const IOC_WRITE: u32 = 1;
    const IOC_NRSHIFT: u32 = 0;
    const IOC_TYPESHIFT: u32 = 8;
    const IOC_SIZESHIFT: u32 = 16;
    const IOC_DIRSHIFT: u32 = 30;
    let ty = b'd' as u32;
    ((dir as c_ulong) << IOC_DIRSHIFT)
        | ((size as c_ulong) << IOC_SIZESHIFT)
        | ((ty as c_ulong) << IOC_TYPESHIFT)
        | (nr as c_ulong)
}

const fn drm_io(dir: u32, nr: u32) -> c_ulong {
    drm_ioc(dir, nr, 0)
}

const DRM_IOCTL_SET_MASTER: c_ulong = drm_io(0, 0x1e);
const DRM_IOCTL_DROP_MASTER: c_ulong = drm_io(0, 0x1f);
const DRM_IOCTL_MODE_GETRESOURCES: c_ulong = drm_ioc(3, 0xA0, std::mem::size_of::<DrmModeCardRes>());
const DRM_IOCTL_MODE_SETCRTC: c_ulong = drm_ioc(3, 0xA2, std::mem::size_of::<DrmModeCrtc>());
const DRM_IOCTL_MODE_GETCONNECTOR: c_ulong =
    drm_ioc(3, 0xA7, std::mem::size_of::<DrmModeGetConnector>());
const DRM_IOCTL_MODE_GETENCODER: c_ulong =
    drm_ioc(3, 0xA6, std::mem::size_of::<DrmModeEncoder>());
const DRM_IOCTL_MODE_RMFB: c_ulong = drm_ioc(3, 0xAF, 4);
const DRM_IOCTL_MODE_PAGE_FLIP: c_ulong =
    drm_ioc(3, 0xB0, std::mem::size_of::<DrmModeCrtcPageFlip>());
const DRM_IOCTL_MODE_ADDFB2: c_ulong = drm_ioc(3, 0xB8, std::mem::size_of::<DrmModeFbCmd2>());

const DRM_MODE_PAGE_FLIP_EVENT: u32 = 0x01;
const DRM_EVENT_FLIP_COMPLETE: u32 = 0x02;
const DRM_MODE_CONNECTED: u32 = 1;
const DRM_MODE_TYPE_PREFERRED: u32 = 1 << 3;
const POLLIN: c_short = 0x001;

const DRM_FORMAT_XRGB8888: u32 = 0x3432_5258; // fourcc('X','R','2','4')

// ---------------------------------------------------------------------------
// GBM ABI (gbm.h) — resolved at runtime.
// ---------------------------------------------------------------------------

#[repr(C)]
union GbmBoHandle {
    ptr: *mut c_void,
    s32: i32,
    u32: u32,
}

struct GbmFns {
    device: *mut c_void,
    surface: *mut c_void,
    create_device: unsafe extern "C" fn(c_int) -> *mut c_void,
    device_destroy: unsafe extern "C" fn(*mut c_void),
    surface_create: unsafe extern "C" fn(*mut c_void, u32, u32, u32, u32) -> *mut c_void,
    surface_destroy: unsafe extern "C" fn(*mut c_void),
    has_free_buffers: unsafe extern "C" fn(*mut c_void) -> c_int,
    lock_front: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    release_buffer: unsafe extern "C" fn(*mut c_void, *mut c_void),
    get_handle: unsafe extern "C" fn(*mut c_void) -> GbmBoHandle,
    get_width: unsafe extern "C" fn(*mut c_void) -> u32,
    get_height: unsafe extern "C" fn(*mut c_void) -> u32,
    get_stride: unsafe extern "C" fn(*mut c_void) -> u32,
}

const GBM_FORMAT_XRGB8888: u32 = DRM_FORMAT_XRGB8888;
const GBM_BO_USE_SCANOUT: u32 = 1 << 0;
const GBM_BO_USE_RENDERING: u32 = 1 << 1;

// ---------------------------------------------------------------------------
// Present-stage diagnostics
//
// The VT session showed a rock-steady ~3-vblank frame cadence whose cause
// could not be attributed from the outside: "swap write" only measures the
// USB eye packet, so the entire present pipeline was a black box. These
// per-stage accumulators break it open - swap (eglSwapBuffers incl. flush +
// any driver throttle), lock (gbm lock_front + ADDFB2), flipq (the page-flip
// ioctl) and flipwait (blocking on the FLIP_COMPLETE event) - so the perf
// report shows exactly which stage eats the periods. Statics because the
// sampler runs inside `present(&mut self)` while the printer lives on the
// app side; all counters reset on read (perf-window semantics).
// ---------------------------------------------------------------------------

struct Stage {
    n: AtomicU64,
    total_us: AtomicU64,
    max_us: AtomicU64,
}

impl Stage {
    const fn new() -> Self {
        Self {
            n: AtomicU64::new(0),
            total_us: AtomicU64::new(0),
            max_us: AtomicU64::new(0),
        }
    }

    fn sample(&self, us: u64) {
        self.n.fetch_add(1, Ordering::Relaxed);
        self.total_us.fetch_add(us, Ordering::Relaxed);
        let mut prev = self.max_us.load(Ordering::Relaxed);
        while us > prev {
            match self
                .max_us
                .compare_exchange_weak(prev, us, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(cur) => prev = cur,
            }
        }
    }

    /// (count, avg us, max us); resets the accumulator.
    fn take(&self) -> Option<(u64, u64, u64)> {
        let n = self.n.swap(0, Ordering::Relaxed);
        if n == 0 {
            return None;
        }
        Some((
            n,
            self.total_us.swap(0, Ordering::Relaxed) / n,
            self.max_us.swap(0, Ordering::Relaxed),
        ))
    }
}

static ST_SWAP: Stage = Stage::new();
static ST_LOCK: Stage = Stage::new();
static ST_FLIPQ: Stage = Stage::new();
static ST_WAIT: Stage = Stage::new();
/// Frames where eglGetError returned non-SUCCESS right after
/// eglSwapBuffers - the visible symptom of rejected GPU pushes (the nouveau
/// `fail ttm_validate` / pushbuf-EINVAL class of failures).
static ST_EGLERR: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// EGL ABI (EGL/egl.h) — resolved at runtime.
// ---------------------------------------------------------------------------

type EglBoolean = u32;
type EglInt = i32;
type EglEnum = u32;
type EglDisplay = *mut c_void;
type EglConfig = *const c_void;
type EglSurface = *mut c_void;
type EglContext = *mut c_void;

const EGL_SUCCESS: EglEnum = 0x3000;
const EGL_NO_DISPLAY: EglDisplay = std::ptr::null_mut();
const EGL_NO_SURFACE: EglSurface = std::ptr::null_mut();
const EGL_NO_CONTEXT: EglContext = std::ptr::null_mut();
const EGL_OPENGL_API: EglEnum = 0x30A2;
const EGL_PLATFORM_GBM_KHR: EglEnum = 0x31D7;
const EGL_SURFACE_TYPE: EglInt = 0x3033;
const EGL_WINDOW_BIT: EglInt = 0x0004;
const EGL_RED_SIZE: EglInt = 0x3024;
const EGL_GREEN_SIZE: EglInt = 0x3023;
const EGL_BLUE_SIZE: EglInt = 0x3022;
const EGL_DEPTH_SIZE: EglInt = 0x3025;
const EGL_RENDERABLE_TYPE: EglInt = 0x3040;
const EGL_OPENGL_BIT: EglInt = 0x0008;
const EGL_NATIVE_VISUAL_ID: EglInt = 0x302E;
const EGL_CONTEXT_CLIENT_TYPE: EglInt = 0x3097;
const EGL_CONTEXT_MAJOR_VERSION: EglInt = 0x3098;
const EGL_CONTEXT_MINOR_VERSION: EglInt = 0x30FB;
const EGL_NONE: EglInt = 0x3038;
const EGL_TRUE: EglBoolean = 1;
/// Surface attributes carrying the driver's accepted swap-interval range.
const EGL_MIN_SWAP_INTERVAL: EglInt = 0x30F7;
const EGL_MAX_SWAP_INTERVAL: EglInt = 0x30F8;

struct EglFns {
    display: EglDisplay,
    surface: EglSurface,
    context: EglContext,
    get_error: unsafe extern "C" fn() -> EglEnum,
    swap_buffers: unsafe extern "C" fn(EglDisplay, EglSurface) -> EglBoolean,
    swap_interval: unsafe extern "C" fn(EglDisplay, EglInt) -> EglBoolean,
    make_current: unsafe extern "C" fn(EglDisplay, EglSurface, EglSurface, EglContext) -> EglBoolean,
    query_surface: unsafe extern "C" fn(EglDisplay, EglSurface, EglInt, *mut EglInt) -> EglBoolean,
    get_proc_addr: unsafe extern "C" fn(*const c_char) -> *const c_void,
    terminate: unsafe extern "C" fn(EglDisplay) -> EglBoolean,
    destroy_surface: unsafe extern "C" fn(EglDisplay, EglSurface) -> EglBoolean,
    destroy_context: unsafe extern "C" fn(EglDisplay, EglContext) -> EglBoolean,
    /// Whether `eglSwapInterval(display, 0)` was rejected by the driver, i.e.
    /// whether `eglSwapBuffers` is still expected to eat one vblank of
    /// throttle before our manual page flip. Decided in [`KmsDisplay::
    /// make_current`] - the call is only valid once a context is current -
    /// and re-derived from measured present errors at runtime either way.
    vsync_throttled: bool,
}

unsafe extern "C" {
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    fn poll(fds: *mut PollFd, nfds: c_ulong, timeout: c_int) -> c_int;
    fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
}

/// Resolves a symbol from a loaded library, or `None` if missing.
fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Option<T> {
    unsafe { lib.get(name).ok().map(|s: Symbol<T>| *s) }
}

/// Loads the GBM entry points from `libgbm.so`.
fn load_gbm(lib: &Library) -> Option<GbmFns> {
    let gbm = GbmFns {
        device: std::ptr::null_mut(),
        surface: std::ptr::null_mut(),
        create_device: sym(lib, b"gbm_create_device\0")?,
        device_destroy: sym(lib, b"gbm_device_destroy\0")?,
        surface_create: sym(lib, b"gbm_surface_create\0")?,
        surface_destroy: sym(lib, b"gbm_surface_destroy\0")?,
        has_free_buffers: sym(lib, b"gbm_surface_has_free_buffers\0")?,
        lock_front: sym(lib, b"gbm_surface_lock_front_buffer\0")?,
        release_buffer: sym(lib, b"gbm_surface_release_buffer\0")?,
        get_handle: sym(lib, b"gbm_bo_get_handle\0")?,
        get_width: sym(lib, b"gbm_bo_get_width\0")?,
        get_height: sym(lib, b"gbm_bo_get_height\0")?,
        get_stride: sym(lib, b"gbm_bo_get_stride\0")?,
    };
    Some(gbm)
}

/// The direct-KMS rendering surface.
pub struct KmsDisplay {
    _gbm_lib: Library,
    _egl_lib: Library,
    file: File,
    gbm: GbmFns,
    egl: EglFns,
    connector_id: u32,
    crtc_id: u32,
    /// The mode selected for scanout (public for refresh-rate reporting).
    pub mode: DrmModeModeInfo,
    /// Width/height of the chosen mode (physical px).
    pub width: u32,
    pub height: u32,
    /// FB id of the buffer currently scanned out (0 before the first flip).
    cur_fb: u32,
    /// GBM bo currently scanned out.
    cur_bo: *mut c_void,
    first: bool,
    /// Path of the card we took over (for diagnostics).
    pub card_path: String,
}

impl KmsDisplay {
    /// Opens a DRM card, takes master, sets the mode, and creates the
    /// GBM/EGL surface.  Picks the card with a connected connector that
    /// offers a ~120 Hz mode (i.e. the projector), preferring that over a
    /// plain connected connector.
    pub fn open() -> Result<KmsDisplay, String> {
        let gbm_lib = unsafe { Library::new("libgbm.so.1") }
            .or_else(|_| unsafe { Library::new("libgbm.so") })
            .map_err(|e| format!("cannot dlopen libgbm: {e}"))?;
        let egl_lib = unsafe { Library::new("libEGL.so.1") }
            .or_else(|_| unsafe { Library::new("libEGL.so") })
            .map_err(|e| format!("cannot dlopen libEGL: {e}"))?;
        let gbm_fns = load_gbm(&gbm_lib).ok_or("libgbm is missing required symbols")?;

        let mut reasons: Vec<String> = Vec::new();
        for i in 0..4 {
            let path = format!("/dev/dri/card{i}");
            let file = match File::options().read(true).write(true).open(&path) {
                Ok(f) => f,
                Err(e) => {
                    reasons.push(format!("{path}: {e}"));
                    continue;
                }
            };
            let fd = file.as_raw_fd();

            if unsafe { ioctl(fd, DRM_IOCTL_SET_MASTER, 0u32) } != 0 {
                reasons.push(format!(
                    "{path}: DRM master unavailable (is a compositor running on this VT? switch to a spare tty, e.g. Ctrl+Alt+F3)"
                ));
                continue;
            }

            let (connector_id, crtc_id, mode) =
                match find_best_connector(fd) {
                    Ok(x) => x,
                    Err(e) => {
                        let _ = unsafe { ioctl(fd, DRM_IOCTL_DROP_MASTER, 0u32) };
                        reasons.push(format!("{path}: {e}"));
                        continue;
                    }
                };

            // Mode setup: create the GBM device + surface, then EGL on top.
            let gbm_device = unsafe { (gbm_fns.create_device)(fd) };
            if gbm_device.is_null() {
                let _ = unsafe { ioctl(fd, DRM_IOCTL_DROP_MASTER, 0u32) };
                reasons.push(format!("{path}: gbm_create_device failed"));
                continue;
            }
            let gbm_surface = unsafe {
                (gbm_fns.surface_create)(
                    gbm_device,
                    mode.hdisplay as u32,
                    mode.vdisplay as u32,
                    GBM_FORMAT_XRGB8888,
                    GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING,
                )
            };
            if gbm_surface.is_null() {
                unsafe { (gbm_fns.device_destroy)(gbm_device) };
                let _ = unsafe { ioctl(fd, DRM_IOCTL_DROP_MASTER, 0u32) };
                reasons.push(format!("{path}: gbm_surface_create failed"));
                continue;
            }

            let egl = match init_egl(&egl_lib, gbm_device, gbm_surface) {
                Ok(e) => e,
                Err(e) => {
                    unsafe { (gbm_fns.surface_destroy)(gbm_surface) };
                    unsafe { (gbm_fns.device_destroy)(gbm_device) };
                    let _ = unsafe { ioctl(fd, DRM_IOCTL_DROP_MASTER, 0u32) };
                    reasons.push(format!("{path}: {e}"));
                    continue;
                }
            };

            eprintln!(
                "nvstusb: KMS mode set on {path}: {}x{} @ {} Hz (connector {}, crtc {})",
                mode.hdisplay, mode.vdisplay, mode.vrefresh, connector_id, crtc_id
            );
            return Ok(KmsDisplay {
                _gbm_lib: gbm_lib,
                _egl_lib: egl_lib,
                file,
                gbm: GbmFns {
                    device: gbm_device,
                    surface: gbm_surface,
                    ..gbm_fns
                },
                egl,
                connector_id,
                crtc_id,
                mode,
                width: mode.hdisplay as u32,
                height: mode.vdisplay as u32,
                cur_fb: 0,
                cur_bo: std::ptr::null_mut(),
                first: true,
                card_path: path,
            });
        }
        Err(format!(
            "no usable DRM card ({}): {}",
            reasons.len(),
            reasons.join("; ")
        ))
    }

    /// Makes the EGL context current so rendering can proceed.  Called once
    /// at startup; `present()` keeps the context current for its lifetime.
    pub fn make_current(&mut self) -> Result<(), String> {
        let ok = unsafe { (self.egl.make_current)(self.egl.display, self.egl.surface, self.egl.surface, self.egl.context) };
        if ok != EGL_TRUE {
            return Err(format!("eglMakeCurrent failed: {:#x}", self.egl_error()));
        }

        // NOW the swap-interval request is valid: a context is current and
        // this surface is its draw surface. Ask for interval 0 so
        // eglSwapBuffers stays asynchronous and OUR page flip is the sole
        // vblank pacer (a throttled swap delays the flip by one full period,
        // halving throughput on top of the flip's own latency).
        let accepted = unsafe { (self.egl.swap_interval)(self.egl.display, 0) } == EGL_TRUE;
        let err = if accepted { EGL_SUCCESS } else { unsafe { (self.egl.get_error)() } };
        let mut min = EglInt::default();
        let mut max = EglInt::default();
        let range_ok = unsafe { (self.egl.query_surface)(self.egl.display, self.egl.surface, EGL_MIN_SWAP_INTERVAL, &mut min) } == EGL_TRUE
            && unsafe { (self.egl.query_surface)(self.egl.display, self.egl.surface, EGL_MAX_SWAP_INTERVAL, &mut max) } == EGL_TRUE;
        eprintln!(
            "nvstusb: eglSwapInterval(0) {} ({:#x}; accepted range {}..{})",
            if accepted { "accepted" } else { "REJECTED" },
            err,
            if range_ok { min.to_string() } else { "?".to_string() },
            if range_ok { max.to_string() } else { "?".to_string() },
        );
        if !accepted {
            eprintln!(
                "nvstusb: falling back to driver-throttled swaps; eye inversion compensates"
            );
        }
        self.egl.vsync_throttled = !accepted;
        Ok(())
    }

    /// Resolves a GL symbol through EGL (used by `Gl::load_with`).
    pub fn get_proc_address(&self, name: *const c_char) -> *const c_void {
        unsafe { (self.egl.get_proc_addr)(name) }
    }

    /// Whether the driver rejected `eglSwapInterval(0)` and so still
    /// vblank-throttles inside `eglSwapBuffers`, pushing our manual page flip
    /// one extra vblank late. The method-4 eye inversion in `nvstusb::swap`
    /// is only correct when this is `true`.
    pub fn vsync_throttled(&self) -> bool {
        self.egl.vsync_throttled
    }

    /// One-line summary of the per-stage present costs accumulated since
    /// the last call, or `None` when nothing was sampled (e.g. the windowed
    /// backend). Printed by the demo's perf report; resets on read.
    pub fn take_present_stats_line() -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        for (name, stage) in [
            ("swap", &ST_SWAP),
            ("lock", &ST_LOCK),
            ("flipq", &ST_FLIPQ),
            ("flipwait", &ST_WAIT),
        ] {
            if let Some((n, avg, max)) = stage.take() {
                parts.push(format!("{name} avg={avg}us max={max}us n={n}"));
            }
        }
        let egl_err = ST_EGLERR.swap(0, Ordering::Relaxed);
        if egl_err > 0 {
            parts.push(format!("egl-err-after-swap n={egl_err}"));
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" | "))
        }
    }

    /// Presents the current back buffer by swapping EGL buffers and page
    /// flipping it to the CRTC, blocking until the flip lands on a vblank.
    /// The first call uses `SETCRTC` to establish the mode.
    ///
    /// Returns the flip's kernel-hardware vblank timestamp (CLOCK_MONOTONIC
    /// microseconds) when one is available - i.e. every call except the
    /// first, which mode-sets synchronously and has no flip event to read.
    pub fn present(&mut self) -> Result<Option<u64>, String> {
        let t0 = Instant::now();
        let ok = unsafe { (self.egl.swap_buffers)(self.egl.display, self.egl.surface) };
        if ok != EGL_TRUE {
            let free = unsafe { (self.gbm.has_free_buffers)(self.gbm.surface) };
            return Err(format!(
                "eglSwapBuffers failed: {:#x} (surface has_free_buffers={})",
                self.egl_error(),
                free
            ));
        }
        // Flush errors are deferred: a failed GPU submit often still returns
        // EGL_TRUE from SwapBuffers and only shows up here. Count them - a
        // nonzero rate correlates with the kernel's `fail ttm_validate`
        // spam and explains missing/stalled frames.
        if unsafe { (self.egl.get_error)() } != EGL_SUCCESS {
            ST_EGLERR.fetch_add(1, Ordering::Relaxed);
        }
        ST_SWAP.sample(t0.elapsed().as_micros() as u64);

        let t1 = Instant::now();
        let next_bo = unsafe { (self.gbm.lock_front)(self.gbm.surface) };
        if next_bo.is_null() {
            return Err("gbm_surface_lock_front_buffer returned NULL".into());
        }
        let fb = self.add_fb(next_bo)?;
        ST_LOCK.sample(t1.elapsed().as_micros() as u64);
        let fd = self.file.as_raw_fd();

        let t2 = Instant::now();
        let r = if self.first {
            unsafe {
                ioctl(
                    fd,
                    DRM_IOCTL_MODE_SETCRTC,
                    &mut DrmModeCrtc {
                        set_connectors_ptr: (&self.connector_id as *const u32) as u64,
                        count_connectors: 1,
                        crtc_id: self.crtc_id,
                        fb_id: fb,
                        x: 0,
                        y: 0,
                        gamma_size: 0,
                        mode_valid: 1,
                        mode: self.mode,
                    },
                )
            }
        } else {
            unsafe {
                ioctl(
                    fd,
                    DRM_IOCTL_MODE_PAGE_FLIP,
                    &mut DrmModeCrtcPageFlip {
                        crtc_id: self.crtc_id,
                        fb_id: fb,
                        flags: DRM_MODE_PAGE_FLIP_EVENT,
                        reserved: 0,
                        user_data: 0,
                    },
                )
            }
        };

        if r != 0 {
            let e = std::io::Error::last_os_error();
            unsafe { (self.gbm.release_buffer)(self.gbm.surface, next_bo) };
            let _ = unsafe { ioctl(fd, DRM_IOCTL_MODE_RMFB, &fb) };
            return Err(format!("page flip failed: {e}"));
        }
        ST_FLIPQ.sample(t2.elapsed().as_micros() as u64);

        if self.first {
            // SETCRTC is synchronous and takes effect at the next vblank.
            self.first = false;
            self.cur_fb = fb;
            self.cur_bo = next_bo;
            return Ok(None);
        }

        // Block until the flip completes at a vblank, then the previously
        // scanned buffer is free to reuse.
        let t3 = Instant::now();
        let flip_mono_us = self.wait_flip()?;
        ST_WAIT.sample(t3.elapsed().as_micros() as u64);
        if self.cur_fb != 0 {
            let _ = unsafe { ioctl(fd, DRM_IOCTL_MODE_RMFB, &self.cur_fb) };
        }
        if !self.cur_bo.is_null() {
            unsafe { (self.gbm.release_buffer)(self.gbm.surface, self.cur_bo) };
        }
        self.cur_fb = fb;
        self.cur_bo = next_bo;
        Ok(Some(flip_mono_us))
    }

    fn egl_error(&self) -> EglEnum {
        unsafe { (self.egl.get_error)() }
    }

    fn add_fb(&mut self, bo: *mut c_void) -> Result<u32, String> {
        let handle = {
            let h = unsafe { (self.gbm.get_handle)(bo) };
            unsafe { h.u32 }
        };
        let width = unsafe { (self.gbm.get_width)(bo) };
        let height = unsafe { (self.gbm.get_height)(bo) };
        let stride = unsafe { (self.gbm.get_stride)(bo) };
        let mut fb = DrmModeFbCmd2 {
            fb_id: 0,
            width,
            height,
            pixel_format: DRM_FORMAT_XRGB8888,
            flags: 0,
            handles: [handle, 0, 0, 0],
            pitches: [stride, 0, 0, 0],
            offsets: [0, 0, 0, 0],
            modifier: [0, 0, 0, 0],
            reserved: [0, 0, 0],
        };
        let fd = self.file.as_raw_fd();
        let r = unsafe { ioctl(fd, DRM_IOCTL_MODE_ADDFB2, &mut fb) };
        if r != 0 {
            return Err(format!("drmModeAddFB2 failed: {}", std::io::Error::last_os_error()));
        }
        Ok(fb.fb_id)
    }

    /// Blocks for the page-flip completion event and returns its
    /// kernel-hardware timestamp (CLOCK_MONOTONIC microseconds, same clock
    /// [`super::drm::DrmVblank`] anchors to).
    ///
    /// The event carries the vblank's own `tv_sec`/`tv_usec` (`struct
    /// drm_event_vblank`), timestamped by the display driver's interrupt
    /// handler. Previously this was parsed only as far as the 8-byte
    /// `drm_event` header and discarded; the caller then stamped "now" with
    /// `Instant::now()` *after* `poll()`+`read()` returned, which folds in
    /// scheduler/wakeup jitter (can be tens to hundreds of us) on top of the
    /// real vblank time. That extra jitter fed straight into the phase the
    /// IR emitter is synced to, which is consistent with the intermittent
    /// edge crosstalk: a resync landing during a jitter spike bakes a bad
    /// phase in until the next resync. Using the event's own timestamp
    /// removes that source of error for free (no extra syscall - it's in
    /// the bytes already read).
    fn wait_flip(&self) -> Result<u64, String> {
        let fd = self.file.as_raw_fd();
        let mut pf = PollFd {
            fd,
            events: POLLIN,
            revents: 0,
        };
        let r = unsafe { poll(&mut pf, 1, 3000) };
        if r <= 0 {
            return Err("timed out waiting for page flip event".into());
        }
        let mut buf = [0u8; 64];
        let n = unsafe { read(fd, buf.as_mut_ptr() as *mut c_void, buf.len()) };
        if n < std::mem::size_of::<DrmEventVblank>() as isize {
            return Err("short read waiting for drm event".into());
        }
        let ev = DrmEvent {
            r#type: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            length: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
        };
        if ev.r#type != DRM_EVENT_FLIP_COMPLETE {
            return Err(format!("unexpected drm event type {:#x}", ev.r#type));
        }
        let vbl = DrmEventVblank {
            base: ev,
            user_data: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            tv_sec: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
            tv_usec: u32::from_le_bytes(buf[20..24].try_into().unwrap()),
            sequence: u32::from_le_bytes(buf[24..28].try_into().unwrap()),
            crtc_id: u32::from_le_bytes(buf[28..32].try_into().unwrap()),
        };
        let mono_us = (vbl.tv_sec as u64)
            .saturating_mul(1_000_000)
            .saturating_add(vbl.tv_usec as u64);
        Ok(mono_us)
    }
}

impl Drop for KmsDisplay {
    fn drop(&mut self) {
        let fd = self.file.as_raw_fd();
        if self.cur_fb != 0 {
            let _ = unsafe { ioctl(fd, DRM_IOCTL_MODE_RMFB, &self.cur_fb) };
        }
        if !self.cur_bo.is_null() {
            unsafe { (self.gbm.release_buffer)(self.gbm.surface, self.cur_bo) };
        }
        if !self.egl.surface.is_null() {
            unsafe { (self.egl.destroy_surface)(self.egl.display, self.egl.surface) };
        }
        if !self.egl.context.is_null() {
            unsafe { (self.egl.destroy_context)(self.egl.display, self.egl.context) };
        }
        if !self.egl.display.is_null() {
            unsafe { (self.egl.terminate)(self.egl.display) };
        }
        if !self.gbm.surface.is_null() {
            unsafe { (self.gbm.surface_destroy)(self.gbm.surface) };
        }
        if !self.gbm.device.is_null() {
            unsafe { (self.gbm.device_destroy)(self.gbm.device) };
        }
        let _ = unsafe { ioctl(fd, DRM_IOCTL_DROP_MASTER, 0u32) };
    }
}

/// Scans a card's connectors and returns the best (connector, crtc, mode).
fn find_best_connector(fd: c_int) -> Result<(u32, u32, DrmModeModeInfo), String> {
    let mut res = DrmModeCardRes::default();
    let mut r = unsafe { ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) };
    if r != 0 {
        return Err(format!("GETRESOURCES: {}", std::io::Error::last_os_error()));
    }
    let mut conn_ids = vec![0u32; res.count_connectors as usize];
    let mut crtc_ids = vec![0u32; res.count_crtcs as usize];
    let mut enc_ids = vec![0u32; res.count_encoders as usize];
    let mut fb_ids = vec![0u32; res.count_fbs as usize];
    // The kernel fills every id array whose count is non-zero, so all four
    // pointers must be valid on the second call (NULL -> EFAULT).
    res.connector_id_ptr = conn_ids.as_mut_ptr() as u64;
    res.crtc_id_ptr = crtc_ids.as_mut_ptr() as u64;
    res.encoder_id_ptr = enc_ids.as_mut_ptr() as u64;
    res.fb_id_ptr = fb_ids.as_mut_ptr() as u64;
    r = unsafe { ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) };
    if r != 0 {
        return Err(format!("GETRESOURCES(2): {}", std::io::Error::last_os_error()));
    }

    let mut best: Option<(u32, u32, DrmModeModeInfo, bool)> = None; // (.., has_120)
    for &cid in &conn_ids {
        let mut conn = DrmModeGetConnector {
            connector_id: cid,
            ..Default::default()
        };
        r = unsafe { ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &mut conn) };
        if r != 0 {
            continue;
        }
        if conn.connection != DRM_MODE_CONNECTED || conn.count_modes == 0 {
            continue;
        }
        let mut modes = vec![DrmModeModeInfo::default(); conn.count_modes as usize];
        conn.modes_ptr = modes.as_mut_ptr() as u64;
        // The fill call also iterates props and encoders; we don't need them,
        // so zero their counts to keep the kernel from writing to NULL ptrs.
        conn.props_ptr = 0;
        conn.prop_values_ptr = 0;
        conn.count_props = 0;
        conn.encoders_ptr = 0;
        conn.count_encoders = 0;
        r = unsafe { ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &mut conn) };
        if r != 0 {
            continue;
        }
        let mode = pick_mode(&modes);
        if mode.is_none() {
            continue;
        }
        let mode = mode.unwrap();
        let has_120 = mode.vrefresh >= 100 && mode.vrefresh <= 130;
        let better = match &best {
            Some((_, _, bm, b120)) => (has_120 && !b120) || (has_120 == *b120 && mode.vrefresh > bm.vrefresh),
            None => true,
        };
        if better {
            best = Some((cid, 0, mode, has_120));
        }
    }

    let (connector_id, _, mode, _) = best.ok_or("no connected connector with modes")?;

    // Route the connector to a CRTC properly: walk the connector's encoders
    // and pick a CRTC it can drive (its currently-bound crtc_id or any CRTC in
    // its possible_crtcs bitmask). SETCRTC will fail with EINVAL otherwise.
    let crtc_id = pick_crtc(fd, connector_id)?;
    Ok((connector_id, crtc_id, mode))
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DrmModeEncoder {
    encoder_id: u32,
    encoder_type: u32,
    crtc_id: u32,
    possible_crtcs: u32,
    possible_clones: u32,
    pad: u32,
}

/// Chooses a CRTC that the connector's encoders can route to. Returns the
/// kernel CRTC object id (from GETRESOURCES), which is what SETCRTC expects.
fn pick_crtc(fd: c_int, connector_id: u32) -> Result<u32, String> {
    // Card CRTC object ids (GETRESOURCES, two-phase with all ptrs set).
    let mut res = DrmModeCardRes::default();
    if unsafe { ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) } != 0 {
        return Err(format!("GETRESOURCES: {}", std::io::Error::last_os_error()));
    }
    let mut crtc_ids = vec![0u32; res.count_crtcs as usize];
    let mut conn_ids = vec![0u32; res.count_connectors as usize];
    let mut enc_ids_all = vec![0u32; res.count_encoders as usize];
    let mut fb_ids = vec![0u32; res.count_fbs as usize];
    res.crtc_id_ptr = crtc_ids.as_mut_ptr() as u64;
    res.connector_id_ptr = conn_ids.as_mut_ptr() as u64;
    res.encoder_id_ptr = enc_ids_all.as_mut_ptr() as u64;
    res.fb_id_ptr = fb_ids.as_mut_ptr() as u64;
    if unsafe { ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) } != 0 {
        return Err(format!("GETRESOURCES(2): {}", std::io::Error::last_os_error()));
    }

    let mut conn = DrmModeGetConnector {
        connector_id,
        ..Default::default()
    };
    if unsafe { ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &mut conn) } != 0 {
        return Err(format!("GETCONNECTOR: {}", std::io::Error::last_os_error()));
    }
    let mut enc_ids = vec![0u32; conn.count_encoders as usize];
    if conn.count_encoders == 0 && conn.encoder_id != 0 {
        enc_ids.push(conn.encoder_id);
    }
    conn.encoders_ptr = enc_ids.as_mut_ptr() as u64;
    // Skip the modes/props arrays we don't need (their counts are non-zero
    // from the first call, and the kernel would write to their NULL ptrs).
    conn.modes_ptr = 0;
    conn.count_modes = 0;
    conn.props_ptr = 0;
    conn.prop_values_ptr = 0;
    conn.count_props = 0;
    if unsafe { ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &mut conn) } != 0 {
        return Err(format!("GETCONNECTOR(encoders): {}", std::io::Error::last_os_error()));
    }

    // possible_crtcs is a bitmask over CRTC *indices*; translate each set bit
    // to the card's CRTC object id. Prefer an encoder's currently-bound CRTC.
    let mut candidates: Vec<u32> = Vec::new();
    for &eid in &enc_ids {
        let mut enc = DrmModeEncoder {
            encoder_id: eid,
            ..Default::default()
        };
        if unsafe { ioctl(fd, DRM_IOCTL_MODE_GETENCODER, &mut enc) } != 0 {
            continue;
        }
        if enc.crtc_id != 0 {
            return Ok(enc.crtc_id);
        }
        for bit in 0..32 {
            if enc.possible_crtcs & (1 << bit) != 0 {
                if let Some(&id) = crtc_ids.get(bit as usize) {
                    if id != 0 && !candidates.contains(&id) {
                        candidates.push(id);
                    }
                }
            }
        }
    }
    candidates
        .first()
        .copied()
        .or_else(|| crtc_ids.first().copied())
        .filter(|&id| id != 0)
        .ok_or("connector has no encoder usable for SETCRTC".into())
}

/// Picks a mode: preferred if present, else a ~120 Hz mode, else the first.
fn pick_mode(modes: &[DrmModeModeInfo]) -> Option<DrmModeModeInfo> {
    let mut preferred: Option<DrmModeModeInfo> = None;
    let mut hz120: Option<DrmModeModeInfo> = None;
    for m in modes {
        if m.r#type & DRM_MODE_TYPE_PREFERRED != 0 && preferred.is_none() {
            preferred = Some(*m);
        }
        if m.vrefresh >= 100 && m.vrefresh <= 130 {
            let better = match &hz120 {
                Some(bm) => m.vrefresh > bm.vrefresh,
                None => true,
            };
            if better {
                hz120 = Some(*m);
            }
        }
    }
    // For the stereo rig we need ~120 Hz (3D Vision rate); prefer a ~120 Hz
    // mode over the EDID "preferred" one (projectors often advertise 60 Hz as
    // preferred), then the preferred mode, then the first.
    hz120.or(preferred).or_else(|| modes.first().copied())
}

fn init_egl(
    egl_lib: &Library,
    gbm_device: *mut c_void,
    gbm_surface: *mut c_void,
) -> Result<EglFns, String> {
    let get_error: unsafe extern "C" fn() -> EglEnum =
        sym(egl_lib, b"eglGetError\0").ok_or("libEGL missing eglGetError")?;
    let get_proc_addr: unsafe extern "C" fn(*const c_char) -> *const c_void =
        sym(egl_lib, b"eglGetProcAddress\0").ok_or("libEGL missing eglGetProcAddress")?;
    let initialize: unsafe extern "C" fn(EglDisplay, *mut EglInt, *mut EglInt) -> EglBoolean =
        sym(egl_lib, b"eglInitialize\0").ok_or("libEGL missing eglInitialize")?;
    let bind_api: unsafe extern "C" fn(EglEnum) -> EglBoolean =
        sym(egl_lib, b"eglBindAPI\0").ok_or("libEGL missing eglBindAPI")?;
    let choose_config: unsafe extern "C" fn(EglDisplay, *const EglInt, *mut EglConfig, EglInt, *mut EglInt) -> EglBoolean =
        sym(egl_lib, b"eglChooseConfig\0").ok_or("libEGL missing eglChooseConfig")?;
    let create_context: unsafe extern "C" fn(EglDisplay, EglConfig, EglContext, *const EglInt) -> EglContext =
        sym(egl_lib, b"eglCreateContext\0").ok_or("libEGL missing eglCreateContext")?;
    let make_current: unsafe extern "C" fn(EglDisplay, EglSurface, EglSurface, EglContext) -> EglBoolean =
        sym(egl_lib, b"eglMakeCurrent\0").ok_or("libEGL missing eglMakeCurrent")?;
    let swap_buffers: unsafe extern "C" fn(EglDisplay, EglSurface) -> EglBoolean =
        sym(egl_lib, b"eglSwapBuffers\0").ok_or("libEGL missing eglSwapBuffers")?;
    let swap_interval: unsafe extern "C" fn(EglDisplay, EglInt) -> EglBoolean =
        sym(egl_lib, b"eglSwapInterval\0").ok_or("libEGL missing eglSwapInterval")?;
    let query_surface: unsafe extern "C" fn(EglDisplay, EglSurface, EglInt, *mut EglInt) -> EglBoolean =
        sym(egl_lib, b"eglQuerySurface\0").ok_or("libEGL missing eglQuerySurface")?;
    let terminate: unsafe extern "C" fn(EglDisplay) -> EglBoolean =
        sym(egl_lib, b"eglTerminate\0").ok_or("libEGL missing eglTerminate")?;
    let destroy_surface: unsafe extern "C" fn(EglDisplay, EglSurface) -> EglBoolean =
        sym(egl_lib, b"eglDestroySurface\0").ok_or("libEGL missing eglDestroySurface")?;
    let destroy_context: unsafe extern "C" fn(EglDisplay, EglContext) -> EglBoolean =
        sym(egl_lib, b"eglDestroyContext\0").ok_or("libEGL missing eglDestroyContext")?;

    // EGL_EXT_platform_base (resolved by name) with the GBM platform.
    let get_platform_display: Option<unsafe extern "C" fn(EglEnum, *mut c_void, *const EglInt) -> EglDisplay> =
        unsafe { egl_lib.get(b"eglGetPlatformDisplayEXT\0") }
            .ok()
            .map(|s: Symbol<unsafe extern "C" fn(EglEnum, *mut c_void, *const EglInt) -> EglDisplay>| *s)
            .or_else(|| {
                // Fall back to resolving through eglGetProcAddress.
                let name = std::ffi::CString::new("eglGetPlatformDisplayEXT").unwrap();
                let p = unsafe { (get_proc_addr)(name.as_ptr()) };
                if p.is_null() {
                    None
                } else {
                    Some(unsafe { mem::transmute::<*const c_void, _>(p) })
                }
            });
    let create_platform_surface: Option<unsafe extern "C" fn(EglDisplay, EglConfig, *mut c_void, *const EglInt) -> EglSurface> =
        unsafe { egl_lib.get(b"eglCreatePlatformWindowSurfaceEXT\0") }
            .ok()
            .map(|s: Symbol<unsafe extern "C" fn(EglDisplay, EglConfig, *mut c_void, *const EglInt) -> EglSurface>| *s)
            .or_else(|| {
                let name = std::ffi::CString::new("eglCreatePlatformWindowSurfaceEXT").unwrap();
                let p = unsafe { (get_proc_addr)(name.as_ptr()) };
                if p.is_null() {
                    None
                } else {
                    Some(unsafe { mem::transmute::<*const c_void, _>(p) })
                }
            });
    let create_window_surface: unsafe extern "C" fn(EglDisplay, EglConfig, *mut c_void, *const EglInt) -> EglSurface =
        match sym(egl_lib, b"eglCreateWindowSurface\0") {
            Some(f) => f,
            None => create_platform_surface
                .ok_or("no eglCreateWindowSurface or platform surface EXT available")?,
        };

    let display = match get_platform_display {
        Some(f) => unsafe { f(EGL_PLATFORM_GBM_KHR, gbm_device, std::ptr::null()) },
        None => {
            let egl_get_display: unsafe extern "C" fn(*mut c_void) -> EglDisplay =
                sym(egl_lib, b"eglGetDisplay\0").ok_or("libEGL missing eglGetDisplay")?;
            unsafe { egl_get_display(gbm_device) }
        }
    };
    if display.is_null() {
        return Err(format!("eglGetDisplay failed: {:#x}", unsafe { (get_error)() }));
    }
    let mut major: EglInt = 0;
    let mut minor: EglInt = 0;
    if unsafe { initialize(display, &mut major, &mut minor) } != EGL_TRUE {
        return Err(format!("eglInitialize failed: {:#x}", unsafe { (get_error)() }));
    }
    if unsafe { bind_api(EGL_OPENGL_API) } != EGL_TRUE {
        return Err(format!("eglBindAPI(EGL_OPENGL_API) failed: {:#x}", unsafe { (get_error)() }));
    }

    // Config: RGB888 window surface with a depth buffer, desktop OpenGL.
    // The GBM surface is created in GBM_FORMAT_XRGB8888, so (per
    // EGL_MESA_platform_gbm) the chosen config's EGL_NATIVE_VISUAL_ID must
    // match that GBM format or surface creation fails (EGL_BAD_NATIVE_WINDOW /
    // EGL_BAD_MATCH). Enumerate and pick the matching config explicitly.
    let config_attrs: [EglInt; 13] = [
        EGL_SURFACE_TYPE,
        EGL_WINDOW_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_DEPTH_SIZE,
        24,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_BIT,
        EGL_NONE,
    ];
    let get_config_attrib: unsafe extern "C" fn(EglDisplay, EglConfig, EglInt, *mut EglInt) -> EglBoolean =
        sym(egl_lib, b"eglGetConfigAttrib\0").ok_or("libEGL missing eglGetConfigAttrib")?;
    let mut num_config: EglInt = 0;
    if unsafe { choose_config(display, config_attrs.as_ptr(), std::ptr::null_mut(), 0, &mut num_config) } != EGL_TRUE
        || num_config < 1
    {
        return Err(format!("eglChooseConfig failed: {:#x}", unsafe { (get_error)() }));
    }
    let mut configs: Vec<EglConfig> = vec![std::ptr::null(); num_config as usize];
    if unsafe { choose_config(display, config_attrs.as_ptr(), configs.as_mut_ptr(), num_config, &mut num_config) } != EGL_TRUE {
        return Err(format!("eglChooseConfig(2) failed: {:#x}", unsafe { (get_error)() }));
    }
    let mut config: EglConfig = configs[0];
    for &candidate in &configs {
        let mut visual: EglInt = 0;
        if unsafe { get_config_attrib(display, candidate, EGL_NATIVE_VISUAL_ID, &mut visual) } == EGL_TRUE
            && visual as u32 == GBM_FORMAT_XRGB8888
        {
            config = candidate;
            break;
        }
    }
    eprintln!("nvstusb: EGL config native visual {:#x}", {
        let mut visual: EglInt = 0;
        if unsafe { get_config_attrib(display, config, EGL_NATIVE_VISUAL_ID, &mut visual) } == EGL_TRUE {
            visual as u32
        } else {
            0
        }
    });

    // Try a 2.1 compatibility context; fall back to the driver default. The
    // API is already set by eglBindAPI(EGL_OPENGL_API); EGL_CONTEXT_CLIENT_TYPE
    // is rejected by EGL 1.5+ drivers, so it is omitted here.
    let ctx_attrs: [EglInt; 5] = [
        EGL_CONTEXT_MAJOR_VERSION,
        2,
        EGL_CONTEXT_MINOR_VERSION,
        1,
        EGL_NONE,
    ];
    let mut context = unsafe { create_context(display, config, EGL_NO_CONTEXT, ctx_attrs.as_ptr()) };
    if context.is_null() {
        let plain: [EglInt; 1] = [EGL_NONE];
        context = unsafe { create_context(display, config, EGL_NO_CONTEXT, plain.as_ptr()) };
    }
    if context.is_null() {
        return Err(format!("eglCreateContext failed: {:#x}", unsafe { (get_error)() }));
    }

    // Create the EGL surface from the GBM surface (native window handle).
    // The legacy eglCreateWindowSurface is the canonical GBM path (kmscube);
    // the EXT platform entry point is only a fallback.
    let surface = unsafe { create_window_surface(display, config, gbm_surface, std::ptr::null()) };
    let mut last_err = unsafe { (get_error)() };
    let mut surface = surface;
    if surface.is_null() {
        if let Some(f) = create_platform_surface {
            let s = unsafe { f(display, config, gbm_surface, std::ptr::null()) };
            last_err = unsafe { (get_error)() };
            surface = s;
        }
    }
    if surface.is_null() {
        unsafe { (destroy_context)(display, context) };
        return Err(format!("eglCreateWindowSurface failed: {:#x}", last_err));
    }

    // NOTE: `eglSwapInterval` is NOT called here. Per the EGL spec the
    // interval applies to "the current context's draw surface" - calling it
    // before any context is current makes Mesa reject it with
    // EGL_BAD_PARAMETER regardless of the requested value, which we misread
    // as "this driver throttles unconditionally". The real request (and the
    // min/max range query) happens in [`KmsDisplay::make_current`], where a
    // context is current and the answer is authoritative.
    //
    // Provisional until make_current decides: assume throttled, matching the
    // behavior the old ordering accidentally produced, so the eye inversion
    // starts from the historically-correct assumption either way.
    let vsync_throttled = true;

    Ok(EglFns {
        display,
        surface,
        context,
        get_error,
        swap_buffers,
        swap_interval,
        make_current,
        query_surface,
        get_proc_addr,
        terminate,
        destroy_surface,
        destroy_context,
        vsync_throttled,
    })
}
