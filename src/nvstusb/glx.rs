//! Runtime loading of the GLX_SGI_video_sync / GLX_SGI_swap_control /
//! GLX_EXT_swap_control extension entry points.
//!
//! This mirrors how the original C code resolves these function pointers at
//! startup through `glXGetProcAddress`, without requiring us to link against
//! any GLX headers.

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_ulong, c_void};
use std::mem;

pub type GlXGetVideoSyncFn = unsafe extern "C" fn(count: *mut u32) -> c_int;
pub type GlXWaitVideoSyncFn =
    unsafe extern "C" fn(divisor: c_int, remainder: c_int, count: *mut u32) -> c_int;
pub type GlXSwapIntervalSGIFn = unsafe extern "C" fn(interval: c_int) -> c_int;
pub type GlXSwapIntervalEXTFn =
    unsafe extern "C" fn(display: *mut c_void, drawable: c_ulong, interval: c_int) -> c_int;
/// GLX_OML_sync_control: returns the master sync values (UST/MSC/SBC) for a
/// drawable.  UST (unadjusted system time, ns) is the timestamp of the last
/// vblank the counter (MSC) reached; on Xorg the MSC tracks the CRTC vblank
/// clock regardless of the window, so this is a real display-clock anchor that
/// needs no /dev/dri access.
pub type GlXGetSyncValuesOMLFn = unsafe extern "C" fn(
    display: *mut c_void,
    drawable: c_ulong,
    ust: *mut i64,
    msc: *mut i64,
    sbc: *mut i64,
) -> c_int;
/// GLX_OML_sync_control: blocks until the MSC satisfies
/// `msc >= target_msc && msc % divisor == remainder`, returning the UST/MSC
/// reached.  With (0, 1, 0) this blocks until the next vblank.
pub type GlXWaitForMscOMLFn = unsafe extern "C" fn(
    display: *mut c_void,
    drawable: c_ulong,
    target_msc: i64,
    divisor: i64,
    remainder: i64,
    ust: *mut i64,
    msc: *mut i64,
    sbc: *mut i64,
) -> c_int;

/// The GLX extension functions we care about, all resolved at runtime.
pub struct GlxExtensions {
    /// Keep libGL.so.1 loaded so the resolved function pointers stay valid.
    _library: Option<Library>,
    pub get_video_sync_sgi: Option<GlXGetVideoSyncFn>,
    pub wait_video_sync_sgi: Option<GlXWaitVideoSyncFn>,
    pub swap_interval_sgi: Option<GlXSwapIntervalSGIFn>,
    pub swap_interval_ext: Option<GlXSwapIntervalEXTFn>,
    pub get_sync_values_oml: Option<GlXGetSyncValuesOMLFn>,
    pub wait_for_msc_oml: Option<GlXWaitForMscOMLFn>,
}

impl GlxExtensions {
    pub fn load() -> Self {
        let mut out = Self {
            _library: None,
            get_video_sync_sgi: None,
            wait_video_sync_sgi: None,
            swap_interval_sgi: None,
            swap_interval_ext: None,
            get_sync_values_oml: None,
            wait_for_msc_oml: None,
        };

        let library = match unsafe { Library::new("libGL.so.1") } {
            Ok(library) => library,
            Err(_) => {
                eprintln!("nvstusb: could not load libGL.so.1");
                return out;
            }
        };

        // Prefer the ARB entry point, falling back to plain glXGetProcAddress.
        let get_proc_addr: Option<unsafe extern "C" fn(*const c_char) -> *const c_void> =
            unsafe { library.get(b"glXGetProcAddressARB\0") }
                .or_else(|_| unsafe { library.get(b"glXGetProcAddress\0") })
                .ok()
                .map(|s: Symbol<unsafe extern "C" fn(*const c_char) -> *const c_void>| *s);

        if let Some(get_proc_addr) = get_proc_addr {
            out.get_video_sync_sgi = resolve(get_proc_addr, "glXGetVideoSyncSGI");
            out.wait_video_sync_sgi = resolve(get_proc_addr, "glXWaitVideoSyncSGI");
            out.swap_interval_sgi = resolve(get_proc_addr, "glXSwapIntervalSGI");
            out.swap_interval_ext = resolve(get_proc_addr, "glXSwapIntervalEXT");
            out.get_sync_values_oml = resolve(get_proc_addr, "glXGetSyncValuesOML");
            out.wait_for_msc_oml = resolve(get_proc_addr, "glXWaitForMscOML");
        } else {
            // No proc-address resolver: fall back to direct symbol lookup.
            out.get_video_sync_sgi = unsafe { library.get(b"glXGetVideoSyncSGI\0") }
                .ok()
                .map(|s| *s);
            out.wait_video_sync_sgi = unsafe { library.get(b"glXWaitVideoSyncSGI\0") }
                .ok()
                .map(|s| *s);
            out.swap_interval_sgi = unsafe { library.get(b"glXSwapIntervalSGI\0") }
                .ok()
                .map(|s| *s);
            out.swap_interval_ext = unsafe { library.get(b"glXSwapIntervalEXT\0") }
                .ok()
                .map(|s| *s);
            out.get_sync_values_oml = unsafe { library.get(b"glXGetSyncValuesOML\0") }
                .ok()
                .map(|s| *s);
            out.wait_for_msc_oml = unsafe { library.get(b"glXWaitForMscOML\0") }
                .ok()
                .map(|s| *s);
        }

        out._library = Some(library);
        out
    }
}

/// Resolves a named extension function through `glXGetProcAddress`, returning
/// `None` when the extension isn't provided by the driver.
fn resolve<T: Copy>(
    get_proc_addr: unsafe extern "C" fn(*const c_char) -> *const c_void,
    name: &str,
) -> Option<T> {
    let name = match std::ffi::CString::new(name) {
        Ok(name) => name,
        Err(_) => return None,
    };
    let ptr = unsafe { get_proc_addr(name.as_ptr()) };
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { mem::transmute_copy(&ptr) })
}
