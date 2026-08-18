//! Creates a tiny hidden X11 window with its own GLX context, used by the
//! optional stereo thread so it can drive the emitter on `GL_STEREO`
//! (quad-buffered) setups without disturbing the main rendering context.
#![allow(dead_code)]
//!
//! Port of the context-creation code inside `nvstusb_stereo_thread()` from
//! `lib/nvstusb.c`.

use crate::gl;
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void};
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

// GLX attribute constants from <GL/glx.h>.
const GLX_USE_GL: c_int = 1;
const GLX_BUFFER_SIZE: c_int = 2;
const GLX_RGBA: c_int = 4;
const GLX_DOUBLEBUFFER: c_int = 5;
const GLX_RED_SIZE: c_int = 8;
const GLX_GREEN_SIZE: c_int = 9;
const GLX_BLUE_SIZE: c_int = 10;
const GLX_NONE: c_int = 0;

// Xlib constants.
const INPUT_OUTPUT: c_uint = 1;
const ALLOC_NONE: c_int = 0;
const TRUE_: c_int = 1;
const CW_COLORMAP: c_ulong = 1 << 2;
const CW_OVERRIDE_REDIRECT: c_ulong = 1 << 9;

/// `XVisualInfo` from <X11/Xutil.h>.
#[repr(C)]
#[derive(Clone, Copy)]
struct XVisualInfo {
    visual: *mut c_void,
    visualid: c_ulong,
    screen: c_int,
    depth: c_int,
    class_: c_int,
    red_mask: c_ulong,
    green_mask: c_ulong,
    blue_mask: c_ulong,
    colormap_size: c_int,
    bits_per_rgb: c_int,
}

/// `XSetWindowAttributes` from <X11/Xlib.h> (fields we touch only).
#[repr(C)]
#[derive(Clone, Copy)]
struct XSetWindowAttributes {
    background_pixmap: c_ulong,
    background_pixel: c_ulong,
    border_pixmap: c_ulong,
    border_pixel: c_ulong,
    bit_gravity: c_int,
    win_gravity: c_int,
    backing_store: c_int,
    backing_planes: c_ulong,
    backing_pixel: c_ulong,
    save_under: c_int,
    event_mask: c_long,
    do_not_propagate_mask: c_long,
    override_redirect: c_int,
    colormap: c_ulong,
    cursor: c_ulong,
}

type XOpenDisplayFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type XCloseDisplayFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type XSyncFn = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type XSetErrorHandlerFn = unsafe extern "C" fn(*const c_void) -> *mut c_void;
static X_ERROR: AtomicBool = AtomicBool::new(false);
unsafe extern "C" fn swallow_x_error(_display: *mut c_void, _event: *const c_void) -> c_int {
    X_ERROR.store(true, Ordering::Relaxed);
    0
}
type XDefaultScreenFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type XRootWindowFn = unsafe extern "C" fn(*mut c_void, c_int) -> c_ulong;
type XCreateColormapFn = unsafe extern "C" fn(*mut c_void, c_ulong, *mut c_void, c_int) -> c_ulong;
type XCreateWindowFn = unsafe extern "C" fn(
    *mut c_void,
    c_ulong,
    c_int,
    c_int,
    c_uint,
    c_uint,
    c_uint,
    c_int,
    c_uint,
    *mut c_void,
    c_ulong,
    *mut XSetWindowAttributes,
) -> c_ulong;
type XMapWindowFn = unsafe extern "C" fn(*mut c_void, c_ulong);
type XFreeFn = unsafe extern "C" fn(*mut c_void) -> c_int;

type GlXChooseVisualFn = unsafe extern "C" fn(*mut c_void, c_int, *const c_int) -> *mut XVisualInfo;
type GlXCreateContextFn =
    unsafe extern "C" fn(*mut c_void, *mut XVisualInfo, *mut c_void, c_int) -> *mut c_void;
type GlXMakeCurrentFn = unsafe extern "C" fn(*mut c_void, c_ulong, *mut c_void) -> c_int;
type GlXDestroyContextFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
type GlXGetProcAddressFn = unsafe extern "C" fn(*const c_char) -> *const c_void;

/// A hidden window with a current GLX context, plus a `gl::Gl` loaded from it.
pub struct HiddenGlx {
    _x11: Library,
    _glx: Library,
    display: *mut c_void,
    window: c_ulong,
    context: *mut c_void,
    pub gl: gl::Gl,
}

unsafe impl Send for HiddenGlx {}

impl HiddenGlx {
    /// Creates a 1x1 hidden window and makes its GLX context current.
    ///
    /// Returns `None` when there is no usable X server / GLX (e.g. headless).
    pub fn create() -> Option<Self> {
        let x11 = unsafe { Library::new("libX11.so.6") }.ok()?;
        let glx_lib = unsafe { Library::new("libGL.so.1") }.ok()?;

        unsafe {
            let x_open_display: Symbol<XOpenDisplayFn> = x11.get(b"XOpenDisplay\0").ok()?;
            let x_sync: Symbol<XSyncFn> = x11.get(b"XSync\0").ok()?;
            let x_close_display: Symbol<XCloseDisplayFn> = x11.get(b"XCloseDisplay\0").ok()?;
            let x_default_screen: Symbol<XDefaultScreenFn> = x11.get(b"XDefaultScreen\0").ok()?;
            let x_root_window: Symbol<XRootWindowFn> = x11.get(b"XRootWindow\0").ok()?;
            let x_create_colormap: Symbol<XCreateColormapFn> =
                x11.get(b"XCreateColormap\0").ok()?;
            let x_create_window: Symbol<XCreateWindowFn> = x11.get(b"XCreateWindow\0").ok()?;
            let x_map_window: Symbol<XMapWindowFn> = x11.get(b"XMapWindow\0").ok()?;
            let x_free: Symbol<XFreeFn> = x11.get(b"XFree\0").ok()?;
            let x_set_error_handler: Symbol<XSetErrorHandlerFn> =
                x11.get(b"XSetErrorHandler\0").ok()?;

            let glx_choose_visual: Symbol<GlXChooseVisualFn> =
                glx_lib.get(b"glXChooseVisual\0").ok()?;
            let glx_create_context: Symbol<GlXCreateContextFn> =
                glx_lib.get(b"glXCreateContext\0").ok()?;
            let glx_make_current: Symbol<GlXMakeCurrentFn> =
                glx_lib.get(b"glXMakeCurrent\0").ok()?;
            let glx_get_proc_address: Symbol<GlXGetProcAddressFn> = glx_lib
                .get(b"glXGetProcAddressARB\0")
                .or_else(|_| glx_lib.get(b"glXGetProcAddress\0"))
                .ok()?;

            // Some drivers/toolkits (Mesa + winit, or an existing GL context
            // current on this thread) reject a second glXMakeCurrent with
            // BadAccess, which would abort the whole app via the default X
            // error handler.  Swallow X errors during hidden-window setup and
            // return None (anchor falls back) instead of crashing.
            let prev_handler = x_set_error_handler(swallow_x_error as *const c_void);
            X_ERROR.store(false, Ordering::Relaxed);

            let display = x_open_display(std::ptr::null());
            if display.is_null() {
                x_set_error_handler(prev_handler);
                return None;
            }

            let screen = x_default_screen(display);
            let attribute_list: [c_int; 9] = [
                GLX_RGBA,
                GLX_DOUBLEBUFFER,
                GLX_RED_SIZE,
                1,
                GLX_GREEN_SIZE,
                1,
                GLX_BLUE_SIZE,
                1,
                GLX_NONE,
            ];

            let visual = glx_choose_visual(display, screen, attribute_list.as_ptr());
            if visual.is_null() {
                x_close_display(display);
                x_set_error_handler(prev_handler);
                return None;
            }

            let root = x_root_window(display, screen);
            let mut attrs: XSetWindowAttributes = mem::zeroed();
            attrs.colormap = x_create_colormap(display, root, (*visual).visual, ALLOC_NONE);
            attrs.override_redirect = TRUE_;

            let window = x_create_window(
                display,
                root,
                0,
                0,
                1,
                1,
                0,
                (*visual).depth,
                INPUT_OUTPUT,
                (*visual).visual,
                CW_COLORMAP | CW_OVERRIDE_REDIRECT,
                &mut attrs,
            );
            x_map_window(display, window);

            let context = glx_create_context(display, visual, std::ptr::null_mut(), TRUE_);
            x_free(visual as *mut c_void);
            glx_make_current(display, window, context);

            let gl = gl::Gl::load_with(|name| {
                let name = std::ffi::CString::new(name).unwrap();
                glx_get_proc_address(name.as_ptr())
            });

            // Force delivery of any queued protocol errors (e.g. BadAccess on
            // MakeCurrent) so X_ERROR reflects reality before we decide.
            x_sync(display, 0);
            let x_error = X_ERROR.load(Ordering::Relaxed);
            x_set_error_handler(prev_handler);
            if x_error {
                // Some X/GLX call failed (e.g. BadAccess on MakeCurrent because
                // this thread already has a GL context current).  Clean up and
                // report unavailability so the caller falls back.
                eprintln!("nvstusb: X/GLX error during hidden-window setup; OML anchor unavailable");
                if let Ok(glx_destroy_context) =
                    glx_lib.get::<GlXDestroyContextFn>(b"glXDestroyContext\0")
                {
                    glx_destroy_context(display, context);
                }
                x_close_display(display);
                return None;
            }

            Some(Self {
                _x11: x11,
                _glx: glx_lib,
                display,
                window,
                context,
                gl,
            })
        }
    }
}

impl Drop for HiddenGlx {
    fn drop(&mut self) {
        unsafe {
            if let Ok(glx_destroy_context) =
                self._glx.get::<GlXDestroyContextFn>(b"glXDestroyContext\0")
            {
                glx_destroy_context(self.display, self.context);
            }
            if let Ok(x_close_display) = self._x11.get::<XCloseDisplayFn>(b"XCloseDisplay\0") {
                x_close_display(self.display);
            }
        }
    }
}

type GlXGetSyncValuesOMLFn = unsafe extern "C" fn(
    *mut c_void,
    c_ulong,
    *mut i64,
    *mut i64,
    *mut i64,
) -> c_int;
type GlXWaitForMscOMLFn = unsafe extern "C" fn(
    *mut c_void,
    c_ulong,
    i64,
    i64,
    i64,
    *mut i64,
    *mut i64,
    *mut i64,
) -> c_int;
type GlXGetCurrentDisplayFn = unsafe extern "C" fn() -> *mut c_void;
type GlXGetCurrentDrawableFn = unsafe extern "C" fn() -> c_ulong;

// EGL equivalents (EGL_CHROMIUM_sync_control, Mesa exposes it for the X11 and
// Wayland platforms): returns the UST/MSC of the given surface.
type EglGetCurrentDisplayFn = unsafe extern "C" fn() -> *mut c_void;
type EglGetCurrentSurfaceFn = unsafe extern "C" fn(c_uint) -> *mut c_void;
type EglGetSyncValuesFn =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u64, *mut u64, *mut u64) -> c_int;
const EGL_DRAW: c_uint = 0x3059;

/// Real display vblank clock read off the *current* GLX or EGL context, so no
/// new window/context has to be created (creating one while winit's context is
/// current triggers BadAccess on some Mesa setups).
///
/// Both GLX_OML_sync_control and EGL_CHROMIUM_sync_control return a per-screen
/// master counter (MSC) and its timestamp (UST) that track the CRTC vblank
/// clock (on Xorg with DRI3/Present), giving a true display-boundary anchor
/// without any /dev/dri access.
pub struct MscAnchor {
    backend: MscBackend,
    /// Instant / UST reference pair to convert UST (ns) into an `Instant`
    /// on the same clock the paced stream uses for its sleep/spin deadlines.
    base_instant: Instant,
    base_ust: i64,
    last_msc: i64,
    last_ust: i64,
    period_us: u64,
}

enum MscBackend {
    Glx {
        get: GlXGetSyncValuesOMLFn,
        wait: Option<GlXWaitForMscOMLFn>,
        display: *mut c_void,
        drawable: c_ulong,
    },
    Egl {
        get: EglGetSyncValuesFn,
        display: *mut c_void,
        surface: *mut c_void,
    },
}

unsafe impl Send for MscAnchor {}

/// The emitter's packet period bounds, shared with the DRM anchor.
const PERIOD_MIN_US: u64 = 7_600;
const PERIOD_MAX_US: u64 = 9_000;

impl MscAnchor {
    /// Captures the display/drawable of the *currently current* GLX or EGL
    /// context and resolves the sync-control entry points.  Call on the thread
    /// where the app's context is current (start_paced runs there at init).
    /// Returns `None` when there is no GLX/EGL context current or the sync
    /// extension is unavailable.
    pub fn capture() -> Option<Self> {
        // GLX first: glXGetCurrentDisplay/Drawable on the current context.
        // NOTE: must NOT abort on GLX failure (`?`); the app may be using EGL
        // (glutin prefers it), in which case GLX is simply not current and we
        // must fall through to the EGL path below.
        let glx = crate::nvstusb::glx::GlxExtensions::load();
        if let Some(get) = glx.get_sync_values_oml {
            if let Some((display, drawable)) = glx_current_handles() {
                if let Some((base_ust, base_msc)) = read_glx(display, drawable, get) {
                    eprintln!("nvstusb: OML sync via GLX");
                    return Some(Self {
                        backend: MscBackend::Glx {
                            get,
                            wait: glx.wait_for_msc_oml,
                            display,
                            drawable,
                        },
                        base_instant: Instant::now(),
                        base_ust,
                        last_msc: base_msc,
                        last_ust: base_ust,
                        period_us: 0,
                    });
                }
            }
        }

        // EGL fallback (glutin prefers EGL on X11/Wayland).
        let egl = unsafe { Library::new("libEGL.so.1") }.ok()?;
        let get_current_display: Symbol<EglGetCurrentDisplayFn> =
            unsafe { egl.get(b"eglGetCurrentDisplay\0") }.ok()?;
        let get_current_surface: Symbol<EglGetCurrentSurfaceFn> =
            unsafe { egl.get(b"eglGetCurrentSurface\0") }.ok()?;
        let display = unsafe { get_current_display() };
        let surface = unsafe { get_current_surface(EGL_DRAW) };
        if display.is_null() || surface.is_null() {
            eprintln!("nvstusb: OML capture: no current EGL display/surface");
            return None;
        }
        // eglGetSyncValuesCHROMIUM is an extension: resolve through
        // eglGetProcAddress.
        let get_proc_addr_sym: Symbol<unsafe extern "C" fn(*const c_char) -> *const c_void> =
            unsafe { egl.get(b"eglGetProcAddress\0") }.ok()?;
        let get_proc_addr = *get_proc_addr_sym;
        let name = std::ffi::CString::new("eglGetSyncValuesCHROMIUM").ok()?;
        let ptr = unsafe { get_proc_addr(name.as_ptr()) };
        let get: EglGetSyncValuesFn = if ptr.is_null() {
            eprintln!("nvstusb: OML capture: eglGetSyncValuesCHROMIUM not available");
            return None;
        } else {
            unsafe { mem::transmute(ptr) }
        };
        let (base_ust, base_msc) = match read_egl(display, surface, get) {
            Some(v) => v,
            None => {
                eprintln!("nvstusb: OML capture: EGL sync read failed (extension unsupported?)");
                return None;
            }
        };
        eprintln!("nvstusb: OML sync via EGL");
        Some(Self {
            backend: MscBackend::Egl {
                get,
                display,
                surface,
            },
            base_instant: Instant::now(),
            base_ust,
            last_msc: base_msc,
            last_ust: base_ust,
            period_us: 0,
        })
    }

    /// Measured vblank period in microseconds (0 until two vblanks observed).
    pub fn period_us(&self) -> u64 {
        self.period_us
    }

    /// Blocks (or polls) until the next vblank on the CRTC clock and returns
    /// its timestamp as an `Instant`.  Prefers a blocking wait (GLX
    /// `glXWaitForMscOML`); falls back to polling the sync values.  Returns
    /// `None` (caller falls back) if the clock never advances, so a dead
    /// XWayland/EGL implementation cannot hang the paced stream.
    pub fn wait_vblank(&mut self) -> Option<Instant> {
        if let MscBackend::Glx { wait, display, drawable, .. } = &self.backend {
            if let Some(wait) = wait {
                let mut ust: i64 = 0;
                let mut msc: i64 = 0;
                let mut sbc: i64 = 0;
                let ok = unsafe {
                    wait(*display, *drawable, 0, 1, 0, &mut ust, &mut msc, &mut sbc)
                };
                if ok != 0 && msc != self.last_msc {
                    self.observe(ust, msc);
                    return Some(self.instant_of(ust));
                }
            }
        }

        // Poll until the master counter advances.  Give up after ~100 ms so a
        // non-advancing clock degrades to the present anchor instead of
        // spinning forever.
        let deadline = Instant::now() + Duration::from_millis(100);
        let mut spins = 0;
        loop {
            let (ust, msc) = match &self.backend {
                MscBackend::Glx { get, display, drawable, .. } => {
                    read_glx(*display, *drawable, *get)?
                }
                MscBackend::Egl { get, display, surface } => {
                    read_egl(*display, *surface, *get)?
                }
            };
            if msc != self.last_msc {
                self.observe(ust, msc);
                return Some(self.instant_of(ust));
            }
            if Instant::now() >= deadline {
                return None;
            }
            if spins < 200 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                thread::sleep(Duration::from_micros(500));
                spins = 0;
            }
        }
    }

    fn instant_of(&self, ust: i64) -> Instant {
        let delta = ust.saturating_sub(self.base_ust).max(0) as u64;
        self.base_instant + Duration::from_nanos(delta)
    }

    fn observe(&mut self, ust: i64, msc: i64) {
        if msc > self.last_msc && ust > self.last_ust {
            let p = (ust - self.last_ust) as u64 / (msc - self.last_msc) as u64 / 1000;
            if (PERIOD_MIN_US..=PERIOD_MAX_US).contains(&p) {
                self.period_us = p;
            }
        }
        self.last_msc = msc;
        self.last_ust = ust;
    }
}

fn glx_current_handles() -> Option<(*mut c_void, c_ulong)> {
    let glx = unsafe { Library::new("libGL.so.1") }.ok()?;
    let get_display: Symbol<GlXGetCurrentDisplayFn> =
        unsafe { glx.get(b"glXGetCurrentDisplay\0") }.ok()?;
    let get_drawable: Symbol<GlXGetCurrentDrawableFn> =
        unsafe { glx.get(b"glXGetCurrentDrawable\0") }.ok()?;
    let display = unsafe { get_display() };
    let drawable = unsafe { get_drawable() };
    if display.is_null() || drawable == 0 {
        None
    } else {
        Some((display, drawable))
    }
}

fn read_glx(display: *mut c_void, drawable: c_ulong, get: GlXGetSyncValuesOMLFn) -> Option<(i64, i64)> {
    let mut ust: i64 = 0;
    let mut msc: i64 = 0;
    let mut sbc: i64 = 0;
    let ok = unsafe { get(display, drawable, &mut ust, &mut msc, &mut sbc) };
    if ok == 0 || (ust == 0 && msc == 0) {
        return None;
    }
    Some((ust, msc))
}

fn read_egl(display: *mut c_void, surface: *mut c_void, get: EglGetSyncValuesFn) -> Option<(i64, i64)> {
    let mut ust: u64 = 0;
    let mut msc: u64 = 0;
    let mut sbc: u64 = 0;
    let ok = unsafe { get(display, surface, &mut ust, &mut msc, &mut sbc) };
    if ok == 0 || (ust == 0 && msc == 0) {
        return None;
    }
    Some((ust as i64, msc as i64))
}
