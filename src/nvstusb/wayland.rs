//! Native Wayland rendering backend with presentation-time vblank sync.
//!
//! The windowed/winit path cannot be trusted for shutter timing: `swap_buffers`
//! only acks a commit, which can lead the real present by several vblanks, and
//! winit hides the `wl_surface` so no presentation feedback can be attached. To
//! get the compositor's *ground truth* of when each frame hit the screen we own
//! the Wayland connection ourselves (via `wayland-client`), create the
//! `wl_surface` / `xdg_toplevel` / `wl_egl_window` and EGL surface directly, and
//! ask for a `wp_presentation.feedback` on every commit.
//!
//! [`WaylandDisplay::present()`] blocks until that frame's `presented` event
//! arrives, so the render loop paces exactly at the vblank and the anchor
//! ([`WaylandPresent`]) is filled with hardware-accurate present timestamps and
//! the measured `refresh` period — the same contract the DRM anchor fills on the
//! KMS path, but for a composited desktop.
//!
//! This low-level module drives the emitter's paced stream off the
//! compositor's `wp_presentation_feedback` present anchor. The 3dv3d demo
//! itself uses the winit/glutin windowed path on Wayland and never selects
//! this backend; it remains available as library infrastructure for callers
//! that drive a surface directly.
//!
//! Uses the *system* `libwayland` (wayland-backend `client_system` feature) so a
//! real `wl_display*` exists to hand to EGL. libEGL is resolved at runtime
//! through `libloading`; libwayland / libwayland-egl are loaded by wayland-rs.

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_short, c_void};
use std::mem;
use std::os::fd::AsRawFd;
use std::os::raw::c_ulong;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::wl_compositor::WlCompositor;
use wayland_client::protocol::wl_keyboard::{self, WlKeyboard};
use wayland_client::protocol::wl_output::{self, WlOutput};
use wayland_client::protocol::wl_registry;
use wayland_client::protocol::wl_seat::{self, WlSeat};
use wayland_client::protocol::wl_surface::WlSurface;
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle};
use wayland_egl::WlEglSurface;
use wayland_protocols::wp::presentation_time::client::wp_presentation::WpPresentation;
use wayland_protocols::wp::presentation_time::client::wp_presentation_feedback::{
    Event, Kind, WpPresentationFeedback,
};
use wayland_protocols::xdg::shell::client::xdg_surface::XdgSurface;
use wayland_protocols::xdg::shell::client::xdg_toplevel::XdgToplevel;
use wayland_protocols::xdg::shell::client::xdg_wm_base::XdgWmBase;

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

const EGL_PLATFORM_WAYLAND_EXT: EglEnum = 0x31D8;
const EGL_SURFACE_TYPE: EglInt = 0x3033;
const EGL_WINDOW_BIT: EglInt = 0x0004;
const EGL_RED_SIZE: EglInt = 0x3024;
const EGL_GREEN_SIZE: EglInt = 0x3023;
const EGL_BLUE_SIZE: EglInt = 0x3022;
const EGL_DEPTH_SIZE: EglInt = 0x3025;
const EGL_RENDERABLE_TYPE: EglInt = 0x3040;
const EGL_OPENGL_BIT: EglInt = 0x0008;
const EGL_CONTEXT_CLIENT_TYPE: EglInt = 0x3097;
const EGL_CONTEXT_MAJOR_VERSION: EglInt = 0x3098;
const EGL_CONTEXT_MINOR_VERSION: EglInt = 0x30FB;
const EGL_NONE: EglInt = 0x3038;
const EGL_TRUE: EglBoolean = 1;
const EGL_OPENGL_API: EglEnum = 0x30A2;

struct EglFns {
    display: EglDisplay,
    surface: EglSurface,
    context: EglContext,
    get_error: unsafe extern "C" fn() -> EglEnum,
    swap_buffers: unsafe extern "C" fn(EglDisplay, EglSurface) -> EglBoolean,
    swap_interval: unsafe extern "C" fn(EglDisplay, EglInt) -> EglBoolean,
    make_current: unsafe extern "C" fn(EglDisplay, EglSurface, EglSurface, EglContext) -> EglBoolean,
    get_proc_addr: unsafe extern "C" fn(*const c_char) -> *const c_void,
    terminate: unsafe extern "C" fn(EglDisplay) -> EglBoolean,
    destroy_surface: unsafe extern "C" fn(EglDisplay, EglSurface) -> EglBoolean,
    destroy_context: unsafe extern "C" fn(EglDisplay, EglContext) -> EglBoolean,
}

#[repr(C)]
struct PollFd {
    fd: c_int,
    events: c_short,
    revents: c_short,
}
const POLLIN: c_short = 0x001;

unsafe extern "C" {
    fn poll(fds: *mut PollFd, nfds: c_ulong, timeout: c_int) -> c_int;
    fn clock_gettime(clockid: c_int, tp: *mut Timespec) -> c_int;
}

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}
const CLOCK_MONOTONIC: c_int = 1;

fn mono_now_ns() -> i128 {
    let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        clock_gettime(CLOCK_MONOTONIC, &mut ts);
    }
    (ts.tv_sec as i128) * 1_000_000_000 + ts.tv_nsec as i128
}

fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Option<T> {
    unsafe { lib.get(name).ok().map(|s: Symbol<T>| *s) }
}

fn init_egl(
    egl_lib: &Library,
    wl_display: *mut c_void,
    native_window: *mut c_void,
) -> Result<EglFns, String> {
    let get_error: unsafe extern "C" fn() -> EglEnum =
        sym(egl_lib, b"eglGetError\0").ok_or("libEGL missing eglGetError")?;
    let get_proc_addr: unsafe extern "C" fn(*const c_char) -> *const c_void =
        sym(egl_lib, b"eglGetProcAddress\0").ok_or("libEGL missing eglGetProcAddress")?;
    let initialize: unsafe extern "C" fn(EglDisplay, *mut EglInt, *mut EglInt) -> EglBoolean =
        sym(egl_lib, b"eglInitialize\0").ok_or("libEGL missing eglInitialize")?;
    let bind_api: unsafe extern "C" fn(EglEnum) -> EglBoolean =
        sym(egl_lib, b"eglBindAPI\0").ok_or("libEGL missing eglBindAPI")?;
    let choose_config: unsafe extern "C" fn(
        EglDisplay,
        *const EglInt,
        *mut EglConfig,
        EglInt,
        *mut EglInt,
    ) -> EglBoolean = sym(egl_lib, b"eglChooseConfig\0").ok_or("libEGL missing eglChooseConfig")?;
    let create_context: unsafe extern "C" fn(EglDisplay, EglConfig, EglContext, *const EglInt) -> EglContext =
        sym(egl_lib, b"eglCreateContext\0").ok_or("libEGL missing eglCreateContext")?;
    let make_current: unsafe extern "C" fn(
        EglDisplay,
        EglSurface,
        EglSurface,
        EglContext,
    ) -> EglBoolean = sym(egl_lib, b"eglMakeCurrent\0").ok_or("libEGL missing eglMakeCurrent")?;
    let swap_buffers: unsafe extern "C" fn(EglDisplay, EglSurface) -> EglBoolean =
        sym(egl_lib, b"eglSwapBuffers\0").ok_or("libEGL missing eglSwapBuffers")?;
    let swap_interval: unsafe extern "C" fn(EglDisplay, EglInt) -> EglBoolean =
        sym(egl_lib, b"eglSwapInterval\0").ok_or("libEGL missing eglSwapInterval")?;
    let terminate: unsafe extern "C" fn(EglDisplay) -> EglBoolean =
        sym(egl_lib, b"eglTerminate\0").ok_or("libEGL missing eglTerminate")?;
    let destroy_surface: unsafe extern "C" fn(EglDisplay, EglSurface) -> EglBoolean =
        sym(egl_lib, b"eglDestroySurface\0").ok_or("libEGL missing eglDestroySurface")?;
    let destroy_context: unsafe extern "C" fn(EglDisplay, EglContext) -> EglBoolean =
        sym(egl_lib, b"eglDestroyContext\0").ok_or("libEGL missing eglDestroyContext")?;

    // EGL_EXT_platform_base -> Wayland platform, from the real `wl_display*`.
    let get_platform_display: Option<
        unsafe extern "C" fn(EglEnum, *mut c_void, *const EglInt) -> EglDisplay,
    > = unsafe { egl_lib.get(b"eglGetPlatformDisplayEXT\0") }
        .ok()
        .map(|s: Symbol<unsafe extern "C" fn(EglEnum, *mut c_void, *const EglInt) -> EglDisplay>| *s)
        .or_else(|| {
            let name = std::ffi::CString::new("eglGetPlatformDisplayEXT").unwrap();
            let p = unsafe { (get_proc_addr)(name.as_ptr()) };
            if p.is_null() {
                None
            } else {
                Some(unsafe { mem::transmute::<*const c_void, _>(p) })
            }
        });

    let display = match get_platform_display {
        Some(f) => unsafe { f(EGL_PLATFORM_WAYLAND_EXT, wl_display, std::ptr::null()) },
        None => {
            let egl_get_display: unsafe extern "C" fn(*mut c_void) -> EglDisplay =
                sym(egl_lib, b"eglGetDisplay\0").ok_or("libEGL missing eglGetDisplay")?;
            unsafe { egl_get_display(wl_display) }
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
    let mut num_config: EglInt = 0;
    if unsafe {
        choose_config(
            display,
            config_attrs.as_ptr(),
            std::ptr::null_mut(),
            0,
            &mut num_config,
        )
    } != EGL_TRUE
        || num_config < 1
    {
        return Err(format!("eglChooseConfig failed: {:#x}", unsafe { (get_error)() }));
    }
    let mut configs: Vec<EglConfig> = vec![std::ptr::null(); num_config as usize];
    if unsafe {
        choose_config(
            display,
            config_attrs.as_ptr(),
            configs.as_mut_ptr(),
            num_config,
            &mut num_config,
        )
    } != EGL_TRUE
    {
        return Err(format!("eglChooseConfig(2) failed: {:#x}", unsafe { (get_error)() }));
    }
    let config: EglConfig = configs[0];

    // Try a 2.1 compatibility context; fall back to the driver default.
    let ctx_attrs: [EglInt; 5] = [
        EGL_CONTEXT_MAJOR_VERSION,
        2,
        EGL_CONTEXT_MINOR_VERSION,
        1,
        EGL_NONE,
    ];
    let mut context =
        unsafe { create_context(display, config, std::ptr::null_mut(), ctx_attrs.as_ptr()) };
    if context.is_null() {
        let plain: [EglInt; 1] = [EGL_NONE];
        context = unsafe { create_context(display, config, std::ptr::null_mut(), plain.as_ptr()) };
    }
    if context.is_null() {
        return Err(format!("eglCreateContext failed: {:#x}", unsafe { (get_error)() }));
    }

    let create_window_surface: unsafe extern "C" fn(EglDisplay, EglConfig, *mut c_void, *const EglInt) -> EglSurface =
        match sym(egl_lib, b"eglCreateWindowSurface\0") {
            Some(f) => f,
            None => {
                let name = std::ffi::CString::new("eglCreatePlatformWindowSurfaceEXT").unwrap();
                let p = unsafe { (get_proc_addr)(name.as_ptr()) };
                if p.is_null() {
                    unsafe { (destroy_context)(display, context) };
                    return Err("no eglCreateWindowSurface or platform surface EXT available".into());
                }
                unsafe { mem::transmute::<*const c_void, _>(p) }
            }
        };
    let surface = unsafe { create_window_surface(display, config, native_window, std::ptr::null()) };
    if surface.is_null() {
        let e = unsafe { (get_error)() };
        unsafe { (destroy_context)(display, context) };
        return Err(format!("eglCreateWindowSurface failed: {:#x}", e));
    }

    // Vsync on: the render loop also paces to the `presented` event, but a
    // vsync'd swap is what makes the compositor actually present every frame.
    let _ = unsafe { (swap_interval)(display, 1) };

    Ok(EglFns {
        display,
        surface,
        context,
        get_error,
        swap_buffers,
        swap_interval,
        make_current,
        get_proc_addr,
        terminate,
        destroy_surface,
        destroy_context,
    })
}

/// Hardware-accurate frame-present anchor fed by `wp_presentation_feedback`.
///
/// Each `presented` event carries a CLOCK_MONOTONIC timestamp (mapped onto the
/// app's `Instant` clock via an offset captured at startup), the `refresh`
/// period in ns, and the vblank counter `seq`. This is the same data the DRM
/// anchor derives from the kernel, but reported by the compositor for *our*
/// surface — so it is valid even on a composited Wayland desktop.
#[derive(Debug)]
pub struct WaylandPresent {
    base_instant: Instant,
    base_mono_ns: i128,
    inner: Mutex<PresentInner>,
    cond: Condvar,
}

#[derive(Debug, Default)]
struct PresentInner {
    /// Latest presented instant, in us since `base_instant` (-1 = none).
    last_us: i64,
    /// Whether the most recent event was `discarded` (frame not shown).
    last_discarded: bool,
    /// Measured refresh period in us (0 until a presented with a refresh).
    period_us: u64,
    /// Monotonic counter bumped on every presented/discarded event.
    generation: u64,
}

impl WaylandPresent {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            base_instant: Instant::now(),
            base_mono_ns: mono_now_ns(),
            inner: Mutex::new(PresentInner::default()),
            cond: Condvar::new(),
        })
    }

    /// The present instant recorded by the last `presented` event, if any.
    pub fn last_present(&self) -> Option<Instant> {
        let inner = self.inner.lock().unwrap();
        if inner.last_us < 0 {
            None
        } else {
            Some(self.base_instant + Duration::from_micros(inner.last_us as u64))
        }
    }

    /// Measured refresh period in microseconds (0 until a presented with refresh).
    pub fn period_us(&self) -> u64 {
        self.inner.lock().unwrap().period_us
    }

    /// The current event generation.
    pub fn generation(&self) -> u64 {
        self.inner.lock().unwrap().generation
    }

    /// Blocks until a new presented/discarded event, returning the presented
    /// instant (`Some`) or `None` if the frame was discarded. Used by the paced
    /// emitter stream as its preferred vblank anchor: it returns exactly when
    /// the render loop's present was confirmed by the compositor.
    ///
    /// `last_gen` is the caller's notion of the last consumed generation; it is
    /// updated in place. A `None` return means the frame was discarded, so the
    /// caller should fall back to its own schedule for that period.
    pub fn wait_next(&self, last_gen: &mut u64) -> Option<Instant> {
        loop {
            {
                let inner = self.inner.lock().unwrap();
                if inner.generation != *last_gen {
                    *last_gen = inner.generation;
                    if inner.last_discarded || inner.last_us < 0 {
                        return None;
                    }
                    return Some(self.base_instant + Duration::from_micros(inner.last_us as u64));
                }
            }
            let (guard, _) = self
                .cond
                .wait_timeout(self.inner.lock().unwrap(), Duration::from_millis(250))
                .unwrap();
            drop(guard);
        }
    }
}

/// Wayland object dispatch state: events update the shared [`WaylandPresent`]
/// anchor, the surface geometry, and the keyboard queue.
#[derive(Default)]
struct WlState {
    present: Option<Arc<WaylandPresent>>,
    configured: bool,
    size: (u32, u32),
    refresh_mhz: u32,
    clock_id: u32,
    /// `wl_keyboard` bound once the seat reports the capability.
    keyboard: Option<WlKeyboard>,
    /// Key characters (already decoded from evdev keycodes) pressed on the
    /// surface, drained by `WaylandDisplay::take_key`.
    keys: Mutex<Vec<char>>,
    /// When the last per-present log line was emitted, to throttle the noise.
    last_present_log: Option<Instant>,
}

impl WlState {
    /// Decodes a `wl_keyboard.key` (Linux evdev keycode) to the character the
    /// app's key handler expects. Layout-specific (US-layout assumption), which
    /// is fine for a diagnostic demo; no xkbcommon dependency.
    fn push_key(&self, keycode: u32) {
        let ch = match keycode {
            16 => 'q',   // KEY_Q
            23 => 'i',   // KEY_I
            25 => 'p',   // KEY_P
            26 => '[',   // KEY_LEFTBRACE
            27 => ']',   // KEY_RIGHTBRACE
            31 => 's',   // KEY_S
            33 => 'f',   // KEY_F
            34 => 'g',   // KEY_G
            39 => ';',   // KEY_SEMICOLON
            46 => 'c',   // KEY_C
            51 => ',',   // KEY_COMMA
            52 => '.',   // KEY_DOT
            _ => return,
        };
        self.keys.lock().unwrap().push(ch);
    }
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for WlState {
    fn event(
        _state: &mut WlState,
        _proxy: &wl_registry::WlRegistry,
        _event: wl_registry::Event,
        _data: &GlobalListContents,
        _conn: &Connection,
        _qh: &QueueHandle<WlState>,
    ) {
    }
}

impl Dispatch<WlCompositor, ()> for WlState {
    fn event(
        _state: &mut WlState,
        _proxy: &WlCompositor,
        _event: <WlCompositor as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<WlState>,
    ) {
    }
}

impl Dispatch<WlSurface, ()> for WlState {
    fn event(
        _state: &mut WlState,
        _proxy: &WlSurface,
        _event: <WlSurface as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<WlState>,
    ) {
    }
}

impl Dispatch<WlSeat, ()> for WlState {
    fn event(
        state: &mut WlState,
        proxy: &WlSeat,
        event: wl_seat::Event,
        _data: &(),
        _conn: &Connection,
        qh: &QueueHandle<WlState>,
    ) {
        if let wl_seat::Event::Capabilities { capabilities } = event {
            if let wayland_client::WEnum::Value(cap) = capabilities {
                if cap.contains(wl_seat::Capability::Keyboard) && state.keyboard.is_none() {
                    state.keyboard = Some(proxy.get_keyboard(qh, ()));
                }
            }
        }
    }
}

impl Dispatch<WlKeyboard, ()> for WlState {
    fn event(
        state: &mut WlState,
        _proxy: &WlKeyboard,
        event: wl_keyboard::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<WlState>,
    ) {
        if let wl_keyboard::Event::Key { key, state: key_state, .. } = event {
            if matches!(key_state, wayland_client::WEnum::Value(wl_keyboard::KeyState::Pressed)) {
                state.push_key(key);
            }
        }
    }
}

impl Dispatch<WlOutput, ()> for WlState {
    fn event(
        state: &mut WlState,
        _proxy: &WlOutput,
        event: wl_output::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<WlState>,
    ) {
        if let wl_output::Event::Mode { flags, width, height, refresh } = event {
            if width > 0 && height > 0 {
                state.size = (width as u32, height as u32);
            }
            if refresh > 0
                && matches!(flags, wayland_client::WEnum::Value(wl_output::Mode::Current))
            {
                state.refresh_mhz = refresh as u32;
            }
        }
    }
}

impl Dispatch<XdgWmBase, ()> for WlState {
    fn event(
        _state: &mut WlState,
        proxy: &XdgWmBase,
        event: <XdgWmBase as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<WlState>,
    ) {
        if let wayland_protocols::xdg::shell::client::xdg_wm_base::Event::Ping { serial } = event {
            proxy.pong(serial);
        }
    }
}

impl Dispatch<XdgSurface, ()> for WlState {
    fn event(
        state: &mut WlState,
        proxy: &XdgSurface,
        event: <XdgSurface as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<WlState>,
    ) {
        if let wayland_protocols::xdg::shell::client::xdg_surface::Event::Configure { serial } = event {
            proxy.ack_configure(serial);
            state.configured = true;
        }
    }
}

impl Dispatch<XdgToplevel, ()> for WlState {
    fn event(
        _state: &mut WlState,
        _proxy: &XdgToplevel,
        _event: <XdgToplevel as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<WlState>,
    ) {
    }
}

impl Dispatch<WpPresentation, ()> for WlState {
    fn event(
        state: &mut WlState,
        _proxy: &WpPresentation,
        event: <WpPresentation as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<WlState>,
    ) {
        if let wayland_protocols::wp::presentation_time::client::wp_presentation::Event::ClockId {
            clk_id,
        } = event
        {
            state.clock_id = clk_id;
        }
    }
}

impl Dispatch<WpPresentationFeedback, ()> for WlState {
    fn event(
        state: &mut WlState,
        _proxy: &WpPresentationFeedback,
        event: <WpPresentationFeedback as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<WlState>,
    ) {
        let Some(present) = state.present.clone() else {
            return;
        };
        match event {
            Event::Presented {
                tv_sec_hi,
                tv_sec_lo,
                tv_nsec,
                refresh,
                seq_hi: _,
                seq_lo: _,
                flags,
            } => {
                let secs = ((tv_sec_hi as u64) << 32) | tv_sec_lo as u64;
                let ns = (secs as i128) * 1_000_000_000 + tv_nsec as i128;
                // Convert the CLOCK_MONOTONIC present timestamp onto our
                // `Instant` clock: `base_instant` was captured at the same
                // instant as `base_mono_ns`.
                let delta_ns = ns - present.base_mono_ns;
                let last_us = (delta_ns / 1000) as i64;

                let period_us = (refresh as u64) / 1000;
                let mut inner = present.inner.lock().unwrap();
                inner.last_us = last_us.max(0);
                inner.last_discarded = false;
                if (4000..=40000).contains(&period_us) {
                    inner.period_us = period_us;
                }
                inner.generation += 1;
                let f = flags_to_u32(flags);
                drop(inner);
                // Throttle the per-present log to ~1 line/second (at 120 fps an
                // every-frame print floods the terminal and slows the loop).
                let now = Instant::now();
                let due = state
                    .last_present_log
                    .map(|t| now.duration_since(t) >= Duration::from_secs(1))
                    .unwrap_or(true);
                if due {
                    state.last_present_log = Some(now);
                    eprintln!(
                        "nvstusb: wayland presented t+{:.1}ms refresh={}us flags={:#x}",
                        (delta_ns as f64) / 1_000_000.0,
                        period_us,
                        f
                    );
                }
                present.cond.notify_all();
            }
            Event::Discarded => {
                let mut inner = present.inner.lock().unwrap();
                inner.last_discarded = true;
                inner.generation += 1;
                drop(inner);
                present.cond.notify_all();
            }
            Event::SyncOutput { .. } => {}
            _ => {}
        }
    }
}

fn flags_to_u32(flags: wayland_client::WEnum<Kind>) -> u32 {
    u32::from(flags)
}

/// The native-Wayland fullscreen rendering surface.
pub struct WaylandDisplay {
    _egl_lib: Library,
    conn: Connection,
    queue: EventQueue<WlState>,
    surface: WlSurface,
    _xdg_surface: XdgSurface,
    _toplevel: XdgToplevel,
    _wl_egl: WlEglSurface,
    egl: EglFns,
    presentation: WpPresentation,
    state: WlState,
    /// Shared anchor, fed by the Dispatch handler and read by the emitter.
    pub present: Arc<WaylandPresent>,
    /// Surface size in physical px.
    pub width: u32,
    pub height: u32,
    /// Current mode refresh rate in Hz (0 until the output reports it).
    pub refresh_hz: f32,
}

impl WaylandDisplay {
    /// Connects to the Wayland server, creates a fullscreen toplevel and EGL
    /// surface, and binds `wp_presentation`.
    pub fn open() -> Result<WaylandDisplay, String> {
        let egl_lib = unsafe { Library::new("libEGL.so.1") }
            .or_else(|_| unsafe { Library::new("libEGL.so") })
            .map_err(|e| format!("cannot dlopen libEGL: {e}"))?;

        let conn = Connection::connect_to_env().map_err(|e| format!("wayland connect: {e}"))?;
        let (globals, mut queue) = registry_queue_init::<WlState>(&conn)
            .map_err(|e| format!("wayland registry init: {e}"))?;
        let qh = queue.handle();

        let compositor: WlCompositor =
            globals.bind(&qh, 4..=4, ()).map_err(|e| format!("bind wl_compositor: {e}"))?;
        let xdg_wm_base: XdgWmBase =
            globals.bind(&qh, 1..=1, ()).map_err(|e| format!("bind xdg_wm_base: {e}"))?;
        let presentation: WpPresentation =
            globals.bind(&qh, 1..=2, ()).map_err(|e| format!("bind wp_presentation: {e}"))?;
        let seat: Option<WlSeat> = globals.bind(&qh, 1..=7, ()).ok();

        let present = WaylandPresent::new();

        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = xdg_wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_title("NVIDIA 3D Vision OpenGL on Linux Demo".into());
        toplevel.set_app_id("nvstereo3d".into());
        toplevel.set_fullscreen(None);

        let mut state = WlState {
            present: Some(present.clone()),
            configured: false,
            size: (0, 0),
            refresh_mhz: 0,
            clock_id: 0,
            keyboard: None,
            keys: Mutex::new(Vec::new()),
            last_present_log: None,
        };

        // First commit (empty) elicits the initial xdg configure; roundtrip so
        // it is acked before we attach a buffer, and so clock_id arrives.
        surface.commit();
        let _ = queue.roundtrip(&mut state);
        // Roundtrip again so the seat's keyboard capability arrives and the
        // wl_keyboard is created (its Capabilities event fires on bind).
        if let Some(_seat) = seat {
            let _ = queue.roundtrip(&mut state);
        }

        // Bind the first output so we know the fullscreen size and refresh.
        if let Ok(_output) = globals.bind::<WlOutput, WlState, ()>(&qh, 1..=4, ()) {
            let _ = queue.roundtrip(&mut state);
        }

        if state.clock_id != 0 && state.clock_id != CLOCK_MONOTONIC as u32 {
            eprintln!(
                "nvstusb: wayland presentation clock is {:#x} (not CLOCK_MONOTONIC); \
                 presented timestamps will be misaligned to Instant",
                state.clock_id
            );
        }

        let (w, h) = if state.size.0 > 0 && state.size.1 > 0 {
            (state.size.0, state.size.1)
        } else {
            eprintln!("nvstusb: wayland output size unknown; defaulting to 1920x1080");
            (1920, 1080)
        };

        let wl_egl = WlEglSurface::new(surface.id(), w as i32, h as i32)
            .map_err(|e| format!("wl_egl_window create: {e:?}"))?;

        // Native display for EGL = the wl_display pointer (the wl_display
        // object's id IS the display); native window = the wl_egl_window.
        let wl_display_ptr = conn.display().id().as_ptr().cast::<c_void>();
        let egl = init_egl(&egl_lib, wl_display_ptr, wl_egl.ptr().cast_mut())?;

        let refresh_hz = state.refresh_mhz as f32 / 1000.0;
        eprintln!("nvstusb: wayland surface {}x{} @ {:.2} Hz", w, h, refresh_hz);

        Ok(WaylandDisplay {
            _egl_lib: egl_lib,
            conn,
            queue,
            surface,
            _xdg_surface: xdg_surface,
            _toplevel: toplevel,
            _wl_egl: wl_egl,
            egl,
            presentation,
            state,
            present,
            width: w,
            height: h,
            refresh_hz,
        })
    }

    /// Makes the EGL context current so rendering can proceed.
    pub fn make_current(&self) -> Result<(), String> {
        let ok = unsafe {
            (self.egl.make_current)(
                self.egl.display,
                self.egl.surface,
                self.egl.surface,
                self.egl.context,
            )
        };
        if ok != EGL_TRUE {
            return Err(format!("eglMakeCurrent failed: {:#x}", self.egl_error()));
        }
        Ok(())
    }

    /// Resolves a GL symbol through EGL.
    pub fn get_proc_address(&self, name: *const c_char) -> *const c_void {
        unsafe { (self.egl.get_proc_addr)(name) }
    }

    fn egl_error(&self) -> EglEnum {
        unsafe { (self.egl.get_error)() }
    }

    /// Pops the next key character received on the surface (if any). Key events
    /// are dispatched during [`Self::present`]; call this after presenting.
    pub fn take_key(&mut self) -> Option<char> {
        self.state.keys.lock().unwrap().pop()
    }

    /// Swaps the GL back buffer, attaches a `wp_presentation_feedback` for the
    /// commit, and returns the present instant for a frame already reported by
    /// the compositor (`Some`), or `None` if none has landed yet this call.
    ///
    /// Crucially this does NOT block waiting for the just-committed frame's
    /// `presented` event: doing so stalled the render loop past the vblank
    /// budget and dropped it to ~60 fps (the swap would return immediately and
    /// the presented wait added a full frame every time). Rendering is paced by
    /// `eglSwapBuffers`' vsync; the `presented` event simply lags the commit by
    /// one frame and is dispatched on the next call, which still feeds the
    /// paced stream's anchor at the correct period and phase.
    pub fn present(&mut self) -> Result<Option<Instant>, String> {
        let ok = unsafe { (self.egl.swap_buffers)(self.egl.display, self.egl.surface) };
        if ok != EGL_TRUE {
            return Err(format!("eglSwapBuffers failed: {:#x}", self.egl_error()));
        }

        // Request feedback, then commit: the feedback associates with this commit.
        self.presentation.feedback(&self.surface, &self.queue.handle(), ());
        self.surface.commit();
        let _ = self.conn.flush();

        // Return the last present already reported (previous frame's event,
        // dispatched below on a prior call).
        let cur = self.present.generation();
        {
            let inner = self.present.inner.lock().unwrap();
            if inner.generation != cur {
                let presented = !inner.last_discarded && inner.last_us >= 0;
                return Ok(if presented {
                    Some(self.present.base_instant + Duration::from_micros(inner.last_us as u64))
                } else {
                    None
                });
            }
        }

        // Non-blocking drain of whatever has already arrived (presented event
        // for an earlier commit, keyboard keys, ...). No waiting: the render
        // loop must not be held up.
        let backend = self.conn.backend();
        let fd = backend.poll_fd();
        let mut pf = PollFd {
            fd: fd.as_raw_fd(),
            events: POLLIN,
            revents: 0,
        };
        let r = unsafe { poll(&mut pf, 1, 0) };
        if r > 0 && (pf.revents & POLLIN) != 0 {
            if let Some(guard) = self.conn.prepare_read() {
                let _ = guard.read();
            }
            let _ = self.queue.dispatch_pending(&mut self.state);
        }
        Ok(None)
    }
}

impl Drop for WaylandDisplay {
    fn drop(&mut self) {
        if !self.egl.surface.is_null() {
            unsafe { (self.egl.destroy_surface)(self.egl.display, self.egl.surface) };
        }
        if !self.egl.context.is_null() {
            unsafe { (self.egl.destroy_context)(self.egl.display, self.egl.context) };
        }
        if !self.egl.display.is_null() {
            unsafe { (self.egl.terminate)(self.egl.display) };
        }
    }
}
