//! Shared libusb transport + emitter configuration for the NVIDIA 3D Vision
//! USB IR emitter.
//!
//! One source serves both consumers in this crate:
//!   - `nvstusb::NvstusbContext` (the 3dv3d demo) uses the transport
//!     (`usb_init` / `open_device` / `write_bulk` / `read_bulk`) and the
//!     configure/send helpers below.
//!   - the `host` helper (wiz3D bridge) uses the same transport plus
//!     `UsbDevice::configure` / `send_eye` / `set_alarm_delay_us`.
//!
//! The byte-level protocol and the RP2040 clone control registers are the
//! single source of truth for both.

use rusb::{Context, DeviceHandle, UsbContext};
use std::time::Duration;

pub const NVSTUSB_VID: u16 = 0x0955;
pub const NVSTUSB_PID: u16 = 0x0007;

const REQUEST_TYPE_VENDOR: u8 = 0x40;
const REQUEST_FIRMWARE: u8 = 0xA0;

/// CPU clock of the emitter's controller.
pub const NVSTUSB_CLOCK: f64 = 48_000_000.0;
/// T0 runs at 4 MHz.
pub const NVSTUSB_T0_CLOCK: f64 = NVSTUSB_CLOCK / 12.0;
/// T2 runs at 12 MHz.
pub const NVSTUSB_T2_CLOCK: f64 = NVSTUSB_CLOCK / 4.0;

/// Converts a microsecond duration to a T0 timer reload value.
fn t0_count(us: f64) -> i32 {
    (-(us * (NVSTUSB_T0_CLOCK / 1_000_000.0)) + 1.0) as i32
}

/// Converts a microsecond duration to a T2 timer reload value.
pub fn t2_count(us: f64) -> i32 {
    (-(us * (NVSTUSB_T2_CLOCK / 1_000_000.0)) + 1.0) as i32
}

/// An opened NVIDIA 3D Vision emitter.
pub struct UsbDevice {
    _context: &'static Context,
    handle: DeviceHandle<Context>,
}

/// Initializes the global libusb context.
///
/// The returned context lives for the whole process; it is intentionally leaked
/// so `DeviceHandle`s don't need a borrow attached to it.
pub fn usb_init() -> Option<&'static Context> {
    let mut context = Context::new().ok()?;
    context.set_log_level(rusb::LogLevel::Info);
    eprintln!("nvstusb: libusb initialized");
    Some(Box::leak(Box::new(context)))
}

impl UsbDevice {
    /// Writes `data` to a bulk OUT endpoint.
    pub fn write_bulk(&self, endpoint: u8, data: &[u8]) -> Result<usize, rusb::Error> {
        self.handle
            .write_bulk(endpoint, data, Duration::from_secs(1))
    }

    /// Reads up to `buf.len()` bytes from a bulk IN endpoint.
    pub fn read_bulk(&self, endpoint: u8, buf: &mut [u8]) -> Result<usize, rusb::Error> {
        self.handle
            .read_bulk(endpoint | 0x80, buf, Duration::from_millis(200))
    }

    /// Performs a vendor request control transfer (used for firmware upload).
    pub fn write_control(
        &self,
        request: u8,
        value: u16,
        data: &[u8],
    ) -> Result<usize, rusb::Error> {
        self.handle.write_control(
            REQUEST_TYPE_VENDOR,
            request,
            value,
            0,
            data,
            Duration::from_secs(1),
        )
    }
}

/// Decides whether the device is a bare EZ-USB dongle waiting for its RAM
/// firmware (`true`), mirroring libnvstusb's `nvstusb_usb_needs_firmware()`.
///
/// Semantics matter here: only an ACTIVE configuration that exists and
/// reports zero endpoints counts as bare.  A device with NO active
/// configuration at all (descriptor read fails) is treated as already
/// running from flash - real hardware, integrated emitters included - and
/// merely gets configuration 1 set below, exactly like the original C
/// library.  Mapping a failed descriptor read to "zero endpoints" sent real
/// emitters down the firmware-upload path where they fail.
fn needs_firmware(handle: &DeviceHandle<Context>) -> bool {
    match handle.device().active_config_descriptor() {
        Ok(config) => {
            let num = config
                .interfaces()
                .next()
                .and_then(|iface| iface.descriptors().next())
                .map(|alt| alt.num_endpoints());
            match num {
                Some(n) => {
                    eprintln!("nvstusb: found {n} endpoints");
                    n == 0
                }
                None => {
                    eprintln!("nvstusb: active config has no interfaces");
                    true
                }
            }
        }
        Err(err) => {
            eprintln!(
                "nvstusb: no active configuration ({err}); assuming device runs from flash"
            );
            false
        }
    }
}

/// Uploads the firmware image (from `nvstusb.fw`) to the device.  The format
/// is a stream of `<len:u16 big-endian><addr:u16 big-endian><bytes>` records.
fn load_firmware(handle: &DeviceHandle<Context>, firmware: &[u8]) -> Result<(), String> {
    eprintln!("nvstusb: loading firmware ({} bytes)...", firmware.len());
    let mut pos = 0usize;

    while pos + 4 <= firmware.len() {
        let length = ((firmware[pos] as u16) << 8) | firmware[pos + 1] as u16;
        let address = ((firmware[pos + 2] as u16) << 8) | firmware[pos + 3] as u16;
        pos += 4;
        if length as usize > firmware.len() - pos {
            return Err("truncated firmware record".into());
        }
        let buf = &firmware[pos..pos + length as usize];
        pos += length as usize;

        handle
            .write_control(
                REQUEST_TYPE_VENDOR,
                REQUEST_FIRMWARE,
                address,
                0,
                buf,
                Duration::from_secs(1),
            )
            .map_err(|e| format!("error uploading firmware: {e}"))?;
    }
    if pos != firmware.len() {
        return Err("firmware ended mid-record".into());
    }
    Ok(())
}

/// Parses a hex USB id from an environment variable (`"0955"`, `"0x0955"`).
fn env_hex_id(name: &str) -> Option<u16> {
    let raw = std::env::var(name).ok()?;
    let raw = raw.trim().trim_start_matches("0x").trim_start_matches("0X");
    u16::from_str_radix(raw, 16).ok()
}

/// Parses a 32-byte timing-register read-back response into `[w, x, y, z]`
/// reload counts, or `None` when the shape is wrong.
///
/// Layout (see [`timings_block`]): 4 header bytes `[offset 0x00, amount 0x1c,
/// 0x00, 0x04]`, then the 28-byte register block — w @ 4, x @ 8, y @ 12,
/// eye/port-B toggles @ 16, and the **z frame-time reload at response offset
/// 24** (data offset 20 of the 24-byte timing block; bytes 28..32 are only
/// padding next to the block).
fn parse_timings_response(resp: &[u8]) -> Option<[i32; 4]> {
    if resp.len() < 32
        || resp[0] != 0x00
        || resp[1] != 0x1c
        || resp[2] != 0x00
        || resp[3] != 0x04
    {
        return None;
    }
    let rd = |i: usize| {
        resp[i] as i32
            | (resp[i + 1] as i32) << 8
            | (resp[i + 2] as i32) << 16
            | (resp[i + 3] as i32) << 24
    };
    Some([rd(4), rd(8), rd(12), rd(24)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_hex_ids_parse() {
        // Unique names so parallel tests never share state.
        std::env::set_var("NVSTUSB_TEST_ID_1", "0x0955");
        assert_eq!(env_hex_id("NVSTUSB_TEST_ID_1"), Some(0x0955));
        std::env::set_var("NVSTUSB_TEST_ID_1", "0007");
        assert_eq!(env_hex_id("NVSTUSB_TEST_ID_1"), Some(0x0007));
        std::env::set_var("NVSTUSB_TEST_ID_1", "700A");
        assert_eq!(env_hex_id("NVSTUSB_TEST_ID_1"), Some(0x700A));
        std::env::set_var("NVSTUSB_TEST_ID_1", "not-hex");
        assert_eq!(env_hex_id("NVSTUSB_TEST_ID_1"), None);
    }

    #[test]
    fn alt_pid_list_is_sorted_probe_order() {
        let mut pids = NVIDIA_ALT_PIDS;
        pids.sort_by_key(|p| NVIDIA_ALT_PIDS.iter().position(|a| a == p).unwrap_or(usize::MAX));
        assert_eq!(pids, NVIDIA_ALT_PIDS);
    }

    /// The Z frame-time reload lives at block data offset 20 (response offset
    /// 24).  A parser that reads `resp[28]` silently reports z=0 from the
    /// padding — the exact lie this regression test guards against.
    #[test]
    fn parse_timings_response_reads_z_from_block_offset() {
        let w = t2_count(4735.58);
        let x = t0_count(0.5);
        let y = t0_count(7334.0);
        let z = t2_count(1e6 / 120.0);

        let mut resp = [0u8; 32];
        resp[0..4].copy_from_slice(&[0x00, 0x1c, 0x00, 0x04]);
        resp[4..8].copy_from_slice(&w.to_le_bytes());
        resp[8..12].copy_from_slice(&x.to_le_bytes());
        resp[12..16].copy_from_slice(&y.to_le_bytes());
        resp[16..24].copy_from_slice(&[0x30, 0x28, 0x24, 0x22, 0x0a, 0x08, 0x05, 0x04]);
        resp[24..28].copy_from_slice(&z.to_le_bytes());
        // Padding after the block must not be read as z.
        resp[28..32].copy_from_slice(&[0xde, 0xad, 0xbe, 0xef]);

        assert_eq!(parse_timings_response(&resp), Some([w, x, y, z]));
    }

    /// Short reads (e.g. a leftover 7-byte button/keys echo from an earlier
    /// session) and responses whose header does not match the timing block
    /// must be rejected, not misparsed.
    #[test]
    fn parse_timings_response_rejects_short_and_foreign() {
        let stale_keys = [0x18, 0x03, 0x00, 0x04, 0x00, 0x00, 0x00];
        assert_eq!(parse_timings_response(&stale_keys), None);

        let mut foreign = [0u8; 32];
        foreign[0..4].copy_from_slice(&[0x18, 0x03, 0x00, 0x04]); // keys offset
        foreign[4..8].copy_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        assert_eq!(parse_timings_response(&foreign), None);
    }
}

/// Emitter USB identity: honors `NVSTUSB_VID` / `NVSTUSB_PID` environment
/// overrides, falling back to the external dongle's fixed NVIDIA ids.
///
/// Integrated emitters (built into monitors or laptops) do not necessarily
/// enumerate as `0955:0007`.  Point the two variables at whatever `lsusb`
/// shows for such a unit to try it with the same protocol.
fn emitter_ids() -> (u16, u16) {
    (
        env_hex_id("NVSTUSB_VID").unwrap_or(NVSTUSB_VID),
        env_hex_id("NVSTUSB_PID").unwrap_or(NVSTUSB_PID),
    )
}

/// Whether a device with these ids is currently on the bus (numeric probe,
/// opens nothing).
fn device_present(context: &Context, vid: u16, pid: u16) -> bool {
    context.devices().map_or(false, |list| {
        list.iter()
            .filter_map(|d| d.device_descriptor().ok())
            .any(|desc| desc.vendor_id() == vid && desc.product_id() == pid)
    })
}

/// Prints devices sharing the emitter's VID, and - with `NVSTUSB_LIST=1` -
/// every device on the bus.  Helps spot emitters that enumerate under
/// unexpected ids (e.g. integrated types).
fn dump_devices(context: &Context, vid: u16, want_all: bool) {
    let Ok(list) = context.devices() else {
        return;
    };
    for d in list.iter() {
        let Ok(desc) = d.device_descriptor() else {
            continue;
        };
        let v = desc.vendor_id();
        let p = desc.product_id();
        if want_all || v == vid {
            eprintln!(
                "nvstusb:   usb {:04x}:{:04x} (bus {:03}, device {:03})",
                v,
                p,
                d.bus_number(),
                d.address()
            );
        }
    }
}

/// Alternate NVIDIA PIDs probed when the external dongle's own id is absent.
/// Integrated emitters (built into monitors/laptops) can enumerate under a
/// different PID while keeping NVIDIA's vendor id - this list is what the
/// 3DVisionActivator project probes for other 3D Vision hardware.
const NVIDIA_ALT_PIDS: [u16; 10] = [
    0x7001, 0x7002, 0x7003, 0x7004, 0x7008, 0x7009, 0x700A, 0x700C, 0x700D, 0x700E,
];

/// Every PID currently on the bus under NVIDIA's vendor id, excluding
/// `except`, ordered known-alternates first (deterministic probe order).
fn nvidia_pids_on_bus(context: &Context, except: u16) -> Vec<u16> {
    let mut found: Vec<u16> = Vec::new();
    if let Ok(list) = context.devices() {
        for d in list.iter() {
            if let Ok(desc) = d.device_descriptor() {
                let v = desc.vendor_id();
                let p = desc.product_id();
                if v == NVSTUSB_VID && p != except && !found.contains(&p) {
                    found.push(p);
                }
            }
        }
    }
    found.sort_by_key(|p| {
        NVIDIA_ALT_PIDS
            .iter()
            .position(|a| a == p)
            .unwrap_or(NVIDIA_ALT_PIDS.len())
    });
    found
}

/// Opens the NVIDIA 3D Vision emitter, uploading firmware first if required.
///
/// Candidate order encodes hardware preference: an explicit `NVSTUSB_VID`/
/// `NVSTUSB_PID` pin wins outright, then the external-dongle id
/// (`0955:0007`, which custom clones also use), then any other NVIDIA-vendor
/// unit on the bus (integrated emitters).  The first candidate whose FULL
/// setup succeeds is used; a higher-priority device that fails mid-setup
/// falls through to the next one instead of taking the whole session down.
pub fn open_device(context: &'static Context, firmware: &[u8]) -> Option<UsbDevice> {
    let (vid, pid) = emitter_ids();
    let pinned =
        std::env::var_os("NVSTUSB_VID").is_some() || std::env::var_os("NVSTUSB_PID").is_some();

    let mut candidates = vec![pid];
    if !pinned {
        candidates.extend(nvidia_pids_on_bus(context, pid));
    }

    for &cand in &candidates {
        let Some(handle) = context.open_device_with_vid_pid(vid, cand) else {
            // Absent vs permissions-denied - say which, quietly per skip.
            if device_present(context, vid, cand) {
                eprintln!(
                    "nvstusb: {vid:04x}:{cand:04x} is present but cannot be opened \
                     (permissions)"
                );
            }
            continue;
        };
        eprintln!("nvstusb: found NVIDIA 3D stereo controller at {vid:04x}:{cand:04x}");
        match bring_up(context, vid, cand, handle, firmware) {
            Ok(handle) => {
                return Some(UsbDevice {
                    _context: context,
                    handle,
                })
            }
            Err(e) => {
                eprintln!("nvstusb: {vid:04x}:{cand:04x} unusable ({e}); trying next candidate")
            }
        }
    }

    // Nothing opened.  Report the headline causes for the PRIMARY id.
    if device_present(context, vid, pid) {
        eprintln!(
            "nvstusb: NVIDIA 3D stereo controller {vid:04x}:{pid:04x} is present \
             but cannot be opened (permissions)"
        );
        eprintln!(
            "nvstusb: install 98-nvstusb.rules into /etc/udev/rules.d/, \
             run `udevadm control --reload && udevadm trigger`, or run as root"
        );
    } else if candidates.len() > 1 || pinned {
        eprintln!(
            "nvstusb: no usable NVIDIA 3D stereo controller ({vid:04x}:{pid:04x} probed first)"
        );
    } else {
        eprintln!("nvstusb: no NVIDIA 3D stereo controller ({vid:04x}:{pid:04x}) found");
        eprintln!(
            "nvstusb: integrated emitters may enumerate under another id - \
             check `lsusb`; set NVSTUSB_VID/NVSTUSB_PID to try one, \
             NVSTUSB_LIST=1 to dump the bus"
        );
        dump_devices(context, vid, std::env::var_os("NVSTUSB_LIST").is_some());
    }
    None
}

/// Full setup of one opened candidate: EZ-USB firmware upload (with polled
/// re-enumeration) when the device is bare, then configuration/claim.
/// Returns the error instead of aborting the process-wide search, so a
/// failing high-priority emitter falls through to the next candidate.
fn bring_up(
    context: &'static Context,
    vid: u16,
    pid: u16,
    handle: DeviceHandle<Context>,
    firmware: &[u8],
) -> Result<DeviceHandle<Context>, String> {
    if !needs_firmware(&handle) {
        // Real hardware: the original just sets configuration 1 and claims
        // interface 0.  On an unconfigured unit this is where its endpoints
        // appear; on an already-configured one the request is a no-op.
        set_up_interface(&handle);
        return Ok(handle);
    }

    // Bare Cypress EZ-USB dongle: upload nvstusb.fw into RAM; the device
    // then re-enumerates (possibly at a new bus address) as the real thing.
    if let Err(e) = load_firmware(&handle, firmware) {
        return Err(format!("firmware load failed: {e}"));
    }
    // The original ignores both reset results; a failure here is not fatal
    // as long as the re-open below succeeds.
    handle.reset().ok();
    drop(handle);

    std::thread::sleep(Duration::from_millis(250));
    // Re-enumeration races us: a fixed 250 ms sleep plus a single open
    // attempt loses whenever the kernel takes longer to re-probe the port.
    // Poll for up to ~6 s instead.
    let mut reopened = None;
    for _ in 0..24 {
        if let Some(h) = context.open_device_with_vid_pid(vid, pid) {
            reopened = Some(h);
            break;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    let handle = reopened.ok_or_else(|| {
        format!("device did not re-enumerate within 6 s of firmware load ({vid:04x}:{pid:04x})")
    })?;
    handle.reset().ok();
    std::thread::sleep(Duration::from_millis(250));
    set_up_interface(&handle);
    Ok(handle)
}

/// Configuration/claim sequence shared by both open paths.  Mirrors the
/// original's `libusb_set_configuration(handle, 1)` + `claim_interface(0)`
/// with error checking added, and asks the kernel to release any driver
/// (hid-generic etc.) bound to a real emitter first.
fn set_up_interface(handle: &DeviceHandle<Context>) {
    handle.set_auto_detach_kernel_driver(true).ok();
    if let Err(e) = handle.set_active_configuration(1) {
        eprintln!("nvstusb: set_active_configuration(1): {e} (continuing)");
    }
    if let Err(e) = handle.claim_interface(0) {
        eprintln!("nvstusb: claim_interface(0) failed: {e}");
    }
}

/// Serializes the emitter's 28-byte shutter-timing register block written to
/// 0x2007 — the same bytes the NVIDIA Windows driver programs per monitor
/// (see 3DVisionActivator's `MonitorTimings.ini` and NV3D-Lib's
/// `nvtimings.json`).
///
/// X = delay from monitor refresh start to the shutter open edge (us, T0
/// reload), Y = shutter open window (us, T0 reload), W = second T2 timer
/// counter (us; stored per monitor, rarely needs tuning); Z = frame time
/// (us), derived from the refresh rate, the T2 frame-period reload.
///
/// Byte 0 of the payload is w, then x, then y, then the eye/port-B toggle
/// tables, then z at payload offset 20 — the same layout the RP2040 clone
/// implements and the read-back echoes verbatim.
pub fn timings_block(rate: f32, x_us: f64, y_us: f64, w_us: f64) -> [u8; 28] {
    let rate = if rate > 60.0 { rate as f64 } else { 120.0 };
    let w = t2_count(w_us);
    let x = t0_count(x_us);
    let y = t0_count(y_us);
    let z = t2_count(1e6 / rate);
    [
        0x01, 0x00, 0x18, 0x00, // write 24 bytes to 0x2007
        w as u8, (w >> 8) as u8, (w >> 16) as u8, (w >> 24) as u8,
        x as u8, (x >> 8) as u8, (x >> 16) as u8, (x >> 24) as u8,
        y as u8, (y >> 8) as u8, (y >> 16) as u8, (y >> 24) as u8,
        0x30, // 2013: left eye off
        0x28, // 2014: left eye on
        0x24, // 2015: right eye off
        0x22, // 2016: right eye on
        0x0a, 0x08, 0x05, 0x04, // Port B toggle bits
        z as u8, (z >> 8) as u8, (z >> 16) as u8, (z >> 24) as u8, // T2 frame-time reload
    ]
}

impl UsbDevice {
    /// Configures the emitter for the given refresh rate and enables driver
    /// mode.  Exact byte stream from 3dv3d's `NvstusbContext::set_rate`, with
    /// the reference 1440p @ 120 Hz shutter timings (X=0.5us, Y=7334.0us,
    /// W=4735.58us) and Z taken from the refresh rate.  `configure` owns the
    /// 0x1c / timeout / driver-enable registers; use [`Self::set_timings_us`]
    /// to reprogram only the shutter timings live afterwards.
    pub fn configure(&self, rate: f32) -> Result<(), String> {
        self.write_bulk(2, &timings_block(rate, 0.5, 7334.0, 4735.58))
            .map_err(|e| format!("timings write failed: {e}"))?;

        let cmd_0x1c: [u8; 6] = [0x01, 0x1c, 0x02, 0x00, 0x02, 0x00];
        self.write_bulk(2, &cmd_0x1c)
            .map_err(|e| format!("0x1c write failed: {e}"))?;

        let timeout = (rate * 4.0) as u16;
        let cmd_timeout: [u8; 6] = [0x01, 0x1e, 0x02, 0x00, timeout as u8, (timeout >> 8) as u8];
        self.write_bulk(2, &cmd_timeout)
            .map_err(|e| format!("timeout write failed: {e}"))?;

        // Driver enable.
        let cmd_0x1b: [u8; 5] = [0x01, 0x1b, 0x01, 0x00, 0x07];
        self.write_bulk(2, &cmd_0x1b)
            .map_err(|e| format!("driver-enable write failed: {e}"))?;

        eprintln!("nvstusb: emitter configured at {rate:.1} Hz (driver mode)");
        Ok(())
    }

    /// Reprograms only the shutter-timing registers (X/Y/W, microseconds)
    /// live.  This is the 3DVisionActivator / NV3D-Lib per-monitor tuning
    /// step: X = delay from monitor refresh start to the shutter open edge,
    /// Y = shutter open window, W = second T2 timer counter (stored per
    /// monitor, rarely needs tuning).  `configure` must have run once first
    /// (it owns the 0x1c / timeout / driver-enable registers, which are
    /// unaffected by re-tuning).  Z (frame time) always follows the configured
    /// refresh rate.  Whether a given emitter actually drives the shutters
    /// from these registers is verified by [`Self::read_timings_registers`]
    /// (see `NvstusbContext::probe_timings_live`).
    pub fn set_timings_us(&self, rate: f32, x_us: f64, y_us: f64, w_us: f64) -> Result<(), String> {
        self.write_bulk(2, &timings_block(rate, x_us, y_us, w_us))
            .map_err(|e| format!("shutter timing write failed: {e}"))?;
        Ok(())
    }

    /// Reads the 0x2007 shutter-timing register block back from the emitter —
    /// the same write-back verification 3DVisionActivator performs after every
    /// `refresh()`.  Returns the four stored timer reload values as `[w, x, y,
    /// z]` (T2/T0/T0/T2 counts) when the device answers with the expected
    /// 32-byte status response (4 header bytes + the 28-byte register block).
    ///
    /// Echo layout (the 24-byte timing block written by [`timings_block`],
    /// padded to 28 bytes): w @ 4, x @ 8, y @ 12, eye/port-B toggles @ 16,
    /// z @ 24 — the frame-time reload is at response offset **24**, not 28,
    /// which is only padding after the block (3DVisionActivator's `resp[28]`
    /// read hits that padding; `z` is at `resp[24]`).
    ///
    /// A device can answer with a *stale* control-in response first — e.g. a
    /// leftover 7-byte button/keys echo from a previous host session that
    /// exited without reading it — so the read is retried and any response
    /// whose header does not match the expected `[offset 0x00, amount 0x1c,
    /// 0x00, 0x04]` shape is consumed and discarded rather than misread as
    /// the timing block.
    pub fn read_timings_registers(&self) -> Option<[i32; 4]> {
        // Plain read, 28 bytes from offset 0x2007 (0x02 = read command,
        // then the 0x2007 offset and 0x001c = 28 data bytes).
        let cmd: [u8; 4] = [0x02, 0x00, 0x1c, 0x00];
        for _ in 0..3 {
            self.write_bulk(2, &cmd).ok()?;
            let mut resp = [0u8; 32];
            let Ok(n) = self.read_bulk(4, &mut resp) else {
                continue; // timeout / bus error: nothing read; re-issue
            };
            if let Some(vals) = parse_timings_response(&resp[..n]) {
                return Some(vals);
            }
            // Short, ill-shaped, or stale response (e.g. a leftover keys echo
            // from a previous host session): the read consumed it; re-issue.
        }
        None
    }

    /// Sets the RP2040 clone's packet -> IR delay (control register 0x23),
    /// used to align the shutter open window with the eye-frame boundary.
    pub fn set_alarm_delay_us(&self, us: u32) {
        let cmd: [u8; 8] = [
            0x01, 0x23, 0x04, 0x00, us as u8, (us >> 8) as u8, (us >> 16) as u8, (us >> 24) as u8,
        ];
        let _ = self.write_bulk(2, &cmd);
    }

    /// Fires one shutter pulse for the given eye.  Same packet as 3dv3d's
    /// `NvstusbContext::set_eye` (master-mode shutter packet, endpoint 1).
    /// `rate` is used to derive the T2 timer reload value.
    pub fn send_eye(&self, right: bool, rate: f32) {
        let rate = if rate > 60.0 { rate as f64 } else { 120.0 };
        let r = t2_count((1e6 / rate) / 1.8) as u32;
        let eye_select = if right { 0xFE } else { 0xFF };
        let buf: [u8; 8] = [
            0xAA, eye_select, 0x00, 0x00, r as u8, (r >> 8) as u8, (r >> 16) as u8, (r >> 24) as u8,
        ];
        let _ = self.write_bulk(1, &buf);
    }
}