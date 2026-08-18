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

/// Counts the endpoints of the active configuration of the emitter.
///
/// A device that still needs its firmware uploaded reports zero endpoints, which
/// is how we decide whether `nvstusb.fw` must be loaded.
fn num_endpoints(handle: &DeviceHandle<Context>) -> Option<u8> {
    let config = handle.device().active_config_descriptor().ok()?;
    let interface = config.interfaces().next()?;
    let altsetting = interface.descriptors().next()?;
    let num = altsetting.num_endpoints();
    eprintln!("nvstusb: found {} endpoints", num);
    Some(num)
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

/// Opens the NVIDIA 3D Vision emitter, uploading firmware first if required.
pub fn open_device(context: &'static Context, firmware: &[u8]) -> Option<UsbDevice> {
    let handle = match context.open_device_with_vid_pid(NVSTUSB_VID, NVSTUSB_PID) {
        Some(h) => h,
        None => {
            eprintln!("nvstusb: no NVIDIA 3D stereo controller found");
            return None;
        }
    };
    eprintln!("nvstusb: found NVIDIA 3D stereo controller");

    if num_endpoints(&handle).unwrap_or(0) == 0 {
        if let Err(e) = load_firmware(&handle, firmware) {
            eprintln!("nvstusb: firmware load failed: {e}");
            return None;
        }
        handle.reset().ok()?;
        drop(handle);

        std::thread::sleep(Duration::from_millis(250));
        let handle = context.open_device_with_vid_pid(NVSTUSB_VID, NVSTUSB_PID)?;
        handle.reset().ok()?;
        std::thread::sleep(Duration::from_millis(250));

        handle.set_active_configuration(1).ok();
        handle.claim_interface(0).ok();
        Some(UsbDevice {
            _context: context,
            handle,
        })
    } else {
        handle.set_active_configuration(1).ok();
        handle.claim_interface(0).ok();
        Some(UsbDevice {
            _context: context,
            handle,
        })
    }
}

impl UsbDevice {
    /// Configures the emitter for the given refresh rate and enables driver
    /// mode.  Exact byte stream from 3dv3d's `NvstusbContext::set_rate`.
    pub fn configure(&self, rate: f32) -> Result<(), String> {
        // Shutter timings for a 2560x1440 @ 120 Hz panel from the NV3D-Lib
        // nvtimings.json reference: X=0.5us (refresh-start->open),
        // Y=7334.0us (open), Z=8333.5us (frame time), W=4735.58us.
        let w = t2_count(4735.58);
        let x = t0_count(0.5);
        let y = t0_count(7334.0);
        let z = t2_count(8333.5);

        let cmd_timings: [u8; 28] = [
            0x01, 0x00, 0x18, 0x00, // write 24 bytes to 0x2007
            w as u8, (w >> 8) as u8, (w >> 16) as u8, (w >> 24) as u8,
            x as u8, (x >> 8) as u8, (x >> 16) as u8, (x >> 24) as u8,
            y as u8, (y >> 8) as u8, (y >> 16) as u8, (y >> 24) as u8,
            0x30, // 2013: left eye off
            0x28, // 2014: left eye on
            0x24, // 2015: right eye off
            0x22, // 2016: right eye on
            0x0a, 0x08, 0x05, 0x04, // Port B toggle bits
            z as u8, (z >> 8) as u8, (z >> 16) as u8, (z >> 24) as u8, // T2 reload
        ];
        self.write_bulk(2, &cmd_timings)
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