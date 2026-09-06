//! Raw control-in probe diagnostic for the NVIDIA 3D Vision USB IR emitter.
//!
//! nvstereo-calibrate's `NvstusbContext::set_rate` collapses the timing-register readback
//! into a Some/None verdict ("TIMING REGS LIVE" vs "TIMING REGS IGNORED").
//! This tool replays the exact same protocol steps but prints *everything*
//! the device answers, so a failed readback can be told apart from a read
//! timeout, a short read, a header-only response, or a polluted pipe.
//!
//!     cargo run --example probe 2>&1 | tee /tmp/probe.log
//!
//! Running it as root (or with the 98-nvstusb.rules udev rule installed) is
//! required, since it claims the emitter interface like the real driver.

use std::env;
use std::time::Duration;

use rusb::{DeviceHandle, UsbContext};

const VID: u16 = 0x0955;
const PID: u16 = 0x0007;

fn env_hex_id(name: &str) -> Option<u16> {
    let raw = env::var(name).ok()?;
    let raw = raw.trim().trim_start_matches("0x").trim_start_matches("0X");
    u16::from_str_radix(raw, 16).ok()
}

fn t0_count(us: f64) -> i32 {
    (-(us * 4.0) + 1.0) as i32
}

fn t2_count(us: f64) -> i32 {
    (-(us * 12.0) + 1.0) as i32
}

fn hex_dump(buf: &[u8]) -> String {
    buf.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")
}

fn write_cmd(handle: &DeviceHandle<rusb::Context>, tag: &str, data: &[u8]) {
    match handle.write_bulk(2, data, Duration::from_secs(1)) {
        Ok(n) => {
            let n = n.min(data.len());
            println!("WRITE  {tag:<28}  {n}B  [{:}]", hex_dump(&data[..n]));
        }
        Err(e) => println!("WRITE  {tag:<28}  ERROR {e}"),
    }
}

fn read_ep(handle: &DeviceHandle<rusb::Context>, tag: &str, buf: &mut [u8], timeout_ms: u64) {
    match handle.read_bulk(0x84, buf, Duration::from_millis(timeout_ms)) {
        Ok(n) => {
            println!(
                "READ   {tag:<28}  {n}B  [{:}]",
                hex_dump(&buf[..n.min(buf.len())])
            );
            if n >= 8 {
                let rd = |i: usize| {
                    buf[i] as i32
                        | (buf[i + 1] as i32) << 8
                        | (buf[i + 2] as i32) << 16
                        | (buf[i + 3] as i32) << 24
                };
                println!(
                    "       parsed                       w={} x={} y={} z={}",
                    rd(4),
                    rd(8),
                    rd(12),
                    rd(24)
                );
            }
        }
        Err(e) => println!("READ   {tag:<28}  ERROR {e}"),
    }
}

fn main() {
    let ctx = rusb::Context::new().expect("libusb context");
    let (vid, pid) = (
        env_hex_id("NVSTUSB_VID").unwrap_or(VID),
        env_hex_id("NVSTUSB_PID").unwrap_or(PID),
    );

    let handle = match ctx.open_device_with_vid_pid(vid, pid) {
        Some(h) => h,
        None => {
            eprintln!("probe: no {vid:04x}:{pid:04x} emitter found on the bus");
            std::process::exit(1);
        }
    };

    println!("probe: opened {vid:04x}:{pid:04x}");

    match handle.device().active_config_descriptor() {
        Ok(cfg) => {
            for iface in cfg.interfaces() {
                for alt in iface.descriptors() {
                    let eps: Vec<String> = alt
                        .endpoint_descriptors()
                        .map(|e| {
                            let kind = match e.transfer_type() {
                                rusb::TransferType::Bulk => "bulk",
                                rusb::TransferType::Interrupt => "intr",
                                _ => "other",
                            };
                            format!("0x{:02x}({kind})", e.address())
                        })
                        .collect();
                    println!(
                        "probe: interface {} alt {} class 0x{:02x} endpoints [{}]",
                        iface.number(),
                        alt.setting_number(),
                        alt.class_code(),
                        eps.join(" ")
                    );
                }
            }
        }
        Err(e) => println!("probe: active config descriptor: {e}"),
    }

    handle.set_auto_detach_kernel_driver(true).ok();
    if let Err(e) = handle.set_active_configuration(1) {
        println!("probe: set_active_configuration: {e} (continuing)");
    }
    if let Err(e) = handle.claim_interface(0) {
        println!("probe: claim_interface: {e}");
        std::process::exit(1);
    }

    let rate = 120.0f64;
    let w = t2_count(4735.58);
    let x = t0_count(0.5);
    let y = t0_count(7334.0);
    let z = t2_count(1e6 / rate);
    let mut block = [0u8; 28];
    block[0..4].copy_from_slice(&[0x01, 0x00, 0x18, 0x00]); // write 24 bytes to 0x2007
    block[4..8].copy_from_slice(&w.to_le_bytes());
    block[8..12].copy_from_slice(&x.to_le_bytes());
    block[12..16].copy_from_slice(&y.to_le_bytes());
    block[16..24].copy_from_slice(&[0x30, 0x28, 0x24, 0x22, 0x0a, 0x08, 0x05, 0x04]);
    block[24..28].copy_from_slice(&z.to_le_bytes());

    println!("--- configure writes (identical to nvstereo-calibrate's configure) ---");
    write_cmd(&handle, "timings block (28B)", &block);
    write_cmd(
        &handle,
        "cnt 0x1c (6B)",
        &[0x01, 0x1c, 0x02, 0x00, 0x02, 0x00],
    );
    if rate > 60.0 {
        let timeout = (rate as i32) * 4;
        write_cmd(
            &handle,
            "timeout 0x1e (6B)",
            &[0x01, 0x1e, 0x02, 0x00, timeout as u8, (timeout >> 8) as u8],
        );
    }
    write_cmd(
        &handle,
        "driver 0x1b (5B)",
        &[0x01, 0x1b, 0x01, 0x00, 0x07],
    );

    println!("--- timing-register readback (`02 00 1c 00`, expect 32B) ---");
    write_cmd(&handle, "read 0x2007 (cmd)", &[0x02, 0x00, 0x1c, 0x00]);
    let mut resp = [0u8; 32];
    read_ep(&handle, "attempt 1 (200ms)", &mut resp, 200);
    read_ep(&handle, "attempt 2 (1s)", &mut resp, 1000);

    println!("--- keys readback (`42 18 03 00`, expect 7B) ---");
    write_cmd(&handle, "read+clear 0x201f", &[0x42, 0x18, 0x03, 0x00]);
    let mut keys = [0u8; 7];
    read_ep(&handle, "keys (200ms)", &mut keys, 200);

    println!("--- blind drain (no command; whatever the pipe holds) ---");
    let mut drain = [0u8; 32];
    read_ep(&handle, "drain (100ms)", &mut drain, 100);
}