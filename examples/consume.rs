//! Diagnostics / interop-check: open a shared region as the CONSUMER (like
//! the helper), drain pending eye-swaps and print config.  Also used by the
//! cross-language validation (a C++ producer mirroring the DLL).

use std::env;
use std::sync::atomic::Ordering;

use nvstereo3d::shm;

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| shm::DEFAULT_SHM_PATH.to_string());
    let s = match shm::Shm::open(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("open failed: {e}");
            std::process::exit(1);
        }
    };
    let eyes = s.drain();
    let names: Vec<&str> = eyes
        .iter()
        .map(|e| match e.eye {
            shm::EYE_LEFT => "L",
            shm::EYE_RIGHT => "R",
            _ => "?",
        })
        .collect();
    println!(
        "rate={} alarm={}us status={} flags={:#x} ring_head={} ring_tail={} connector={:?} drained=[{}]",
        s.rate_hz(),
        s.alarm_delay_us(),
        s.status.load(Ordering::Relaxed),
        s.flags.load(Ordering::Relaxed),
        s.head.load(Ordering::Relaxed),
        s.tail.load(Ordering::Relaxed),
        s.connector_name(),
        names.join(",")
    );
}