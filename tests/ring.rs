//! End-to-end test of the shared-memory ring: a producer handle and a
//! consumer handle to the *same* backing file, like the DLL (producer) and
//! this helper (consumer) in production.  Exercises wrap-around, full-ring
//! rejection, and config-field reads.

use nvstereo3d::shm::{self, Shm, EYE_LEFT, EYE_RIGHT};

fn temp_path(name: &str) -> String {
    format!(
        "/tmp/nvstusb-host-test-{}-{name}.shm",
        std::process::id()
    )
}

#[test]
fn ring_roundtrip_and_wraparound() {
    let path = temp_path("roundtrip");
    let _ = std::fs::remove_file(&path);

    // Producer "creates" (init) the region.
    let mut prod = Shm::open(&path).unwrap();
    prod.init_region();
    assert_eq!(prod.cap(), 1024);

    // Consumer opens the already-initialized region.
    let cons = Shm::open(&path).unwrap();
    assert_eq!(cons.cap(), 1024);

    // Push a few, drain, verify order.
    assert!(prod.enqueue(EYE_LEFT));
    assert!(prod.enqueue(EYE_RIGHT));
    assert!(prod.enqueue(EYE_LEFT));
    let drained: Vec<shm::Swap> = cons.drain();
    assert_eq!(
        drained.iter().map(|s| s.eye).collect::<Vec<_>>(),
        vec![EYE_LEFT, EYE_RIGHT, EYE_LEFT]
    );
    assert!(cons.drain().is_empty());

    // Wrap-around: push more than the capacity so indices exceed u32 mask.
    for i in 0..1100 {
        let eye = if i % 2 == 0 { EYE_LEFT } else { EYE_RIGHT };
        // Ring is never drained during the push, so once it fills the next
        // pushes must be rejected.
        if i >= 1024 {
            assert!(!prod.enqueue(eye), "expected full at {i}");
        } else {
            assert!(prod.enqueue(eye));
        }
    }
    let drained = cons.drain();
    assert_eq!(drained.len(), 1024, "should drain exactly cap items");

    // A fresh enqueue after draining works again (indices wrapped).
    assert!(prod.enqueue(EYE_RIGHT));
    let drained: Vec<shm::Swap> = cons.drain();
    assert_eq!(drained.iter().map(|s| s.eye).collect::<Vec<_>>(), vec![EYE_RIGHT]);

    std::fs::remove_file(&path).ok();
}

#[test]
fn config_fields_roundtrip() {
    let path = temp_path("config");
    let _ = std::fs::remove_file(&path);

    let mut a = Shm::open(&path).unwrap();
    a.init_region();

    // Producer writes config (f32 bits + delay).
    a.rate_hz.store(120.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
    a.alarm_delay_us
        .store(3000, std::sync::atomic::Ordering::Relaxed);

    // Consumer reads them back.
    let b = Shm::open(&path).unwrap();
    assert_eq!(b.rate_hz(), 120.0);
    assert_eq!(b.alarm_delay_us(), 3000);

    std::fs::remove_file(&path).ok();
}
