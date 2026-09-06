//! Binary entry point for `nvstereo-calibrate`, the interactive stereo
//! calibration/tuning demo.  All of its logic lives in the crate root
//! (`nvstereo3d::`); this target is just `main`.

fn main() {
    nvstereo3d::run_demo();
}