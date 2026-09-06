//! Binary entry point for the `nvstereo3d` Linux host bridge.  All of its
//! logic lives in `nvstereo3d::host`; this target is just `main`.

fn main() {
    nvstereo3d::host::run();
}