# nvstereo3d

NVIDIA 3D Vision glasses on Linux, for games running under
[Wine/Proton with wiz3D](https://github.com/Skydeke/wiz3D/tree/custom_nvidia_3d_vision_output).

Two binaries share one USB emitter driver:

- **`nvstereo3d`** — host bridge. Reads eye-swap commands from wiz3D's
  `Nvidia3DOutput.dll` via shared memory and fires the IR emitter, synced to
  the monitor's real vblank.
- **`nvstereo-calibrate`** — tuning demo. Verifies stereo sync and tunes the
  per-monitor shutter timing.

## Why

Wine gives Windows DLLs no USB device access, so `Nvidia3DOutput.dll` can't
drive the emitter itself. It drops eye-swap commands into shared memory, and
`nvstereo3d` does the USB work on the Linux side.

Requires an **AMD** GPU: the emitter is synced to the kernel vblank clock
(`WAIT_VBLANK`), which the AMD drivers expose and the NVIDIA driver does not.

## Install

Arch: build the `-git` PKGBUILD in the repo's `build/` folder:

    git clone https://github.com/Skydeke/nvstereo3d
    cd nvstereo3d/build
    makepkg -sic

Running it from `build/` keeps every generated file inside that folder, out
of the checkout. The PKGBUILD clones the latest **pushed** HEAD over SSH
(your GitHub SSH key must be set up; push changes before rebuilding) and
installs both binaries plus the udev rule — re-plug the emitter after
installing.

Any distro: `cargo build --release`.

## Quick start

1. **Set the display to 120 Hz.** Enable 3D mode on a DLP projector, then pick
   the 120 Hz refresh in your desktop's display settings — on Wayland the
   compositor owns the display modes, so no xrandr modelines.
2. **Calibrate once** — `nvstereo-calibrate`. Switch scenes with `1`–`4`;
   start with 3, then 1, 2, 4:
   - Scene 3 (red/blue): close an eye — the whole screen shows only that eye's
     colour, full screen.
   - Scene 1 (hexagons/triangles): close an eye — the open eye sees its own
     pattern bright and the other dark/invisible.
   - Scene 2 (random-dot): the square should pop out in 3D toward you.
   - Scene 4 (pulsar): the rotating field should read as one solid 3D volume
     with real depth.
   - `t` selects the value to tune, `,`/`.` (fine) and `[`/`]` (coarse) adjust
     it, `k` sets the step size. Press `i` if the eyes are swapped.
   - Press `s` to save the profile.
3. **Start the host** — `nvstereo3d`.
4. **Play.** Launch the game in Wine/Proton with wiz3D's `Nvidia3DOutput`.
   Depth inverted? Press the emitter's 3D button once.

## Config

Timing profiles live in `monitor_timings.json` — read from the working
directory first (PWD takes precedence), otherwise from
`~/.config/nvstereo3d/`. Saving (`s`) writes to wherever an existing file was
found. Profiles are matched per monitor by EDID and applied by both binaries.

Already have a local DB? Move it to the per-user location:

    mkdir -p ~/.config/nvstereo3d
    mv monitor_timings.json ~/.config/nvstereo3d/

## Acknowledgements

- [eruffaldi/libnvstusb](https://github.com/eruffaldi/libnvstusb) — emitter USB protocol & firmware
- [NTM-3D/RP2040-3D-Vision-Emitter](https://github.com/NTM-3D/RP2040-3D-Vision-Emitter) — RP2040 clone emitter
- [lukis101/3DVisionAVR](https://github.com/lukis101/3DVisionAVR) — AVR clone emitter
- [rajkosto/NvTimingsEd](https://github.com/rajkosto/NvTimingsEd) — monitor timing editor