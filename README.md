# nvstereo3d — NVIDIA 3D Vision for Linux

One Rust project with **two binaries** that both drive the same NVIDIA 3D
Vision USB IR emitter:

| Binary            | What it is |
|-------------------|------------|
| `3dv3d` (default) | The OpenGL stereo demo, in sync with the display's real vblank through the shared `nvstusb` emitter driver |
| `nvstereo3d-host` | The wiz3D bridge: consumes the shared-memory ring that `Nvidia3DOutput.dll` feeds and fires each eye packet anchored to the DRM vblank clock |

`cargo run` builds the default binary (`3dv3d`); build a specific target with
`cargo build --bin <name>`.

## Build

    cargo build --release        # both binaries
    cargo build --bin 3dv3d
    cargo build --bin nvstereo3d-host

The shared lib (`nvstereo3d`) is compiled once and used by both binaries, so
`drm.rs` and `usb.rs` live in a single place (`src/nvstusb/`) rather than
being duplicated across the two programs.

## 3dv3d — the demo

Scene keys:

| Key | Scene                   |
|-----|-------------------------|
| `1` | hexagon / triangles diagnostic pattern (default) |
| `2` | "pulsar" (Paul Bourke's scene) |
| `3` | random-dot stereogram (`medimg`) |
| `4` | alternating blue/red sync checker |

Startup tries the **KMS/DRM** backend first (renders straight to the display
engine; best on a bare VT with the projector at 120 Hz). If it cannot take the
display it falls back to the windowed path (winit/glutin on the Wayland
compositor):

    ./target/release/3dv3d

Headless sanity check without the emitter and display:

    NVSTUSB_KMS=1 ./target/release/3dv3d --no-emitter

Common keys (all scenes): `Esc`/`q` quit, `c` camera type, `f` force eye,
`s` screenshot, `,`/`.`/`[`/`]` shutter phase, `i` eye swap,
`1`/`2`/`3`/`4` scene switch.

## nvstereo3d-host — wiz3D bridge

wiz3D's `Nvidia3DOutput.dll` (under Wine/Proton) pushes one eye-swap command
per presented frame into a shared-memory ring at `/tmp/nvstusb.shm` and sends a
one-byte UDP datagram to wake the helper. The helper owns the emitter, consumes
the ring, and fires the shutter packet anchored to the display engine's real
vblank clock (`DRM_IOCTL_WAIT_VBLANK`) so the glasses lock in master mode.

    ./target/release/nvstereo3d-host

Environment (all optional):

| Var | Default | Meaning |
|-----|---------|---------|
| `NVSTUSB_SHM_PATH` | `/tmp/nvstusb.shm` | shared-memory ring path |
| `NVSTUSB_HOST_PORT` | `8777` | UDP wake port |
| `NVSTUSB_POLL_MS` | `5` | wake-socket timeout before re-checking the ring |
| `NVSTUSB_DEBUG` | off | per-second packet/period telemetry |

Needs the USB emitter accessible (run as root or via the bundled
`98-nvstusb.rules` udev rule).

Diagnostics: `cargo run --example consume` opens the ring as a consumer and
prints config plus any pending eye-swaps. `cargo test` exercises the ring
(wraparound, full-ring rejection, config round-trip).

## Structure

- `src/lib.rs` — the demo (`run_demo`), plus `pub mod shm` / `pub mod host`
- `src/nvstusb/` — **shared** emitter driver: `usb.rs` (libusb transport +
  configure/`send_eye`/`set_alarm_delay_us`, single copy) and `drm.rs`
  (vblank anchor, single copy), used by both binaries
- `src/shm.rs` — shared-memory ring (host)
- `src/host.rs` — host-helper logic (`host::run`)
- `src/bin/3dv3d.rs`, `src/bin/nvstereo3d-host.rs` — thin `main` entries
- `src/scene.rs`, `src/pulsar.rs`, `src/medimg.rs` — demo scenes
- `firmware/nvstusb.fw` — emitter firmware image, embedded at build time