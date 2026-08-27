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
| `1` | hexagon / triangles diagnostic pattern (default; each eye's frame is labelled on screen) |
| `2` | random-dot stereogram (`medimg`) |
| `3` | alternating blue/red sync checker |
| `4` | "pulsar" (Paul Bourke's scene) |

Startup tries the **KMS/DRM** backend first (renders straight to the display
engine; best on a bare VT with the projector at 120 Hz). If it cannot take the
display it falls back to the windowed path (winit/glutin on the Wayland
compositor):

    ./target/release/3dv3d

Headless sanity check without the emitter and display:

    NVSTUSB_KMS=1 ./target/release/3dv3d --no-emitter

Scene assets that are expensive to build (the `medimg` random-dot field,
the `pulsar` display list) are compiled up front at startup and re-built on
window resize, so a mid-run `1`/`2`/`3`/`4` scene switch never stalls the
swap loop — a lazy first build de-phases the shutter packets and shows a
wrong-eye/wrong-depth flash (e.g. the RDS square fusing behind the screen)
until the stream re-locks.

Emitter packet timing: the windowed path (vblank method 1) paces each eye
packet against the boundary where the frame will *actually* appear. After
every swap it confirms the present via GLX_OML_sync_control's swap-block
counter and re-anchors to that instant; if rendering finishes too late for
the predicted present, the packet and the present slip one period together,
so glasses and content stay paired through compositor hiccups instead of
falling into permanent inversion. The IR flip lands a fixed ~3 ms lead
before its frame edge - the same contract as the NVIDIA Windows driver.
Tune the residual phase with `,`/`.` (coarse with `[`/`]`); the value is
the packet lead in microseconds. On a standard sample-and-hold panel the
mis-phased state is visible as a horizontal colour split that slides up or
down with each press - park it off-screen. On a monitor with motion-blur
reduction (DyAc/ELMB/ULMB backlight strobing) the same keys tend to swap
the colours wholesale instead: the backlight only fires briefly after
scanout completes, so there is no partial-scanout light and the shutter
edge either misses the flash (no change) or crosses it (colour swap).
Aim for the middle of a solid-colour stretch there; that plateau is the
jitter-safe setting.

Common keys (all scenes): `Esc`/`q` quit, `c` camera type, `f` force eye,
`s` screenshot, `,`/`.`/`[`/`]` shutter phase, `i` eye swap,
`o` cycle sync anchor display (multi-monitor fallback), `1`/`2`/`3`/`4`
scene switch. On the medimg RDS scene, `+`/`-` move the square's pop-out
closer/further and `a`/`d` push the background's convergence.

The default hexagon/triangle scene labels each eye's frame on screen, in
that eye's own colour: the left lens should show `LEFT: GREEN HEXAGONS`,
the right lens `RIGHT: BLUE TRIANGLES`. The alternating red/blue sync
checker labels itself the same way, in white text: left lens `LEFT: RED`,
right lens `RIGHT: BLUE`. If a lens shows the other pattern or colour
(wrong label or wrong eye), the eyes are swapped — press `i`.

On a single monitor the sync anchor picks the right head automatically. With
several monitors on one GPU it binds to the one showing the window; if the
driver does not expose that mapping (nvidia-drm denies connector queries to
non-master clients - the log then says `connector enumeration unavailable`),
press `o` while on the blue/red scene until the split disappears: each press
re-binds the anchor to the next CRTC pipe and wraps back to auto.

## nvstereo3d-host — wiz3D bridge

wiz3D's `Nvidia3DOutput.dll` (under Wine/Proton) pushes one eye-swap command
per presented frame into a shared-memory ring at `/tmp/nvstusb.shm` and sends a
one-byte UDP datagram to wake the helper. The helper owns the emitter and fires
each packet anchored to the display engine's real vblank clock
(`DRM_IOCTL_WAIT_VBLANK`) so the glasses lock in master mode.

    ./target/release/nvstereo3d-host

### Where the eye comes from (and why not from the ring)

The helper fires a **strict L/R alternator on the vblank grid** — the same
cadence the 3dv3d demo proves out. It does NOT pop eyes off the ring:

- wiz3D enqueues swaps as an unconditional L,R,L,R sequence per game frame, so
  ring order carries no phase information about what actually scanned out.
- With vsync-blocking presents a swap arrives only AFTER its frame has already
  flipped, i.e. after any fire deadline that could have paired with it.
  Content-following the ring therefore fires every eye one slot late
  (persistent inversion) and flickers correct/inverted under jitter — seen as
  both images in both eyes.

The ring is still read every slot: the swap COUNT vs packet COUNT detects
dropped/doubled presents, and a persistent discrete slip is corrected with a
single same-eye pair (~200 ms hold). A sustained rate mismatch (presents not
landing one per vblank: coalescing, half-rate content, SyncInterval=0) cannot
be shutter-corrected by anything on this side; it is reported loudly in the
per-second telemetry instead and must be fixed upstream (force vsync / full-
rate presents).

Environment (all optional):

| Var | Default | Meaning |
|-----|---------|---------|
| `NVSTUSB_SHM_PATH` | `/tmp/nvstusb.shm` | shared-memory ring path |
| `NVSTUSB_HOST_PORT` | `8777` | UDP wake port |
| `NVSTUSB_POLL_MS` | `5` | wake-socket timeout before re-checking the ring |
| `NVSTUSB_HOST_LEAD_US` | `250` | pre-boundary fire lead; the shutter flips this many us before the vblank (minus USB write time). Raise/lower if games show edge ghosting the demo does not - same tuning as the demo's `,`/`.` |
| `NVSTUSB_DEBUG` | off | per-second packet/period telemetry |

Reading the per-second line (`N/s packets | M/s swaps | ...`):

- packets ≈ swaps ≈ display Hz, alternating N/N, `IN-LOCK-WINDOW` — healthy;
  residual wrong-eye means the constant pipeline offset: press the button.
- `SWAP STARVATION: M swaps/s vs N packets/s` — Nvidia3DOutput is enqueueing
  far below display rate (game paused/menu, stereo disengaged into mono
  fallback, or presents coalescing under Wine). The emitter cannot follow
  content the screen is not showing; fix the present path upstream.
- period min/max outside 7600–9000 us or same-eye repeats — glasses unlocked;
  check the anchor log line (`DRM vblank anchor on ...`) and USB latency.

The emitter's **3D button** toggles manual eye inversion while a game runs
(the host-side equivalent of the demo's `i` key). A systematic one-frame
offset between the game's present-to-scanout pipeline depth and the fire grid
is invisible to every automatic check - both streams stay self-consistent -
so if depth perception says the eyes are swapped, press it once.

Needs the USB emitter accessible (run as root or via the bundled
`98-nvstusb.rules` udev rule).

### Emitter detection (integrated emitters, overrides)

Candidates are tried in a fixed preference order, and the FIRST one whose
full setup succeeds wins:

1. An explicit `NVSTUSB_VID`/`NVSTUSB_PID` pin - absolute, no fallback.
2. The external dongle id `0955:0007`, which custom clones also use.  With
   your RP2040 emitter and the laptop's integrated unit both attached, the
   custom dongle is always picked.
3. Any other NVIDIA-vendor unit on the bus (integrated emitters; known
   alternate PIDs first, the same list 3DVisionActivator probes).

A higher-priority device that is present but fails mid-setup (firmware
upload, re-enumeration, claim) is logged and skipped - the next candidate
still runs instead of the session dying.

Integrated units run from flash: unlike a bare EZ-USB dongle they are NOT
firmware-loaded; like the original libnvstusb we only set configuration 1
and claim the interface.  A bare dongle gets `nvstusb.fw` uploaded and is
then polled for up to ~6 s while it re-enumerates (a single fixed delay
loses whenever the kernel re-probes the port slowly).  If `lsusb` shows
nothing NVIDIA-shaped with all other emitters unplugged, the internal unit
is not wired to an accessible USB port and cannot be driven by this stack.
2. Found it under another id? Try it directly:
   `NVSTUSB_VID=xxxx NVSTUSB_PID=yyyy ./target/release/3dv3d`
   (same protocol is attempted; whether an integrated unit answers it is up
   to its firmware).
3. `present but cannot be opened (permissions)` means the udev rule is not
   installed: copy `98-nvstusb.rules` to `/etc/udev/rules.d/`, run
   `udevadm control --reload && udevadm trigger` and re-plug.
4. Clones and genuine units share the same `0955:0007` id - with several of
   them connected at once the preference order above decides; pin
   `NVSTUSB_PID=` to force one deterministically.

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
- `src/text.rs` — 5x7 bitmap-font overlay (scene labels)
- `firmware/nvstusb.fw` — emitter firmware image, embedded at build time