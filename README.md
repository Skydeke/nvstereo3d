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
The `,`/`.`/`[`/`]` keys tune the per-monitor shutter timing registers
(X/Y/W, microseconds) rather than the host packet lead - see "Per-monitor
shutter timings" below; the host lead stays where the environment put it
(`NVSTUSB_PHASE_US`). `k` cycles how coarse the keys are
(100 -> 1000 -> 10 -> 1 us; 3DVisionActivator's `I` toggle). On a standard
sample-and-hold panel the mis-phased state is visible as a horizontal colour
split that slides up or down with each press - park it off-screen. On a
monitor with motion-blur reduction (DyAc/ELMB/ULMB backlight strobing) the
same keys tend to swap the colours wholesale instead: the backlight only
fires briefly after scanout completes, so there is no partial-scanout light
and the shutter edge either misses the flash (no change) or crosses it
(colour swap). Aim for the middle of a solid-colour stretch there; that
plateau is the jitter-safe setting.

**How the windowed anchor is chosen (vendor/driver-agnostic).** Method 1
needs a hardware-accurate boundary. It prefers the kernel DRM-vblank clock;
when that is unavailable on a composited desktop it falls back to
`wp_presentation_feedback` — the compositor's own report of when each frame
really hit the screen — attached to winit's `wl_surface` via a guest adopt of
winit's Wayland connection (`presentation_fb`). Because the *compositor*, not
the GPU driver, supplies these timestamps, they are valid on every GPU vendor
and every NVIDIA branch: AMD/Intel under Wayland (where a plain client cannot
queue CRTC vblank waits the compositor owns), NVIDIA 610+ (before `vblank=1`),
and NVIDIA 390/595 (where DRM vblank is compiled out entirely). Only when
neither mechanism is available does pacing fall back to the swap-return
callback clock. The perf report's `anchor` field names the one in use
(`drm-vblank[...]`, `wp_presentation_feedback`, or `swap-return`).

Common keys (all scenes): `Esc`/`q` quit, `c` camera type, `f` force eye,
`s` save the tuned shutter timings to `monitor_timings.json` (see "Per-monitor
shutter timings" below), `S` screenshot, `,`/`.` (fine) and `[`/`]`
(coarse, x10) tune the selected
shutter timing, `k` cycles the step size (100/1000/10/1 us), `t` cycles
which of X/Y/W/LEAD (the host packet lead) they adjust, `i` eye swap,
`o` cycle sync anchor display
(multi-monitor fallback), `1`/`2`/`3`/`4`
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

### Per-monitor shutter timings (X/Y/W)

3DVisionActivator ships one `MonitorTimings.ini` profile per monitor/refresh
with four emitter timing registers, all in microseconds:

- `X` — delay from the monitor's refresh start to the shutter open edge.
  This is a T0 timer (4 MHz) reload; libnvstusb's firmware notes describe it
  as "timer 0 will be started with this value by timer 2".
- `Y` — shutter open window, the delay until the eye is turned back off
  (also a T0 reload; the wiring notes say "delay until turning eye off?").
- `Z` — full frame time == 1/refresh, the T2 (12 MHz) "timer 2 reload value"
  that keeps the frame period. For a given monitor the rate is fixed, so Z
  is, too.
- `W` — a second T2 timer counter ("some timer 2 counter, 1020 is subtracted
  from this, loaded at startup"). Every per-monitor profile stores one, and
  genuinely different values with refresh rate, but 3DVisionActivator's own
  author measured "no effect" on the panels he tried — a real register, not
  dead code, just usually the one you leave alone.

The demo carries the same profile and programs the emitter's 0x2007 timing
registers:

- Defaults: the 1440p @ 120 Hz reference (X=0.5, Y=7334.0, W=4735.58 us -
  the same numbers as the `Samsung LC27G5xT 1440p 120Hz` profile in the
  bundled 3DVisionActivator `MonitorTimings.ini`).
- Load a whole file: `NVSTUSB_TIMINGS_INI=/path/to/MonitorTimings.ini`
  (3DVisionActivator format; the profile whose `RefreshRateHz:` matches the
  monitor's measured rate wins, since the demo binds `Z` to the real
  refresh).
- Per-value overrides (floats, us): `NVSTUSB_X_US=...`,
  `NVSTUSB_Y_US=...`, `NVSTUSB_W_US=...`.
- `monitor_timings.json`: the demo and `nvstereo3d-host` both read the repo's
  per-monitor database in the project's OWN flat format (point to another file
  with `NVSTUSB_TIMINGS_JSON=/path`). Entries are keyed exactly as
  NV3D-Lib keys them — `VENDOR_PRODUCT_REFRESH`, e.g. `ACI_23F7_120`, where
  `VENDOR` is the monitor's 3-letter EDID PNP code and `PRODUCT` its 16-bit EDID
  product id — and both binaries derive that identity from the active monitor's
  DRM EDID (read from sysfs `/sys/class/drm/*/edid`). Each entry carries the
  measured refresh, the X/Y/W shutter registers, **and the host IR lead**
  (`lead_us`, the packet-before-vblank lead the `LEAD`/Phase knob tunes):
  ```json
  {
    "ACI_23F7_120": {
      "refresh_hz": 119.983,
      "frequency_10khz": 28675,
      "x_us": 0.5,
      "y_us": 7334.0,
      "w_us": 4735.58,
      "lead_us": 3100.0
    }
  }
  ```
  Lookup is exact (`_<rounded rate>`) then falls back to the monitor's
  highest-refresh entry, so a profile saved for one rate still applies if the
  mode's rounded rate differs. Press **`s`** in the demo to save the currently
  tuned X/Y/W + lead + refresh to `monitor_timings.json` — the file holds ONLY your
  own tuned monitor(s), so `s` replaces it with just that entry. It is then
  picked up again on the next `cargo run` *and* by `nvstereo3d-host`: once the
  host `configure`s the emitter for the rate the game reports, it resolves the
  same EDID identity and applies the matching X/Y/W profile *and* lead
  automatically (the host auto-detects which monitor is the stereo head from
  the DRM vblank anchor; `NVSTUSB_HOST_LEAD_US` overrides the stored lead). The
  demo's startup precedence is:
  `NVSTUSB_TIMINGS_INI` < `monitor_timings.json` < `NVSTUSB_*_US` env overrides (env
  wins). The exact key the tuned profile would be saved under (e.g.
  `ACI_23F7_120`) is shown on the bottom-left HUD (`JSON <...>` line) and in
  the `[perf]`/`[monitor]` logs, so the on-disk name is always visible.
  Under Wayland the wl_output isn't known until a frame or two after startup,
  so the JSON profile is (re-)applied as soon as the output resolves in the
  `[monitor]` recheck — which is what actually loads a saved profile on a
  typical first `cargo run`.
- Live tuning: `,`/`.` (fine) and `[`/`]` (coarse, x10) bump the currently
  selected value by the armed step and program the emitter immediately; `k`
  cycles the step 100 -> 1000 -> 10 -> 1 us; `t` cycles X -> Y -> W -> LEAD.
  X (delay from refresh start to the shutter open edge) is the primary
  band-position knob. **LEAD** tunes the host-side packet lead — how many us
  before the next vblank boundary the eye packet is sent — which is the
  genuine-emitter way to nudge the shutter edge when the timing-register
  readback is unverified but still observable. (On a firmware that ignores the
  register block, LEAD is the knob that actually still does something.)

The current X/Y/W/LEAD values, the armed parameter, the step size, the refresh
rate and whether the emitter honors the timing registers are shown in the
bottom-left on-screen debug HUD (bitmap text overlay) and in the per-second
`[perf]` line, so tuning stays readable on the AltBlink checker without a
terminal.

**Why the keys may do nothing — genuine vs clone emitters.** Only the
genuine NVIDIA firmware implements a shutter-timing *generator* that consumes
these registers: it stores X/Y/W/Z at startup, and every monitor profile
re-programs them. Clone emitters that don't need the `.fw` upload (e.g.
RP2040-based) usually implement only the per-frame packet path — they accept
and ignore the 0x2007 block, so X/Y/W writes visibly change nothing. At
startup the app verifies this the way 3DVisionActivator does, by reading the
register block back: a real firmware answers with the stored values
(`TIMING REGS LIVE` in the HUD), a clone stays silent (`TIMING REGS
IGNORED`).

If you want the *exact replica* behavior the per-monitor profiles were
written for, run the genuine firmware: a bare Cypress CY7C68013A (FX2LP)
dongle plugs in as an otherwise-unconfigured EZ-USB, the app uploads the
genuine `nvstusb.fw` at startup, and X/Y/W tuning then has the same effect
as on an original NVIDIA emitter.

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

### Enabling the anchor on NVIDIA

nvstusb waits on the modern, driver-agnostic CRTC-sequence ioctls
(`CRTC_GET_SEQUENCE`/`CRTC_QUEUE_SEQUENCE`), which the DRM core serves on
every KMS driver — but only if the driver actually initialized its vblank
infrastructure via `drm_vblank_init`.  Whether nvidia-drm does that depends
entirely on the driver branch:

- **600 / 610 series and newer** — nvidia-drm *can* init vblank, but it is
  **off by default**, gated behind a module option named `vblank`.  Until it
  is set, *every* vblank ioctl — legacy `WAIT_VBLANK` and the modern
  CRTC-sequence ones — returns `EOPNOTSUPP` (os error 95) and the anchor
  cannot lock on.  Fix:

  ```
  sudo cp 98-nvidia-drm.conf /etc/modprobe.d/
  # then reboot, or reload just the DRM glue:
  sudo modprobe -r nvidia_drm && sudo modprobe nvidia_drm
  # verify the option actually took (must print Y, not N):
  cat /sys/module/nvidia_drm/parameters/vblank
  ```

  The option is `vblank=1` on the `nvidia-drm` module — **not** the C variable
  name `nvidia_drm_vblank`, and **not** `modeset=1` (the option NVIDIA registers
  is `module_param_named(vblank, nv_drm_vblank_module_param, bool, 0400)` in
  `nvidia-drm-linux.c`; writing `nvidia_drm_vblank=1` is silently ignored and
  leaves vblank disabled with only a harmless "unknown parameter" warning).
  Internally, `vblank=1` starts a "nvidia-drm-vblank-notification" worker
  thread that software-synthesizes the DRM vblank counter from NVKMS flip
  events — so the counter only advances while the CRTC is actually flipping
  (a compositor or our KMS app doing so is enough).

- **595.x and older** (including the legacy **390.x** branch for older GPUs) —
  there is **no** `vblank` module option at all.  On modern kernels (≥ 4.19)
  nvidia-drm skips `drm_vblank_init` — its code gates it behind
  `#if !defined(NV_DRM_CRTC_STATE_HAS_NO_VBLANK)`, and that macro is *defined*
  since the kernel gained `drm_crtc_state.no_vblank` in Linux 4.19 — with no
  knob to turn it back on.  This is the historical consensus NVIDIA engineers
  have stated publicly: vblank is "not exposed by NVIDIA drivers on Linux".
  So on a ≥ 4.19 kernel the DRM vblank anchor is simply **unavailable** on
  595/390 — upgrade to 600/610+ (or remove the `vblank=1` line; it does not
  exist on these branches). On the **windowed Wayland path** the demo instead
  falls back to the `wp_presentation_feedback` anchor (see above), which works
  on 595/390 since the compositor — not nvidia-drm — supplies the present
  timestamps; on the **VT/KMS path** (where there is no compositor) it runs
  with the swap-return anchor.

  The one exception is an **ancient kernel < 4.19** (e.g. a Linux 4.15-era
  install, not uncommon on an old 390 laptop): there the macro is *not*
  defined, so `drm_vblank_init` runs **unconditionally on every branch** —
  390 included — and the DRM vblank anchor **works out of the box with no
  module option**.  That is the case where "the old 390 laptop should also
  work" is true, and nothing needs to be configured.  The catch is purely the
  DKMS kernel: ≥ 4.19, no vblank; < 4.19, vblank for free.

After enabling on 610+, the anchor log line should show `CRTC-sequence`
(instead of falling back to `legacy WAIT_VBLANK`).  If nvidia-drm also denies
connector enumeration (log: `connector enumeration unavailable`), press `o` in
the demo to cycle the anchor pipe until the blue/red scene separates cleanly.

The emitter's **3D button** toggles manual eye inversion while a game runs
(the host-side equivalent of the demo's `i` key). A systematic one-frame
offset between the game's present-to-scanout pipeline depth and the fire grid
is invisible to every automatic check - both streams stay self-consistent -
so if depth perception says the eyes are swapped, press it once.

### Following the window across heads (re-anchor)

Each CRTC free-runs with its own phase, so the anchor MUST sit on the head the
game window is actually on.  At startup it picks the first usable head (or
`NVSTUSB_ANCHOR_OUTPUT=`), and once the window owner publishes the target
display into the shared header the helper **re-anchors** when it changes:

- The wiz3D `Nvidia3DOutput.dll` publishes the target monitor's EDID
  `VENDOR_PRODUCT` base (e.g. `SAM_707A`) in the shared-memory header, and the
  helper reverse-resolves it to the DRM connector carrying that EDID.
- A concrete connector name (e.g. `DP-2`) is honored directly, so a Linux-side
  window owner that knows its exact connector (the 3dv3d demo's
  `current_monitor()` name) drives the same re-anchor unambiguously.

On a move the helper logs `window moved to <conn> ... re-anchored`, re-opens
the vblank anchor on the new head and re-applies that head's `monitor_timings.json`
profile.  Limitation: two connectors sharing one physical panel/EDID (e.g.
DP-1 and DP-2 both `SAM_707A`) are indistinguishable by EDID, so an EDID-base
publish resolves to the first matching active head; only an exact connector
name fully disambiguates that rare same-panel setup.

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
- `src/text.rs` — 5x7 bitmap-font overlay (scene labels + bottom-left timing HUD)
- `firmware/nvstusb.fw` — emitter firmware image, embedded at build time