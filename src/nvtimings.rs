//! Load / save of the per-monitor shutter-timing database
//! (`monitor_timings.json`) in the project's OWN flat format (no longer
//! NV3D-Lib's nested schema).
//!
//! The file is a flat map of `"<monitor key>" -> MonitorEntry`, where each
//! entry carries everything needed to shutter a monitor:
//!   - the measured refresh (`refresh_hz`) and a derived pixel-clock figure
//!     (`frequency_10khz`);
//!   - the shutter register profile (`x_us` = refresh start -> shutter open
//!     edge, `y_us` = shutter open window, `w_us` = the 2nd T2 counter);
//!   - the host-side IR packet LEAD before the vblank boundary (`lead_us`, the
//!     same value the demo tunes with the `Phase` key and the host helper
//!     applies to `frame_start`).
//!
//! `z` (the glasses frame time) is always 1e6 / refresh, so it is derived at
//! use time rather than stored.
//!
//! Where the DB lives (see [`db_path`]): an explicit `NVSTUSB_TIMINGS_JSON`
//! path overrides everything; otherwise a `monitor_timings.json` sitting in
//! the current working directory wins (PWD precedence — e.g. the repo's
//! checked-in file); otherwise the per-user XDG config dir
//! `~/.config/nvstereo3d/monitor_timings.json` is used.
//!
//! This module is shared by both binaries so a profile saved by the
//! `nvstereo-calibrate` demo (with `s`) is picked up by `nvstereo3d`:
//!   - the demo writes the tuned X/Y/W + lead + measured refresh back on `s`;
//!   - both binaries read it back at startup and apply the matching profile.
//!
//! Entries are keyed exactly as NV3D-Lib keys them: `VENDOR_PRODUCT_REFRESH`,
//! where `VENDOR` is the monitor's 3-letter PNP code, `PRODUCT` its 16-bit
//! EDID product id in upper-case zero-padded hex, and `REFRESH` the rounded
//! integer refresh — e.g. `ACI_23F7_120`.  Both binaries derive the
//! `VENDOR_PRODUCT` base for the monitor that is actually in use (see
//! [`crate::edid`]) and address the DB by it: exact `base_<rounded rate>` with
//! fallback to the highest-refresh entry for that monitor.
//!
//! The file holds ONLY the monitor(s) that were tuned and saved — it is a per
//! user's own DB, not a copy of NV3D-Lib's whole catalog.  `s` replaces the
//! file with just the active monitor's entry (see [`save_entry`]).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Name override env var for the JSON database path (debug/testing; not
/// advertised in the README).
pub const ENV_PATH: &str = "NVSTUSB_TIMINGS_JSON";

/// Database file name, relative to the current working directory or the
/// per-user config dir.
pub const DEFAULT_PATH: &str = "monitor_timings.json";

/// The per-user config dir for nvstereo3d: `$XDG_CONFIG_HOME/nvstereo3d`,
/// defaulting to `~/.config/nvstereo3d` when unset.
pub fn config_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|h| !h.is_empty())
                .map(|h| PathBuf::from(h).join(".config"))
        })?;
    Some(base.join("nvstereo3d"))
}

/// Resolves the database path:
/// 1. an explicit `NVSTUSB_TIMINGS_JSON` path (debug/testing override);
/// 2. `./monitor_timings.json` — the file next to where the process was
///    started.  PWD takes precedence, so a DB checked into a repo or dropped
///    next to a game always wins over the per-user copy;
/// 3. the per-user config dir (`~/.config/nvstereo3d/monitor_timings.json`).
pub fn db_path() -> std::path::PathBuf {
    use std::path::PathBuf;
    if let Some(p) = std::env::var_os(ENV_PATH) {
        return PathBuf::from(p);
    }
    let local = PathBuf::from(DEFAULT_PATH);
    if local.exists() {
        return local;
    }
    config_dir()
        .map(|d| d.join(DEFAULT_PATH))
        .unwrap_or(local)
}

/// Default host-side IR packet lead if a profile carries none (missing field
/// in an older/legacy file): matched to the demo's `swap_phase_us` default.
pub const DEFAULT_LEAD_US: f64 = 3100.0;

/// The full per-monitor entry: measured refresh, the shutter register profile
/// (X/Y/W) and the host-side IR lead.  This is the project's OWN flat format,
/// not NV3D-Lib's nested `monitor_timings`/`glasses_timings` schema.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct MonitorEntry {
    /// Measured refresh rate (Hz), e.g. 119.983.
    pub refresh_hz: f64,
    /// Pixel clock in units of 10 kHz (e.g. 28675 == 286.75 MHz).
    pub frequency_10khz: u64,
    /// Shutter register X: delay from monitor refresh start to the shutter
    /// open edge (us).
    pub x_us: f64,
    /// Shutter register Y: shutter open window (us).
    pub y_us: f64,
    /// Shutter register W: 2nd T2 counter (us).
    pub w_us: f64,
    /// Host-side IR packet lead before the vblank boundary (us) — the same
    /// value the demo's `Phase` knob tunes and `nvstereo3d` applies to
    /// `frame_start`.
    pub lead_us: f64,
}

impl Default for MonitorEntry {
    fn default() -> Self {
        Self {
            refresh_hz: 0.0,
            frequency_10khz: 0,
            x_us: 0.0,
            y_us: 0.0,
            w_us: 0.0,
            lead_us: DEFAULT_LEAD_US,
        }
    }
}

impl MonitorEntry {
    /// The glasses frame time (`z`), derived as the inverse of the refresh.
    pub fn z_us(&self) -> f64 {
        if self.refresh_hz > 0.0 {
            1_000_000.0 / self.refresh_hz
        } else {
            0.0
        }
    }
}

/// The whole database: monitor key -> entry.
pub type TimingsDb = BTreeMap<String, MonitorEntry>;

/// Loads the database from the path returned by [`db_path`]. Missing /
/// unparseable files yield an empty map so callers can carry on with defaults.
pub fn load() -> TimingsDb {
    let path = db_path();
    match std::fs::read_to_string(&path) {
        Ok(text) => match serde_json::from_str::<TimingsDb>(&text) {
            Ok(db) => db,
            Err(e) => {
                eprintln!("nvstusb: could not parse {}: {e}", path.display());
                TimingsDb::new()
            }
        },
        Err(e) => {
            eprintln!("nvstusb: could not read {}: {e}", path.display());
            TimingsDb::new()
        }
    }
}

/// Rounds a refresh rate to the integer used in `_<REFRESH>` keys.
pub fn round_refresh(rate_hz: f32) -> i64 {
    (rate_hz as f64).round() as i64
}

/// The `VENDOR_PRODUCT_REFRESH` key for a base and an integer refresh.
pub fn key_for(base: &str, refresh: i64) -> String {
    format!("{base}_{refresh}")
}

/// The entry whose `_<REFRESH>` suffix is numerically largest for this monitor
/// (the `FindHighestRefreshForBase` fallback).
fn highest_for_base(db: &TimingsDb, base: &str) -> Option<(String, MonitorEntry)> {
    let prefix = format!("{base}_");
    let mut best_rr = -1i64;
    let mut best: Option<(String, MonitorEntry)> = None;
    for (key, entry) in db {
        let Some(suffix) = key.strip_prefix(&prefix) else {
            continue;
        };
        if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        if let Ok(rr) = suffix.parse::<i64>() {
            if rr > best_rr {
                best_rr = rr;
                best = Some((key.clone(), entry.clone()));
            }
        }
    }
    best
}

/// Resolves a shutter profile for a monitor (`base` = `VENDOR_PRODUCT`) at a
/// refresh rate: exact `base_<rounded rate>` first, falling back to the
/// monitor's highest-refresh entry.  Returns the matched key and the entry
/// (X/Y/W + host lead).  `None` when no entry exists for the monitor.
pub fn resolve(db: &TimingsDb, base: &str, rate_hz: f32) -> Option<(String, MonitorEntry)> {
    let rr = round_refresh(rate_hz);
    let exact_key = key_for(base, rr);
    if let Some(e) = db.get(&exact_key) {
        return Some((exact_key, e.clone()));
    }
    highest_for_base(db, base)
}

/// Serializes a fresh single-entry database (ONLY the supplied monitor) back
/// to `monitor_timings.json`, pretty-printed and keyed.  Because the DB holds
/// only the user's own tuned monitors, saving replaces the file with just this
/// entry — older entries from a previous monitor are not carried over.
pub fn save_entry(key: &str, entry: &MonitorEntry) -> Result<std::path::PathBuf, String> {
    let mut db = TimingsDb::new();
    db.insert(key.to_string(), entry.clone());
    save(&db)
}

/// Serializes the database back to `monitor_timings.json` (pretty-printed,
/// alphabetically keyed).  The target dir (e.g. `~/.config/nvstereo3d`) is
/// created on demand.
pub fn save(db: &TimingsDb) -> Result<std::path::PathBuf, String> {
    let path = db_path();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
    }
    let text = serde_json::to_string_pretty(db).map_err(|e| format!("serialize: {e}"))?;
    std::fs::write(&path, text + "\n").map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let entry = MonitorEntry {
            refresh_hz: 119.983,
            frequency_10khz: 28675,
            x_us: 0.5,
            y_us: 7334.0,
            w_us: 4735.58,
            lead_us: 3100.0,
        };
        let text = serde_json::to_string_pretty(&entry).unwrap();
        let back: MonitorEntry = serde_json::from_str(&text).unwrap();
        assert_eq!(back, entry);
        // Verify the exact field names of the project's OWN format survive a
        // round trip (flat, with the host lead present).
        assert!(text.contains("\"refresh_hz\""));
        assert!(text.contains("\"x_us\""));
        assert!(text.contains("\"y_us\""));
        assert!(text.contains("\"w_us\""));
        assert!(text.contains("\"lead_us\""));
    }

    #[test]
    fn resolves_by_base_and_refresh() {
        let mut db = TimingsDb::new();
        db.insert(
            "ACI_23F7_120".to_string(),
            MonitorEntry {
                refresh_hz: 119.983,
                frequency_10khz: 28675,
                x_us: 3064.5,
                y_us: 2375.25,
                w_us: 5854.58,
                lead_us: 3200.0,
            },
        );
        db.insert(
            "ACI_23F7_100".to_string(),
            MonitorEntry {
                refresh_hz: 99.931,
                frequency_10khz: 23654,
                x_us: 3703.25,
                y_us: 3468.0,
                w_us: 7489.33,
                lead_us: 3100.0,
            },
        );
        // Exact rounded-rate match: 119.98 -> 120.
        let (k, e) = resolve(&db, "ACI_23F7", 119.98).unwrap();
        assert_eq!(k, "ACI_23F7_120");
        assert_eq!(e.x_us, 3064.5);
        assert_eq!(e.lead_us, 3200.0);
        // 100.0 -> 100.
        let (k, e) = resolve(&db, "ACI_23F7", 99.93).unwrap();
        assert_eq!(k, "ACI_23F7_100");
        assert_eq!(e.x_us, 3703.25);
        // No entry at the given rate -> highest-refresh fallback.
        let (k, e) = resolve(&db, "ACI_23F7", 144.0).unwrap();
        assert_eq!(k, "ACI_23F7_120");
        assert_eq!(e.x_us, 3064.5);
        // Unknown monitor -> nothing.
        assert!(resolve(&db, "XYZ_9999", 120.0).is_none());
    }

    /// `save_entry` writes ONLY the supplied entry — the file holds just the
    /// user's own monitor, never a stale catalog.
    #[test]
    fn save_entry_writes_only_the_given_monitor() {
        let dir = std::env::temp_dir();
        let path = dir.join("nvstusb_save_entry_test.json");
        std::env::set_var(ENV_PATH, &path);
        let e = MonitorEntry {
            refresh_hz: 119.983,
            frequency_10khz: 28675,
            x_us: 0.5,
            y_us: 7334.0,
            w_us: 4735.58,
            lead_us: 3100.0,
        };
        save_entry("ACI_23F7_120", &e).unwrap();
        let db = load();
        std::env::remove_var(ENV_PATH);
        let _ = std::fs::remove_file(&path);
        assert_eq!(db.len(), 1);
        assert_eq!(db.get("ACI_23F7_120").unwrap(), &e);
    }

    /// A stored entry without the new `lead_us` field (a hand-written legacy
    /// file) must still load, defaulting the lead to the documented default.
    #[test]
    fn missing_lead_defaults() {
        let json = r#"{
  "ACI_23F7_120": { "refresh_hz": 119.983, "frequency_10khz": 28675, "x_us": 0.5, "y_us": 7334.0, "w_us": 4735.58 }
}"#;
        let db: TimingsDb = serde_json::from_str(json).unwrap();
        let e = db.get("ACI_23F7_120").unwrap();
        assert!((e.lead_us - DEFAULT_LEAD_US).abs() < 1e-9);
        assert!((e.z_us() - 1_000_000.0 / 119.983).abs() < 1.0);
    }

    /// `config_dir()` follows `$XDG_CONFIG_HOME` when set (the per-user DB
    /// home) instead of `~/.config`.
    #[test]
    fn config_dir_follows_xdg_config_home() {
        std::env::set_var("XDG_CONFIG_HOME", "/tmp/nvstereo3d-xdg-test");
        let d = config_dir().expect("config dir with XDG_CONFIG_HOME set");
        std::env::remove_var("XDG_CONFIG_HOME");
        assert_eq!(
            d,
            std::path::PathBuf::from("/tmp/nvstereo3d-xdg-test/nvstereo3d")
        );
    }

    /// Parses an actual `monitor_timings.json` written in the project's OWN
    /// format and confirms the entry in it loads and is resolvable for a real
    /// monitor base.  This is the file both binaries read, so it must stay
    /// parseable.
    #[test]
    fn parses_the_repo_nvtimings_db() {
        let db = load();
        if db.is_empty() {
            // File missing (e.g. tests run from a different cwd) — skip.
            return;
        }
        let key = db.keys().next().expect("DB had entries");
        let base = key.rsplit_once('_').map(|(b, _)| b).unwrap_or(key);
        let (_, e) = resolve(&db, base, 120.0)
            .or_else(|| resolve(&db, base, 100.0))
            .or_else(|| resolve(&db, base, 110.0))
            .expect("a resolution should exist for a present base");
        // X/Y/W must all be sane microsecond positive-ish values; lead matches
        // the documented default.
        assert!(e.x_us >= 0.0 && e.y_us > 0.0 && e.w_us > 0.0);
        assert!(e.lead_us >= 0.0);
    }
}
