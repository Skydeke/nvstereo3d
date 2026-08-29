//! EDID → `VENDOR_PRODUCT` monitor key, the exact mirror of NV3D-Lib's
//! `ParseMonitorEdid` (https://github.com/oneup03/NV3D-Lib).  NV3D-Lib reads
//! the raw EDID through `NvAPI_DISP_GetEdidData`; on Linux the same bytes live
//! in sysfs at `/sys/class/drm/<card>-<connector>/edid`, which any process
//! can read (no master/ioctl privileges needed).
//!
//! The key is built from the two identity words in the EDID base block:
//!   - bytes 8-9: the 3-letter PNP vendor code, packed with five bits per
//!     letter in base-5 ("A"-1 = 1 .. "Z"-1 = 26);
//!   - bytes 10-11: the 16-bit product/manufacturer code, little-endian.
//!
//! Both binaries derive this key for the monitor that is actually in use (the
//! demo from the wl_output / KMS connector it picked, the host from its DRM
//! anchor connector) and address `monitor_timings.json` by it — so an entry saved
//! under e.g. `ACI_23F7_120` is found again on any later run, exactly the way
//! NV3D-Lib's `FindByBaseAndRefresh` / `FindHighestRefreshForBase` work.

/// Decoded identity words from an EDID base block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonitorIdentity {
    /// 3-letter PNP vendor code, e.g. `"ACI"`.
    pub vendor: String,
    /// 16-bit product code, e.g. `0x23F7`.
    pub product: u16,
}

impl MonitorIdentity {
    /// The `VENDOR_PRODUCT` base key (e.g. `"ACI_23F7"`), matching NV3D-Lib's
    /// formatting: lowercase-letters vendor, then `_`, then the product id as
    /// uppercase zero-padded 4-digit hex.
    pub fn base_key(&self) -> String {
        format!("{}_{:04X}", self.vendor, self.product)
    }
}

/// Parses a raw EDID base block (at least 128 bytes) into [`MonitorIdentity`].
/// Returns `None` if the block is too short to be a valid EDID.
pub fn parse(edid: &[u8]) -> Option<MonitorIdentity> {
    if edid.len() < 128 {
        return None;
    }
    // Same byte math and layout as NV3D-Lib's ParseMonitorEdid.
    let vendor_id = (u16::from(edid[8]) << 8) | u16::from(edid[9]);
    let vendor: String = [
        ((vendor_id >> 10) & 0x1F) as u8,
        ((vendor_id >> 5) & 0x1F) as u8,
        (vendor_id & 0x1F) as u8,
    ]
    .iter()
    .map(|&b| (b + b'A' - 1) as char)
    .collect();
    let product = (u16::from(edid[11]) << 8) | u16::from(edid[10]);
    Some(MonitorIdentity { vendor, product })
}

/// Reads the raw EDID for a kernel connector (e.g. `"DP-1"`) from sysfs, and
/// decodes it.  `card` is optional: when given (e.g. `/dev/dri/card0`) only
/// that card's connector directories are searched; otherwise the first match
/// across all cards is used.  Returns `None` if no EDID is readable.
pub fn read_connector(connector: &str, card: Option<&str>) -> Option<MonitorIdentity> {
    let edid = read_connector_bytes(connector, card)?;
    parse(&edid)
}

/// Locates and reads the raw EDID bytes for a kernel connector from sysfs.
/// Connector matching is case-insensitive (the kernel uses `DP-1`, but callers
/// may pass `dp-1`).
pub fn read_connector_bytes(connector: &str, card: Option<&str>) -> Option<Vec<u8>> {
    let sys = "/sys/class/drm";
    let dir = std::fs::read_dir(sys).ok()?;
    let want = format!("-{}", connector.to_lowercase());
    for ent in dir.flatten() {
        let name = ent.file_name().to_string_lossy().into_owned();
        // Connector dirs look like `card0-DP-1`; skip the non-connector
        // `card0` / `renderD128` / `version` etc. entries.
        if let Some(card_prefix) = card {
            // card may be given as `/dev/dri/card0` or `card0` or `0`.
            let short = card_prefix
                .trim_start_matches("/dev/dri/")
                .trim_start_matches("card")
                .to_string();
            if !name.starts_with(&format!("card{short}-")) {
                continue;
            }
        } else if !name.contains('-') || !name.starts_with("card") {
            continue;
        }
        if name.to_lowercase().ends_with(&want) {
            if let Ok(bytes) = std::fs::read(format!("{sys}/{name}/edid")) {
                if !bytes.is_empty() {
                    return Some(bytes);
                }
            }
        }
    }
    None
}

/// Returns whether a kernel connector with the given name exists in sysfs
/// (case-insensitive), i.e. whether the name is a real live connector rather
/// than a synthetic/EDID-base identifier.  Used to decide whether a value
/// published into the shared header is a concrete connector name or an EDID
/// `VENDOR_PRODUCT` base to reverse-resolve.
pub fn connector_exists(connector: &str) -> bool {
    let sys = "/sys/class/drm";
    let want = format!("-{}", connector.to_lowercase());
    let Ok(dir) = std::fs::read_dir(sys) else {
        return false;
    };
    for ent in dir.flatten() {
        let name = ent.file_name().to_string_lossy().into_owned();
        if name.starts_with("card") && name.contains('-') && name.to_lowercase().ends_with(&want) {
            return true;
        }
    }
    false
}

/// Convenience for callers that know the kernel connector name and optionally
/// the card: reads + decodes the EDID and returns the `VENDOR_PRODUCT` base
/// key, or `None` if the identity can't be determined (e.g. no EDID in sysfs).
pub fn resolve_base(connector: &str, card: Option<&str>) -> Option<String> {
    read_connector(connector, card).map(|id| id.base_key())
}

/// Reverse lookup: given a `VENDOR_PRODUCT` base key (e.g. `"SAM_707A"`),
/// returns the kernel connector name (e.g. `"DP-2"`) of a connector whose
/// EDID decodes to that base.  Connected/enabled connectors are preferred over
/// disconnected ones, and within a base the first connected match wins.  This
/// lets the helper re-anchor when the window owner publishes only the monitor's
/// EDID base rather than a concrete connector name.  Returns `None` when no
/// connector carries an EDID matching `base`.
pub fn find_connector_by_base(base: &str) -> Option<String> {
    let want = base.trim().to_lowercase();
    if want.is_empty() {
        return None;
    }
    let sys = "/sys/class/drm";
    let mut connected_match: Option<String> = None;
    let mut any_match: Option<String> = None;
    for ent in std::fs::read_dir(sys).ok()?.flatten() {
        let dir = ent.file_name().to_string_lossy().into_owned();
        // Connector dirs look like `card0-DP-1`; skip `card0`, `renderD128` etc.
        if !dir.starts_with("card") || !dir.contains('-') {
            continue;
        }
        let path = format!("{sys}/{dir}");
        let Some(bytes) = std::fs::read(format!("{path}/edid")).ok() else {
            continue;
        };
        if bytes.is_empty() {
            continue;
        }
        let Some(id) = parse(&bytes) else {
            continue;
        };
        if id.base_key().to_lowercase() != want {
            continue;
        }
        // Strip the `cardN-` prefix to get the bare connector name (DP-1).
        let name = dir.split_once('-').map(|(_, n)| n.to_string()).unwrap_or(dir);
        let connected = std::fs::read_to_string(format!("{path}/status"))
            .map(|s| s.trim() == "connected")
            .unwrap_or(false);
        if connected {
            if connected_match.is_none() {
                connected_match = Some(name);
            }
        } else if any_match.is_none() {
            any_match = Some(name);
        }
    }
    connected_match.or(any_match)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid EDID base block with a known vendor/product, so the byte
    /// math is checked against NV3D-Lib's decoding, not just self-consistency.
    fn edid_with(vendor_hi: u8, vendor_lo: u8, product_hi: u8, product_lo: u8) -> Vec<u8> {
        let mut e = vec![0u8; 128];
        e[0] = 0x00;
        e[7] = 0x01; // reserved (version header byte present)
        e[8] = vendor_hi;
        e[9] = vendor_lo;
        e[10] = product_lo;
        e[11] = product_hi;
        e
    }

    // ACI -> letters A=1,C=3,I=9 -> id = 1<<10 | 3<<5 | 9 = 1024+96+9 = 0x469.
    // ACI_23F7 => bytes 8-9 = 0x04,0x69 ; product 0x23F7 => bytes 10-11 little.
    #[test]
    fn decodes_aci_23f7() {
        let e = edid_with(0x04, 0x69, 0x23, 0xF7);
        let id = parse(&e).unwrap();
        assert_eq!(id.vendor, "ACI");
        assert_eq!(id.product, 0x23F7);
        assert_eq!(id.base_key(), "ACI_23F7");
    }

    #[test]
    fn rejects_short_block() {
        assert!(parse(&[0u8; 127]).is_none());
    }

    #[test]
    fn formatting_zero_pads_and_uppercases_product() {
        // product 0x000A must render as "000A".
        let id = MonitorIdentity {
            vendor: "XYZ".into(),
            product: 0x000A,
        };
        assert_eq!(id.base_key(), "XYZ_000A");
    }

    /// Live-sysfs invariants of the reverse lookup / existence helpers.  On a
    /// machine with no DRM connectors or EDIDs (e.g. a headless CI box) the
    /// helpers must degrade gracefully (no panic, None/false).  On real
    /// hardware, resolving a known connector's own EDID base must round-trip
    /// back to a connector name.
    #[test]
    fn live_sysfs_helpers_roundtrip_or_degrade() {
        let _ = connector_exists("DP-1");
        let _ = connector_exists("DP-2");
        let _ = connector_exists("XYZ-9");
        let _ = find_connector_by_base("SAM_707A");
        let _ = find_connector_by_base("ZZZ_0000");
        let _ = find_connector_by_base("");
        if let Some(base) = resolve_base("DP-1", None) {
            let found = find_connector_by_base(&base);
            assert!(found.is_some(), "reverse lookup of own EDID base should find a connector");
        }
    }
}
