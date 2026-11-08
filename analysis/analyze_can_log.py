#!/usr/bin/env python3
"""Summarise a CAN capture CSV: normal baseline, attack window(s), and recovery.

Works on the CSV your receiver/bridge logs (columns: timestamp_us, can_id, ...),
or on --log-frames output from canbert_monitor.py. No live serial copy-pasting needed --
point this at the saved file after a run.

Detects two kinds of attack automatically, no manual timestamp entry required:
  - FLOOD : any CAN ID appearing that isn't in --normal-ids (default 0x101-0x107)
  - SPOOF : a normal ID whose first payload byte equals --marker-byte (default 0xA5,
            matching MARKER_BYTE in can_attacker_node.ino's spoof mode)

For each contiguous attack period (frames of that kind within --gap-ms of each other
are grouped together) it reports: start/end time, duration, attack frame rate, and the
NORMAL-sensor frame rate just before/during/after -- so you can see the drop and the
recovery in one table instead of eyeballing thousands of CSV rows.

Usage
  python analyze_can_log.py capture.csv
  python analyze_can_log.py capture.csv --normal-ids 0x101,0x102,0x103,0x104,0x105,0x106,0x107
  python analyze_can_log.py capture.csv --report report.json   # also saves machine-readable summary
"""
import argparse, csv, json, re, sys
from pathlib import Path

DEFAULT_NORMAL = [0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107]
# The logger always prints the id as 0x%03X (at least 3 hex digits). A row whose id field
# doesn't look like that is corruption (dropped/garbled serial bytes), not a real frame.
ID_RE = re.compile(r"^0x[0-9A-Fa-f]{3,7}$")
# DLC each ID actually uses on the normal sender (can_sensor_sender.ino). The attacker's
# sendSpoof() always sends DLC 8, so a mismatch here is a much more reliable spoof signal
# than the marker byte alone -- a real sensor byte can coincidentally equal the marker.
DEFAULT_DLC = {0x101: 2, 0x102: 4, 0x103: 4, 0x104: 6, 0x105: 6, 0x106: 6, 0x107: 6}


def parse_id(s):
    s = s.strip()
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def load(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        # tolerate either canbert_monitor.py's frame log (frame_idx,ts_us,can_id,label)
        # or a raw receiver CSV (timestamp_us,can_id,frame_type,dlc,data_0..7)
        ts_key = "ts_us" if "ts_us" in r.fieldnames else "timestamp_us"
        id_key = "can_id"
        data0_key = "data_0" if "data_0" in (r.fieldnames or []) else None
        dlc_key = "dlc" if "dlc" in (r.fieldnames or []) else None
        n_corrupt = 0
        for row in r:
            raw_id = row.get(id_key)
            if raw_id is None or not ID_RE.match(raw_id.strip()):
                n_corrupt += 1
                continue    # doesn't look like a real "0xNNN" id -- dropped/garbled serial bytes
            try:
                ts = int(row[ts_key])
                cid = parse_id(raw_id)
            except (KeyError, ValueError, AttributeError, TypeError):
                n_corrupt += 1
                continue
            if ts <= 0:
                n_corrupt += 1
                continue    # a real capture never has timestamp 0; corruption artifact
            d0 = None
            if data0_key and row.get(data0_key):
                try: d0 = int(row[data0_key], 16)
                except ValueError: pass
            dlc = None
            if dlc_key and row.get(dlc_key):
                try: dlc = int(row[dlc_key])
                except ValueError: pass
            rows.append((ts, cid, d0, dlc))
    rows.sort(key=lambda r: r[0])
    if n_corrupt:
        print(f"note: skipped {n_corrupt} corrupted/malformed row(s) in {path} (serial drop-out, not real frames)",
              file=sys.stderr)
    return rows


def group_periods(flag_ts, gap_us):
    """Merge a sorted list of timestamps into (start, end, count) periods."""
    if not flag_ts:
        return []
    periods = []
    s = e = flag_ts[0]
    n = 1
    for t in flag_ts[1:]:
        if t - e <= gap_us:
            e = t; n += 1
        else:
            periods.append((s, e, n))
            s = e = t; n = 1
    periods.append((s, e, n))
    return periods


def rate_in_window(normal_ts, lo, hi):
    n = sum(1 for t in normal_ts if lo <= t < hi)
    dur = max(hi - lo, 1) / 1e6
    return n, n / dur


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    ap.add_argument("--normal-ids", default=",".join(hex(i) for i in DEFAULT_NORMAL),
                    help="comma-separated normal CAN IDs; anything else counts as a flood frame")
    ap.add_argument("--marker-byte", default="0xA5", help="data_0 value that marks a spoofed frame (backup signal)")
    ap.add_argument("--expected-dlc", default=",".join(f"{hex(k)}:{v}" for k, v in DEFAULT_DLC.items()),
                    help="id:dlc pairs the normal sender actually uses; a mismatch is the primary spoof signal "
                         "(set to '' to fall back to the marker byte only)")
    ap.add_argument("--gap-ms", type=float, default=500,
                    help="attack frames within this many ms of each other are one period")
    ap.add_argument("--around-s", type=float, default=5,
                    help="seconds of normal-rate context to show before/after each attack period")
    ap.add_argument("--report", help="optional path to save a JSON summary")
    ap.add_argument("--markdown", help="optional path to save a copy-paste-ready markdown table for the thesis")
    a = ap.parse_args()

    normal_ids = {parse_id(x) for x in a.normal_ids.split(",")}
    marker = parse_id(a.marker_byte)
    expected_dlc = {}
    if a.expected_dlc.strip():
        for pair in a.expected_dlc.split(","):
            k, v = pair.split(":")
            expected_dlc[parse_id(k)] = int(v)
    gap_us = int(a.gap_ms * 1000)

    rows = load(a.csv_path)
    if not rows:
        sys.exit(f"no usable rows found in {a.csv_path}")

    def is_spoof(cid, d0, dlc):
        if cid not in normal_ids:
            return False
        if dlc is not None and cid in expected_dlc:
            return dlc != expected_dlc[cid]          # reliable: attacker always sends DLC 8
        return d0 == marker                           # fallback when DLC isn't in the log

    t0, t1 = rows[0][0], rows[-1][0]
    normal_ts = [t for t, cid, d0, dlc in rows if cid in normal_ids and not is_spoof(cid, d0, dlc)]
    flood_ts = [t for t, cid, d0, dlc in rows if cid not in normal_ids]
    spoof_ts = [t for t, cid, d0, dlc in rows if is_spoof(cid, d0, dlc)]

    base_n, base_rate = rate_in_window(normal_ts, t0, t0 + min(10_000_000, (t1 - t0)))

    print(f"capture: {a.csv_path}")
    print(f"span: {(t1 - t0) / 1e6:.1f} s ({len(rows)} frames total)")
    print(f"baseline normal-traffic rate (first 10s): {base_rate:.1f} frames/s ({base_n} frames)")
    print(f"normal IDs assumed: {', '.join(hex(i) for i in sorted(normal_ids))}")
    print()

    report = {"capture": str(a.csv_path), "span_s": (t1 - t0) / 1e6, "n_frames": len(rows),
              "baseline_rate_fps": base_rate, "flood_periods": [], "spoof_periods": []}

    def describe(kind, periods, attack_ts_set):
        if not periods:
            print(f"-- no {kind} periods detected --\n")
            return
        for i, (s, e, n) in enumerate(periods, 1):
            dur = max(e - s, 1) / 1e6
            attack_rate = n / dur
            before_n, before_rate = rate_in_window(normal_ts, s - int(a.around_s * 1e6), s)
            during_n, during_rate = rate_in_window(normal_ts, s, e)
            after_lo = e
            after_hi = e + int(a.around_s * 1e6)
            after_n, after_rate = rate_in_window(normal_ts, after_lo, after_hi)
            # how long until normal rate is back within 80% of baseline, checked in 1s steps
            recovered_at = None
            t = e
            while t < min(t1, e + 30_000_000):
                n2, r2 = rate_in_window(normal_ts, t, t + 1_000_000)
                if r2 >= 0.8 * base_rate:
                    recovered_at = t
                    break
                t += 1_000_000
            print(f"{kind} period {i}: {s/1e6:.3f}s -> {e/1e6:.3f}s  (duration {dur:.2f}s, {n} attack frames, "
                  f"{attack_rate:.0f} attack frames/s)")
            print(f"  normal-sensor rate   before: {before_rate:.1f}/s   during: {during_rate:.1f}/s   "
                  f"after: {after_rate:.1f}/s   (baseline {base_rate:.1f}/s)")
            if recovered_at is not None:
                print(f"  recovery: normal rate back to >=80% of baseline by t={recovered_at/1e6:.3f}s "
                      f"({(recovered_at - e)/1e6:.2f}s after the attack stopped)")
            else:
                print(f"  recovery: normal rate did NOT return to >=80% of baseline within 30s after the attack")
            print()
            report[f"{kind.lower()}_periods"].append({
                "start_us": s, "end_us": e, "duration_s": dur, "attack_frames": n,
                "attack_rate_fps": attack_rate, "normal_rate_before_fps": before_rate,
                "normal_rate_during_fps": during_rate, "normal_rate_after_fps": after_rate,
                "recovered_at_us": recovered_at,
                "recovery_delay_s": (recovered_at - e) / 1e6 if recovered_at is not None else None,
            })

    flood_periods = group_periods(sorted(flood_ts), gap_us)
    spoof_periods = group_periods(sorted(spoof_ts), gap_us)
    describe("FLOOD", flood_periods, flood_ts)
    describe("SPOOF", spoof_periods, spoof_ts)

    if not flood_periods and not spoof_periods:
        print("No attack frames found -- this looks like a normal-only capture.")

    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=2))
        print(f"summary saved to {a.report}")

    if a.markdown:
        lines = ["| # | Type | Start (s) | End (s) | Duration (s) | Attack frames | Attack rate (fps) | "
                 "Normal rate before (fps) | Normal rate during (fps) | Normal rate after (fps) | "
                 "Recovery delay (s) |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        i = 0
        for kind in ("flood", "spoof"):
            for p in report[f"{kind}_periods"]:
                i += 1
                rec = f"{p['recovery_delay_s']:.2f}" if p["recovery_delay_s"] is not None else "not within 30s"
                lines.append(f"| {i} | {kind.upper()} | {p['start_us']/1e6:.3f} | {p['end_us']/1e6:.3f} | "
                             f"{p['duration_s']:.2f} | {p['attack_frames']} | {p['attack_rate_fps']:.0f} | "
                             f"{p['normal_rate_before_fps']:.1f} | {p['normal_rate_during_fps']:.1f} | "
                             f"{p['normal_rate_after_fps']:.1f} | {rec} |")
        if i == 0:
            lines.append("| - | no attack periods detected | | | | | | | | | |")
        Path(a.markdown).write_text(
            f"Baseline normal-traffic rate: {report['baseline_rate_fps']:.1f} frames/s. "
            f"Capture span: {report['span_s']:.1f} s, {report['n_frames']} frames total.\n\n" + "\n".join(lines) + "\n")
        print(f"markdown table saved to {a.markdown}")


if __name__ == "__main__":
    main()
