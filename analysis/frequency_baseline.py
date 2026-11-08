"""Frequency-based baseline for the attack evaluation recording (Section 5.3.4).

Usage:  python frequency_baseline.py capture.csv

For each 32-frame window, count how often each CAN ID occurred in the 1 s interval
ending at the window's last frame. Flag the window if any ID's count exceeds the
maximum count seen for that ID in the normal traffic before the first attack
(unknown IDs have a maximum of 0). Calibration windows are excluded from evaluation.
Ground truth follows Section 4.3: DoS = IDs outside the 7-ID vocabulary,
spoofing = ID 0x101 with DLC 8; attack frames <= 500 ms apart form one period.
"""
import io, sys
import numpy as np
import pandas as pd

T, RATE_WIN, GAP = 32, 1.0, 0.5
VOCAB = {"0x101", "0x102", "0x103", "0x104", "0x105", "0x106", "0x107"}

def load(path):
    lines = open(path).read().splitlines()
    start = [i for i, l in enumerate(lines) if l.startswith("timestamp_us")][-1]  # data after the last reset
    return pd.read_csv(io.StringIO("\n".join(lines[start:])))

def periods(t, mask):
    ts = t[mask]; out = []; s = p = ts[0]
    for x in ts[1:]:
        if x - p <= GAP: p = x
        else: out.append((s, p)); s = p = x
    out.append((s, p)); return out

def main(path):
    df = load(path)
    t = df.timestamp_us.values / 1e6; ids = df.can_id.values
    P = {"DoS": periods(t, ~df.can_id.isin(VOCAB).values),
         "Spoofing": periods(t, ((df.can_id == "0x101") & (df.dlc == 8)).values)}
    uids = sorted(set(ids)); col = {u: i for i, u in enumerate(uids)}
    X = np.zeros((len(t), len(uids)), int); X[np.arange(len(t)), [col[u] for u in ids]] = 1
    C = np.vstack([np.zeros(len(uids), int), np.cumsum(X, 0)])
    lo = np.searchsorted(t, t - RATE_WIN, side="right")
    cnt = C[np.arange(1, len(t) + 1)] - C[lo]
    ws, we = t[:-T + 1], t[T - 1:]; wc = cnt[np.arange(T - 1, len(t))]
    cal = (we <= P["DoS"][0][0] - 1.0) & (ws >= t[0] + RATE_WIN)
    flag = (wc > wc[cal].max(0)).any(1)
    ov = lambda per: np.any([(we >= s) & (ws <= e) for s, e in per], axis=0)
    att = {k: ov(v) for k, v in P.items()}
    normal = ~att["DoS"] & ~att["Spoofing"] & ~cal
    print(f"calibration windows: {cal.sum()}, evaluated normal windows: {normal.sum()}")
    for k, a in att.items():
        TP = (flag & a).sum(); FN = (~flag & a).sum(); FP = (flag & normal).sum(); TN = (~flag & normal).sum()
        lat = [round(we[np.where(flag & (we >= s) & (ws <= e))[0][0]] - s, 3) for s, e in P[k]]
        print(f"{k:8s} precision {TP/(TP+FP):.3f} recall {TP/(TP+FN):.3f} F1 {2*TP/(2*TP+FP+FN):.3f} "
              f"false-alarm {FP/(FP+TN):.2%} | TP {TP} FN {FN} FP {FP} | latency (s) {lat}")

if __name__ == "__main__":
    main(sys.argv[1])
