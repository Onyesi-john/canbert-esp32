#!/usr/bin/env python3
"""Evaluate the trained CAN-BERT model on the attack evaluation recording.

Reproduces the detection-performance and latency results of the thesis
(Chapter 5): for each candidate k it builds every 32-identifier window of the
recording, runs the model with the top-k rule, labels the windows against the
ground-truth attack periods, and reports precision, recall, F1, the false-alarm
rate, and the per-attack detection latency.

Ground truth follows Section 4.3:
  - DoS   : frames whose identifier is outside the seven normal IDs (0x000).
  - Spoof : identifier 0x101 with a DLC different from the genuine DLC of 2.
  - Frames of one attack type are excluded when evaluating the other type;
    the same normal windows are used for both.

Usage:
  python evaluate_detection.py --checkpoint ../model/canbert_T32.pt \
      --csv ../data/attack_evaluation_recording.csv
"""
import argparse
import csv
import math

import numpy as np
import torch
from torch import nn

NORMAL_IDS = [0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107]
GENUINE_DLC = {0x101: 2, 0x102: 4, 0x103: 4, 0x104: 6, 0x105: 6, 0x106: 6, 0x107: 6}
GAP_US = 500_000  # frames within 500 ms are one attack period


class CANBERT(nn.Module):
    def __init__(self, vocab_size, n_out, max_len, d_model, n_layers, n_heads, d_ff, dropout, pad_id):
        super().__init__()
        self.d = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           activation="relu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, x):
        z = self.emb(x) * math.sqrt(self.d) + self.pe[:, : x.size(1)]
        return self.head(self.encoder(z))


def load_rows(path):
    ts, ids, dlc = [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ts.append(int(row["timestamp_us"]))
                ids.append(int(row["can_id"], 16))
                dlc.append(int(row["dlc"]))
            except (ValueError, KeyError, TypeError):
                continue
    order = np.argsort(ts, kind="stable")
    return np.array(ts)[order], np.array(ids)[order], np.array(dlc)[order]


def periods(mask, ts):
    t = ts[mask]
    if len(t) == 0:
        return []
    out, s, e = [], t[0], t[0]
    for x in t[1:]:
        if x - e <= GAP_US:
            e = x
        else:
            out.append((s, e)); s = e = x
    out.append((s, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ks", default="1,2,3,5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    ks = [int(x) for x in a.ks.split(",")]

    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    cfg = ck["config"]
    vocab = ck["vocabulary"]; vocab = vocab.get("token_to_id", vocab)
    T = int(ck["sequence_length"])
    pad, mask_tok, unk = vocab["[PAD]"], vocab["[MASK]"], vocab["[UNK]"]
    special = {"[PAD]", "[MASK]", "[UNK]"}
    observed = sorted(tid for tok, tid in vocab.items() if tok not in special)
    canid_to_tok = {int(tok, 16): tid for tok, tid in vocab.items() if tok not in special}
    cls_map = torch.full((len(vocab),), -100, dtype=torch.long)
    cls_map[torch.tensor(observed)] = torch.arange(len(observed))
    cls_map = cls_map.to(dev)

    model = CANBERT(len(vocab), len(observed), T, cfg["d_model"], cfg["layers"],
                    cfg["heads"], cfg["ffn"], cfg["dropout"], pad).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ratio = float(cfg["mask_ratio"])

    ts, ids, dlc = load_rows(a.csv)
    n = len(ts)
    print(f"frames: {n}, span {ts[0]/1e6:.3f}-{ts[-1]/1e6:.3f} s")

    # Tokenise: unknown IDs -> [UNK]
    tokens = np.array([canid_to_tok.get(int(c), unk) for c in ids], dtype=np.int64)

    # Ground-truth periods
    dos_mask = ~np.isin(ids, NORMAL_IDS)
    spoof_mask = (ids == 0x101) & (dlc != GENUINE_DLC[0x101])
    dos_p = periods(dos_mask, ts)
    spoof_p = periods(spoof_mask, ts)
    print(f"DoS periods: {len(dos_p)}, spoofing periods: {len(spoof_p)}")

    # Windows (stride 1)
    nw = n - T + 1
    X = np.stack([tokens[i:i + T] for i in range(nw)])
    w_start = ts[:nw]
    w_end = ts[T - 1:]

    def overlaps(per):
        o = np.zeros(nw, bool)
        for s, e in per:
            o |= (w_end >= s) & (w_start <= e)
        return o

    dos_win = overlaps(dos_p)
    spoof_win = overlaps(spoof_p)
    normal_win = ~dos_win & ~spoof_win
    print(f"windows: {nw}  | DoS {int(dos_win.sum())}  "
          f"spoof {int(spoof_win.sum())}  normal {int(normal_win.sum())}")

    # Run the detector once, collecting flags for every k
    Xt = torch.from_numpy(X).to(dev)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    n_mask = max(1, int(round(T * ratio)))
    flags = {k: np.zeros(nw, bool) for k in ks}
    with torch.inference_mode():
        for i in range(0, nw, 1024):
            xb = Xt[i:i + 1024]
            r = torch.rand(xb.shape, device=dev, generator=gen).argsort(1).argsort(1)
            ch = r < n_mask
            top = model(xb.masked_fill(ch, mask_tok)).topk(max(ks), dim=-1).indices
            true_cls = cls_map[xb].unsqueeze(-1)
            for k in ks:
                hit = (top[..., :k] == true_cls).any(-1)
                flags[k][i:i + len(xb)] = (ch & ~hit).any(1).cpu().numpy()

    def metrics(flag, attack_win):
        tp = int((flag & attack_win).sum()); fn = int((~flag & attack_win).sum())
        fp = int((flag & normal_win).sum()); tn = int((~flag & normal_win).sum())
        pr = tp / (tp + fp) if tp + fp else float("nan")
        rc = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else float("nan")
        fa = fp / (fp + tn) if fp + tn else float("nan")
        return pr, rc, f1, fa

    print("\n=== Table 5.5: detection performance ===")
    print(f"{'attack':9} {'k':>2} {'prec':>7} {'recall':>7} {'F1':>7} {'FA':>7}")
    for name, aw in (("DoS", dos_win), ("Spoofing", spoof_win)):
        for k in ks:
            pr, rc, f1, fa = metrics(flags[k], aw)
            print(f"{name:9} {k:>2} {pr:7.3f} {rc:7.3f} {f1:7.3f} {fa:7.2%}")

    print("\n=== Table 5.6: detection latency (s) ===")
    for name, per in (("DoS", dos_p), ("Spoofing", spoof_p)):
        for s, e in per:
            row = [f"{name} {s/1e6:8.2f}"]
            for k in ks:
                idx = np.where(flags[k] & (w_end >= s) & (w_start <= e))[0]
                row.append(f"{(w_end[idx[0]] - s)/1e6:.3f}" if len(idx) else "  -  ")
            print("  ".join(row))


if __name__ == "__main__":
    main()
