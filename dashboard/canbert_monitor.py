#!/usr/bin/env python3
"""CAN-BERT live intrusion-detection dashboard for the ESP32 CAN testbed.

Part of the bachelor thesis "Implementation and Evaluation of CAN Bus Anomaly
Detection on ESP32 Hardware". This program reads the CSV stream produced by the
receiver node (over USB serial, or replayed from a saved .csv file), runs the
trained CAN-BERT model on 32-identifier windows, applies the top-k detection
rule, and serves a live web dashboard that shows NORMAL or ALERT.

Detection core (identical to the training notebook): round(0.45 * 32) = 14
positions per window are masked; for each masked position the model's top-k
identifiers are taken; if the true identifier is not among them the window is
flagged. Identifiers outside the seven known IDs (and extended frames) map to
[UNK], which is never predicted, so a window containing one is always flagged.

Alert logic (the "alert-clearing rule"):
  --confirm      N consecutive flagged windows raise the alert (default 3)
  --clear-after  M consecutive unflagged windows clear it to NORMAL (default 32)

Examples:
  # Live, from the receiver on /dev/ttyUSB0, using the selected k = 1:
  python canbert_monitor.py --checkpoint canbert_T32.pt --k 1 \
      --source serial --port /dev/ttyUSB0

  # Replay a saved recording at real-time speed:
  python canbert_monitor.py --checkpoint canbert_T32.pt --k 1 \
      --source csv --csv ../data/attack_evaluation_recording.csv

The dashboard is served at http://localhost:8000. Optional --log-frames and
--log-windows write the per-frame and per-window records used for analysis.
"""

import argparse
import csv
import json
import math
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------
# CAN-BERT model
# ---------------------------------------------------------------------

class CANBERT(nn.Module):
    def __init__(
        self, vocab_size, n_out, max_len, d_model,
        n_layers, n_heads, d_ff, dropout, pad_id
    ):
        super().__init__()

        self.d = d_model
        self.emb = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_id
        )

        pos = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            n_layers,
            enable_nested_tensor=False,
        )

        self.head = nn.Linear(d_model, n_out)

    def forward(self, x):
        z = self.emb(x) * math.sqrt(self.d)
        z = z + self.pe[:, :x.size(1)]
        return self.head(self.encoder(z))


# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------

class Detector:
    def __init__(
        self, checkpoint, k, passes, stride,
        max_batch, device, seed
    ):
        ck = torch.load(
            checkpoint,
            map_location=device,
            weights_only=False,
        )

        cfg = ck["config"]
        vocab = ck["vocabulary"]
        vocab = vocab.get("token_to_id", vocab)

        self.T = int(ck["sequence_length"])

        self.pad = vocab["[PAD]"]
        self.mask = vocab["[MASK]"]
        self.unk = vocab["[UNK]"]

        special = {"[PAD]", "[MASK]", "[UNK]"}

        observed = sorted(
            token_id
            for token, token_id in vocab.items()
            if token not in special
        )

        self.canid_to_tok = {
            int(token, 16): token_id
            for token, token_id in vocab.items()
            if token not in special
        }

        cls = torch.full(
            (len(vocab),), -100, dtype=torch.long
        )

        cls[torch.tensor(observed)] = torch.arange(
            len(observed)
        )

        self.cls_map = cls.to(device)

        self.model = CANBERT(
            len(vocab),
            len(observed),
            self.T,
            cfg["d_model"],
            cfg["layers"],
            cfg["heads"],
            cfg["ffn"],
            cfg["dropout"],
            self.pad,
        ).to(device)

        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

        self.ratio = float(cfg["mask_ratio"])
        self.k = k
        self.passes = passes
        self.stride = stride
        self.max_batch = max_batch
        self.device = device

        self.gen = torch.Generator(
            device=device
        ).manual_seed(seed)

        self.hist = np.empty(0, dtype=np.int64)
        self.n_seen = 0
        self.skipped = 0

    def tokens(self, can_ids, extended):
        return np.array(
            [
                self.unk if ext
                else self.canid_to_tok.get(can_id, self.unk)
                for can_id, ext in zip(can_ids, extended)
            ],
            dtype=np.int64,
        )

    @torch.inference_mode()
    def _flag(self, windows):
        x = torch.from_numpy(windows).to(self.device)

        n_mask = max(
            1, int(round(x.size(1) * self.ratio))
        )

        true_cls = self.cls_map[x].unsqueeze(-1)

        flagged = torch.zeros(
            len(x), dtype=torch.bool, device=self.device
        )

        for _ in range(self.passes):
            mask_order = torch.rand(
                x.shape,
                device=self.device,
                generator=self.gen,
            ).argsort(1).argsort(1)

            changed = mask_order < n_mask

            predictions = self.model(
                x.masked_fill(changed, self.mask)
            ).topk(self.k, dim=-1).indices

            missed = changed & ~(
                predictions == true_cls
            ).any(-1)

            flagged |= missed.any(1)

        return flagged.cpu().numpy()

    def push(self, tokens):
        arr = np.concatenate([self.hist, tokens])

        first_global = self.n_seen - len(self.hist)

        ends_global = np.arange(
            self.n_seen, self.n_seen + len(tokens)
        )

        ends_global = ends_global[
            (ends_global >= self.T - 1)
            & (ends_global % self.stride == self.stride - 1)
        ]

        if len(ends_global) > self.max_batch:
            self.skipped += len(ends_global) - self.max_batch

            indices = np.linspace(
                0, len(ends_global) - 1, self.max_batch
            ).round().astype(int)

            ends_global = ends_global[indices]

        self.n_seen += len(tokens)

        self.hist = (
            arr[-(self.T - 1):]
            if self.T > 1
            else arr[:0]
        )

        if len(ends_global) == 0:
            return ends_global, np.zeros(0, dtype=bool)

        locations = ends_global - first_global

        windows = np.stack(
            [
                arr[end - self.T + 1:end + 1]
                for end in locations
            ]
        )

        return ends_global, self._flag(windows)


# ---------------------------------------------------------------------
# Read receiver CSV over USB serial
# Expected columns:
# timestamp_us,can_id,frame_type,dlc,data_0,...,data_7
# ---------------------------------------------------------------------

def serial_chunks(port, baud, batch_ms=10):
    import serial

    connection = serial.Serial(
        port,
        baudrate=baud,
        timeout=0.005,
    )

    buffer = bytearray()
    frames = []
    batch_started = time.monotonic()
    batch_interval = max(1, batch_ms) / 1000.0

    try:
        while True:
            chunk = connection.read(
                max(1, connection.in_waiting)
            )

            if chunk:
                buffer.extend(chunk)

            while True:
                newline = buffer.find(b"\n")

                if newline < 0:
                    break

                raw_line = bytes(
                    buffer[:newline]
                ).strip()

                del buffer[:newline + 1]

                if not raw_line:
                    continue

                try:
                    row = next(
                        csv.reader(
                            [
                                raw_line.decode(
                                    "utf-8",
                                    errors="replace",
                                )
                            ]
                        )
                    )

                    if len(row) < 4:
                        continue

                    timestamp_us = int(row[0].strip())

                    identifier = int(
                        row[1].strip(), 16
                    ) & 0x1FFFFFFF

                    frame_type = row[2].strip().upper()
                    dlc = int(row[3].strip())

                    if frame_type not in {"STD", "EXT"}:
                        continue

                    if not 0 <= dlc <= 8:
                        continue

                    is_extended = frame_type == "EXT"

                    frames.append(
                        (
                            timestamp_us,
                            identifier,
                            is_extended,
                        )
                    )

                except (
                    ValueError,
                    csv.Error,
                    UnicodeError,
                ):
                    # Ignore the CSV header and malformed lines.
                    continue

            now = time.monotonic()

            if now - batch_started >= batch_interval:
                batch = frames
                frames = []
                batch_started = now
                yield batch

    finally:
        connection.close()


# ---------------------------------------------------------------------
# Optional CSV replay
# ---------------------------------------------------------------------

def csv_chunks(path, speed, chunk_us=50_000):
    rows = []

    with open(path, newline="") as file:
        for row in csv.reader(file):
            try:
                timestamp = int(row[0])
                identifier = int(row[1], 16)
                rows.append((timestamp, identifier))
            except (ValueError, IndexError):
                continue

    if not rows:
        raise ValueError(
            "No valid timestamp and CAN-ID rows found in CSV."
        )

    frames = [(t, can_id, False) for t, can_id in rows]

    start = time.perf_counter()
    index = 0

    while index < len(frames):
        if speed > 0:
            end_time = frames[index][0] + chunk_us
            end_index = index

            while (
                end_index < len(frames)
                and frames[end_index][0] < end_time
            ):
                end_index += 1

            end_index = max(end_index, index + 1)

            delay = (
                (frames[index][0] - frames[0][0])
                / 1e6 / speed
                - (time.perf_counter() - start)
            )

            if delay > 0:
                time.sleep(delay)
        else:
            end_index = min(len(frames), index + 256)

        yield frames[index:end_index]
        index = end_index


# ---------------------------------------------------------------------
# Dashboard state
# ---------------------------------------------------------------------

class State:
    def __init__(self, window_s=60):
        self.lock = threading.Lock()

        self.status = "starting"
        self.n_frames = 0
        self.n_windows = 0
        self.n_flagged = 0

        self.run_flag = 0
        self.run_ok = 0

        self.rate_history = deque(maxlen=window_s)
        self._last_bucket = None
        self._bucket_count = 0

        self.events = deque(maxlen=50)
        self.start_time = time.time()

    def tick_frame(self):
        now = time.time()
        bucket = int(now)

        if self._last_bucket is None:
            self._last_bucket = bucket

        if bucket != self._last_bucket:
            self.rate_history.append(
                (self._last_bucket, self._bucket_count)
            )

            self._last_bucket = bucket
            self._bucket_count = 0

        self._bucket_count += 1
        self.n_frames += 1

    def snapshot(self):
        with self.lock:
            history = list(self.rate_history)

            if self._last_bucket is not None:
                history.append(
                    (self._last_bucket, self._bucket_count)
                )

            return {
                "status": self.status,
                "n_frames": self.n_frames,
                "n_windows": self.n_windows,
                "n_flagged": self.n_flagged,
                "rate_history": history[-60:],
                "events": list(self.events),
                "uptime_s": time.time() - self.start_time,
            }


# ---------------------------------------------------------------------
# Browser page: HTML, CSS and JavaScript
# The JavaScript belongs inside this string; do not run it separately.
# ---------------------------------------------------------------------

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>CAN-BERT Live Monitor</title>
<style>
body {
  font-family: Arial, Helvetica, sans-serif;
  background: #0b0e14;
  color: #e6e6e6;
  margin: 0;
  padding: 24px;
}
h1 {
  font-size: 20px;
  margin: 0 0 20px;
  color: white;
}
.status {
  font-size: 36px;
  font-weight: 800;
  padding: 24px;
  border-radius: 12px;
  text-align: center;
  margin-bottom: 20px;
}
.status.normal {
  background: #0d3b24;
  color: #4ade80;
}
.status.alert {
  background: #4a0d0d;
  color: #f87171;
}
.status.starting {
  background: #1f2937;
  color: #9ca3af;
}
.grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 20px;
}
.card {
  background: #131722;
  border-radius: 10px;
  padding: 16px;
}
.label {
  font-size: 12px;
  color: #9ca3af;
  text-transform: uppercase;
}
.value {
  font-size: 28px;
  font-weight: 700;
  margin-top: 4px;
}
canvas {
  width: 100%;
  height: 120px;
  background: #131722;
  border-radius: 10px;
}
.section-title {
  font-size: 13px;
  color: #9ca3af;
  text-transform: uppercase;
  margin: 20px 0 8px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
td, th {
  padding: 8px 10px;
  text-align: left;
  border-bottom: 1px solid #1f2937;
}
.ev-alert { color: #f87171; }
.ev-clear { color: #4ade80; }

@media (max-width: 700px) {
  .grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>

<body>
<h1>CAN-BERT Live Intrusion Monitor</h1>

<div id="status" class="status starting">Waiting for data...</div>

<div class="grid">
  <div class="card">
    <div class="label">Frames seen</div>
    <div class="value" id="n_frames">0</div>
  </div>
  <div class="card">
    <div class="label">Windows scored</div>
    <div class="value" id="n_windows">0</div>
  </div>
  <div class="card">
    <div class="label">Flagged windows</div>
    <div class="value" id="n_flagged">0</div>
  </div>
  <div class="card">
    <div class="label">Current frame rate</div>
    <div class="value" id="rate">0/s</div>
  </div>
</div>

<div class="section-title">Frame rate (last 60 seconds)</div>
<canvas id="chart" width="1000" height="120"></canvas>

<div class="section-title">Recent detection events</div>
<table id="events">
  <thead>
    <tr><th>Time</th><th>Event</th></tr>
  </thead>
  <tbody></tbody>
</table>

<script>
async function tick() {
  try {
    const response = await fetch('/state');
    const s = await response.json();

    const status = document.getElementById('status');
    status.className = 'status ' + s.status;

    if (s.status === 'alert') {
      status.textContent = 'ALERT - ATTACK DETECTED';
    } else if (s.status === 'normal') {
      status.textContent = 'NORMAL';
    } else {
      status.textContent = 'WAITING FOR MODEL RESULTS';
    }

    document.getElementById('n_frames').textContent = s.n_frames;
    document.getElementById('n_windows').textContent = s.n_windows;
    document.getElementById('n_flagged').textContent = s.n_flagged;

    const history = s.rate_history || [];
    const currentRate = history.length
      ? history[history.length - 1][1]
      : 0;

    document.getElementById('rate').textContent =
      currentRate + '/s';

    const canvas = document.getElementById('chart');
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const maxRate = Math.max(
      10,
      ...history.map(item => item[1])
    );

    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 2;
    ctx.beginPath();

    history.forEach((item, index) => {
      const x = (
        index / Math.max(history.length - 1, 1)
      ) * canvas.width;

      const y = canvas.height
        - (item[1] / maxRate) * canvas.height * 0.9
        - 5;

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    const tbody = document.querySelector('#events tbody');
    tbody.innerHTML = '';

    (s.events || []).slice().reverse().forEach(event => {
      const tr = document.createElement('tr');

      const timeCell = document.createElement('td');
      timeCell.textContent =
        new Date(event.t * 1000).toLocaleTimeString();

      const eventCell = document.createElement('td');
      eventCell.className = 'ev-' + event.kind;
      eventCell.textContent = event.msg;

      tr.appendChild(timeCell);
      tr.appendChild(eventCell);
      tbody.appendChild(tr);
    });

  } catch (error) {
    // Keep trying while the server or serial stream starts.
  }

  setTimeout(tick, 400);
}

tick();
</script>
</body>
</html>
"""


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == "/state":
                body = json.dumps(state.snapshot()).encode()
                content_type = "application/json"
            else:
                body = PAGE.encode()
                content_type = "text/html; charset=utf-8"

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--confirm", type=int, default=3)
    parser.add_argument("--clear-after", type=int, default=32)

    parser.add_argument(
        "--source",
        choices=["serial", "csv"],
        required=True,
    )

    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--csv")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--log-frames")
    parser.add_argument("--log-windows")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--http-port", type=int, default=8000)

    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")

    if args.confirm < 1 or args.clear_after < 1:
        parser.error("--confirm and --clear-after must be at least 1")

    checkpoint = Path(args.checkpoint)

    if not checkpoint.is_file():
        sys.exit(f"Checkpoint not found: {checkpoint}")

    detector = Detector(
        checkpoint,
        args.k,
        args.passes,
        args.stride,
        args.max_batch,
        args.device,
        args.seed,
    )

    state = State()

    server = ThreadingHTTPServer(
        ("127.0.0.1", args.http_port),
        make_handler(state),
    )

    threading.Thread(
        target=server.serve_forever,
        daemon=True,
    ).start()

    print(
        f"Dashboard: http://localhost:{args.http_port}"
    )
    print(
        f"Model sequence length: {detector.T}; k={args.k}"
    )

    if args.source == "serial":
        if not args.port:
            sys.exit("--port is required for serial input")

        chunks = serial_chunks(args.port, args.baud)
        print(
            f"Reading CSV serial data from {args.port} "
            f"at {args.baud} baud..."
        )

    else:
        if not args.csv:
            sys.exit("--csv is required for CSV replay")

        chunks = csv_chunks(args.csv, args.speed)
        print(f"Replaying {args.csv}...")

    frames_file = (
        open(args.log_frames, "w", newline="")
        if args.log_frames else None
    )

    windows_file = (
        open(args.log_windows, "w", newline="")
        if args.log_windows else None
    )

    frames_writer = (
        csv.writer(frames_file) if frames_file else None
    )

    windows_writer = (
        csv.writer(windows_file) if windows_file else None
    )

    if frames_writer:
        frames_writer.writerow(
            ["frame_idx", "ts_us", "can_id"]
        )

    if windows_writer:
        windows_writer.writerow(
            ["frame_idx", "flagged"]
        )

    in_alert = False
    scored_any = False

    try:
        for frames in chunks:
            if not frames:
                continue

            can_ids = [frame[1] for frame in frames]
            extended = [frame[2] for frame in frames]

            base = state.n_frames

            if frames_writer:
                frames_writer.writerows(
                    [
                        [
                            base + index,
                            int(frame[0]),
                            f"0x{frame[1]:X}",
                        ]
                        for index, frame in enumerate(frames)
                    ]
                )

            for _ in frames:
                state.tick_frame()

            end_indices, flags = detector.push(
                detector.tokens(can_ids, extended)
            )

            for end_index, flagged in zip(end_indices, flags):
                scored_any = True
                state.n_windows += 1

                if flagged:
                    state.n_flagged += 1
                    state.run_flag += 1
                    state.run_ok = 0
                else:
                    state.run_ok += 1
                    state.run_flag = 0

                if windows_writer:
                    windows_writer.writerow(
                        [int(end_index), int(flagged)]
                    )

                if not in_alert and state.run_flag >= args.confirm:
                    in_alert = True
                    state.status = "alert"

                    state.events.append({
                        "t": time.time(),
                        "kind": "alert",
                        "msg": (
                            "Alert raised at frame "
                            f"{int(end_index)}"
                        ),
                    })

                elif in_alert and state.run_ok >= args.clear_after:
                    in_alert = False
                    state.status = "normal"

                    state.events.append({
                        "t": time.time(),
                        "kind": "clear",
                        "msg": (
                            "Alert cleared at frame "
                            f"{int(end_index)}"
                        ),
                    })

                elif not in_alert and scored_any:
                    state.status = "normal"

    except KeyboardInterrupt:
        print("\nStopping dashboard...")

    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise

    finally:
        for file in (frames_file, windows_file):
            if file:
                file.close()

        server.shutdown()
        server.server_close()
        print("Stopped.")


if __name__ == "__main__":
    main()