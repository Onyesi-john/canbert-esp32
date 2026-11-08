# CAN Bus Anomaly Detection on ESP32 Hardware

Code, firmware, data, and the trained model for the bachelor thesis
**"Implementation and Evaluation of CAN Bus Anomaly Detection on ESP32 Hardware:
Real-Time Detection of DoS and Spoofing Attacks Using CAN-BERT."**

A three-node ESP32 test bench produces normal CAN traffic, records it, and
injects two kinds of attack (DoS flooding and identifier spoofing). A
transformer masked-language model (CAN-BERT, adapted from Alkhatib et al.) is
trained on the normal identifier sequences and then used to flag windows of
traffic that do not match the learned pattern. This repository lets a reader
reproduce the whole pipeline: flash the three nodes, record data, rebuild the
model from the Colab notebook, and run the analysis, evaluation, and live
dashboard.

> This is a research test bench on an **isolated** three-node CAN bus. The
> attacker firmware is here only to reproduce the thesis experiments and is not
> intended for use on any real or shared vehicle network.

---

## Repository layout

```
.
├── firmware/                     Arduino/ESP-IDF sketches for the three nodes
│   ├── sender_node/              normal sensor traffic (7 CAN IDs)
│   ├── receiver_node/            passive logger -> CSV over USB serial
│   └── attacker_node/            DoS flooding and spoofing (push-button)
├── training/
│   └── canbert_colab_training.ipynb   trains the model and writes canbert_T32.pt
├── model/
│   └── canbert_T32.pt            trained checkpoint used in the thesis
├── data/
│   ├── normal_training_dataset.csv     87,848 normal frames (training/validation)
│   ├── attack_evaluation_recording.csv 38,759 frames, normal + attacks
│   └── raw/                      the untouched serial captures 
├── analysis/
│   ├── analyze_can_log.py        traffic-impact tables 
│   └── frequency_baseline.py     rule-based frequency baseline 
├── evaluation/
│   └── evaluate_detection.py     detection performance + latency 
├── dashboard/
│   └── canbert_monitor.py        live detection dashboard 
├── requirements.txt
└── README.md
```

---

## Hardware test bench

Three ESP32 boards share one CAN bus. Each board drives an **SN65HVD230**
transceiver; the bus runs classic CAN at **500 kbit/s**. The TWAI (CAN)
controller uses the same pins on all three nodes:

| Signal | ESP32 pin |
|--------|-----------|
| CAN TX | GPIO19    |
| CAN RX | GPIO18    |

- **Sender node** reads five sensor modules and transmits seven message types,
  one per standard 11-bit identifier:

  | ID    | Source            | DLC | Period  |
  |-------|-------------------|-----|---------|
  | 0x101 | LDR (light)       | 2   | 500 ms  |
  | 0x102 | DHT11 (temp/hum)  | 4   | 2000 ms |
  | 0x103 | Ultrasonic        | 4   | 500 ms  |
  | 0x104 | Water level       | 6   | 1000 ms |
  | 0x105 | Accelerometer     | 6   | 200 ms  |
  | 0x106 | Gyroscope         | 6   | 200 ms  |
  | 0x107 | MPU6050 angles    | 6   | 200 ms  |

- **Receiver node** is a passive logger. It prints every received frame as one
  CSV row over USB serial; a program on the host records that stream to a file.
  A 10 s startup delay lets the host open the serial port first, so recordings
  begin at about 10 s.

- **Attacker node** generates the attack traffic, toggled by the on-board BOOT
  button. Two modes (set `ATTACK_MODE` in the sketch):
  - **Flood (DoS):** transmits the unused identifier **0x000** as fast as the
    bus allows. 0x000 is outside the seven normal IDs, so the detector maps it
    to `[UNK]`.
  - **Spoof:** transmits the legitimate identifier **0x101** far faster than the
    sender (every 20 ms vs 500 ms), with the first payload byte set to a marker
    (0xA5) and DLC 8 (genuine 0x101 uses DLC 2) so the recording can be labelled.

### Flashing
Open each sketch in the Arduino IDE (ESP32 core installed) or `arduino-cli`,
select your ESP32 board, and upload. The sender node also needs the `DHT` and
`MPU6050_light` libraries.

---

## The model

`model/canbert_T32.pt` is the trained checkpoint used in the thesis. It is a
BERT-style transformer encoder trained only on **sequences of CAN identifiers**
(not payloads or DLC) from the normal data, using masked-identifier prediction.

| Setting | Value |
|---------|-------|
| Window length T | 32 identifiers (stride 1) |
| Encoder layers / heads | 4 / 1 |
| Model / FFN width | 256 / 512 |
| Dropout | 0.1 |
| Mask ratio | 0.45 (14 of 32 positions) |
| Vocabulary | 7 known IDs (0x101–0x107) + `[PAD]`, `[MASK]`, `[UNK]` |
| Parameters | 2,120,967 |
| Best epoch / val MLM loss | 81 / 5.07 × 10⁻⁵ |
| Seed | 42 |

### Detection rule
For each 32-identifier window the model re-predicts a random subset of masked
positions. A window is **flagged** if any masked known-ID position is not in the
model's **top-k** predictions. The DoS identifier 0x000 is unknown, becomes
`[UNK]`, and is never a valid prediction, so a window containing it is flagged as
soon as that position is masked — this is the simple unknown-ID case, distinct
from the sequence-based detection that catches spoofing of a *known* ID. `k = 1`
was chosen as the smallest `k` whose false-alarm rate on normal validation
windows met the 0.1 % target.


---

## Running the analysis

Install dependencies (a virtual environment is recommended):

```bash
pip install -r requirements.txt
```

### Traffic impact — Tables 5.1 and 5.2
```bash
python analysis/analyze_can_log.py data/attack_evaluation_recording.csv
```
Reports the attack periods and their effect on bus load and per-ID timing.

### Detection performance and latency 
```bash
python evaluation/evaluate_detection.py \
    --checkpoint model/canbert_T32.pt \
    --csv data/attack_evaluation_recording.csv
```
Builds every 32-identifier window of the recording, runs the top-k rule for
`k = 1, 2, 3, 5`, labels windows against the ground-truth attack periods
(DoS = IDs outside 0x101–0x107; spoofing = 0x101 with the wrong DLC), and prints
precision, recall, F1, false-alarm rate, and per-attack latency.

### Frequency baseline — Section 5.3.4
```bash
python analysis/frequency_baseline.py data/attack_evaluation_recording.csv
```
A simple rule that flags a window when any identifier arrives more often than it
ever did in the calibration (pre-attack) traffic — a point of comparison for the
learned detector.

### Live dashboard
```bash
python dashboard/canbert_monitor.py --help
```
Runs the detector on a live serial feed or a replayed CSV and shows alerts in a
browser. An alert is raised after 3 consecutive flagged windows and cleared
after 32 consecutive clean ones.

---



---

## Requirements

- Python 3.10+ with the packages in `requirements.txt` (PyTorch, NumPy, pandas).
- For the firmware: Arduino IDE or `arduino-cli` with the ESP32 core, plus the
  `DHT` and `MPU6050_light` libraries for the sender node.

## Reference

CAN-BERT is adapted from:
N. Alkhatib et al., *"CAN-BERT do it? Controller Area Network Intrusion
Detection System based on BERT."*
