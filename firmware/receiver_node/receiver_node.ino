// =====================================================================
// Receiver node (ESP32 #2) for the CAN bus anomaly-detection testbed
// =====================================================================
//
// Part of the project "Implementation and Evaluation of CAN Bus
// Anomaly Detection on ESP32 Hardware". This node is a passive logger: it
// receives every frame on the shared CAN bus through its TWAI controller
// and prints each one as a single CSV row over USB serial. A program on
// the host computer captures this serial stream to a .csv file; that file
// is the dataset used for training and for the attack evaluation.
//
// Output format (one header line, then one row per received frame):
//   timestamp_us,can_id,frame_type,dlc,data_0,data_1,...,data_7
// for example:
//   10024489,0x105,STD,6,FD,FF,FD,FF,E9,03,,
//
// timestamp_us comes from esp_timer_get_time() (microseconds since boot).
// The 10 s startup delay lets the host open the serial port and begin
// recording before the first frame is printed, so recordings start at
// about 10 s.
//
// Bus settings are identical on all three nodes: TWAI TX = GPIO19,
// RX = GPIO18, 500 kbit/s, normal mode (the receiver acknowledges frames
// like any other node).

#include <Arduino.h>
#include "driver/twai.h"
#include "esp_timer.h"

// CAN transceiver (SN65HVD230) connections on the ESP32 receiver:
//   ESP32 GPIO19 (TX) -> transceiver CTX/TXD
//   ESP32 GPIO18 (RX) -> transceiver RTX/RXD
#define CAN_TX_PIN GPIO_NUM_19
#define CAN_RX_PIN GPIO_NUM_18

// Time to allow the PC to open the serial port and start recording.
static const uint32_t SERIAL_RECORDING_STARTUP_DELAY_MS = 10000;

static bool setupCan() {
  twai_general_config_t g =
      TWAI_GENERAL_CONFIG_DEFAULT(CAN_TX_PIN, CAN_RX_PIN, TWAI_MODE_NORMAL);

  // Buffer a short burst of incoming frames while loop() is processing.
  g.rx_queue_len = 64;

  twai_timing_config_t t = TWAI_TIMING_CONFIG_500KBITS();
  twai_filter_config_t f = TWAI_FILTER_CONFIG_ACCEPT_ALL();

  if (twai_driver_install(&g, &t, &f) != ESP_OK) {
    return false;
  }

  if (twai_start() != ESP_OK) {
    twai_driver_uninstall();
    return false;
  }

  return true;
}

static void printCsvHeader() {
  Serial.println(
      "timestamp_us,can_id,frame_type,dlc,"
      "data_0,data_1,data_2,data_3,"
      "data_4,data_5,data_6,data_7");
}

void setup() {
  Serial.begin(115200);

  
  // this delay gives the logger time to attach before any output.
  delay(SERIAL_RECORDING_STARTUP_DELAY_MS);

  if (!setupCan()) {
    
    while (true) {
      delay(1000);
    }
  }

  // This will start the CSV stream with the header, then print received frames.
  printCsvHeader();
}

void loop() {
  twai_message_t m = {};

  // Wait briefly for a frame. Each accepted frame becomes one CSV row.
  if (twai_receive(&m, pdMS_TO_TICKS(100)) != ESP_OK) {
    return;
  }

  
  if (m.rtr) {
    return;
  }

  // This shows the timestamp when this frame is processed by the receiver.
  const int64_t timestampUs = esp_timer_get_time();

  
  const uint8_t dlc = (m.data_length_code <= 8) ? m.data_length_code : 8;

  Serial.printf("%lld,0x%03X,%s,%u",
                (long long)timestampUs,
                (unsigned int)m.identifier,
                m.extd ? "EXT" : "STD",
                (unsigned int)dlc);

  for (uint8_t i = 0; i < 8; ++i) {
    Serial.print(',');

    if (i < dlc) {
      Serial.printf("%02X", m.data[i]);
    }
    
  }

  Serial.println();
}
