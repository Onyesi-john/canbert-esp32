// =====================================================================
// Sender node (ESP32 #1) for the CAN bus anomaly-detection testbed
// =====================================================================
//
// Part of the bachelor thesis "Implementation and Evaluation of CAN Bus
// Anomaly Detection on ESP32 Hardware". This node reads five sensor
// modules and transmits the normal sensor traffic on the shared CAN bus,
// using seven standard 11-bit identifiers, one per message type:
//
//   0x101  LDR               raw ADC                         DLC 2, 500 ms
//   0x102  DHT11             temperature x100, humidity x100  DLC 4, 2000 ms
//   0x103  Ultrasonic        distance x10 cm, validity flag   DLC 4, 500 ms
//   0x104  Water level       raw ADC, mV, percentage          DLC 6, 1000 ms
//   0x105  Accelerometer     ax, ay, az (milli-g)             DLC 6, 200 ms
//   0x106  Gyroscope         gx, gy, gz (0.1 deg/s)           DLC 6, 200 ms
//   0x107  MPU6050 angles    x, y, z (0.1 deg)                DLC 6, 200 ms
//
// Each sensor runs in its own FreeRTOS task; a CAN task transmits the
// frames at the periods above. Bus settings are identical on all three
// nodes: TWAI TX = GPIO19, RX = GPIO18, 500 kbit/s, normal mode. The
// receiver node logs the resulting traffic (see receiver_node.ino).
//
// The serial logger (serialTask) prints a separate human-readable line to
// this node's own USB port for local monitoring only; it is not the CAN
// dataset. The dataset is the CSV produced by the receiver node.

#include <Arduino.h>
#include <Wire.h>
#include <DHT.h>
#include <MPU6050_light.h>
#include <math.h>
#include "driver/twai.h"

// =====================================================
// 1. PIN CONFIGURATION
// =====================================================

#define LDR_PIN             35

#define MPU_SDA_PIN         21
#define MPU_SCL_PIN         22

#define TRIG_PIN            25
#define ECHO_PIN            26   // HC-SR04 echo is 5 V: use a divider (e.g. 1k + 2k) to 3.3 V
#define CLOSE_DISTANCE      20.0f

#define DHT_PIN             4
#define DHT_TYPE            DHT11

#define WATER_SENSOR_PIN    34

// SN65HVD230 CAN transceiver. NOT 21/22: those belong to the MPU6050.
#define CAN_TX_PIN          GPIO_NUM_19   // -> transceiver CTX
#define CAN_RX_PIN          GPIO_NUM_18   // -> transceiver RTX

// =====================================================
// 2. SETTINGS
// =====================================================

const int LIGHT_THRESHOLD = 1000;

const int DRY_VALUE = 0;
const int WET_VALUE = 2000;

const uint32_t LDR_INTERVAL_MS        = 500;
const uint32_t MPU_INTERVAL_MS        = 200;
const uint32_t ULTRASONIC_INTERVAL_MS = 500;
const uint32_t DHT_INTERVAL_MS        = 2000;
const uint32_t WATER_INTERVAL_MS      = 1000;
const uint32_t LOG_INTERVAL_MS          = 1000;
const uint32_t CAN_SCHEDULER_TICK_MS    = 10;   // scheduler resolution; per-sensor TX periods below

// CAN frame IDs (standard 11-bit). Each logical sensor/message has its own ID.
// Update the receiver/decoder to use this same table.
#define CAN_ID_LDR         0x101  // LDR raw ADC
#define CAN_ID_DHT11       0x102  // temperature x100, humidity x100
#define CAN_ID_ULTRASONIC  0x103  // distance x10 cm, validity flag
#define CAN_ID_WATER       0x104  // raw ADC, voltage mV, percentage
#define CAN_ID_ACCEL       0x105  // ax, ay, az (milli-g)
#define CAN_ID_GYRO        0x106  // gx, gy, gz (0.1 deg/s)
#define CAN_ID_ANGLE       0x107  // x, y, z (0.1 deg)

// =====================================================
// 3. SENSOR OBJECTS
// =====================================================

DHT dht(DHT_PIN, DHT_TYPE);
MPU6050 mpu(Wire);

// =====================================================
// 4. SHARED SENSOR DATA
// =====================================================

struct SensorData {
  int ldrRaw = 0;

  float accX = NAN, accY = NAN, accZ = NAN;
  float gyroX = NAN, gyroY = NAN, gyroZ = NAN;
  float angleX = NAN, angleY = NAN, angleZ = NAN;

  float distanceCm = NAN;
  bool distanceValid = false;

  float temperatureC = NAN;
  float humidityPct = NAN;

  int waterRaw = 0;
  float waterVoltage = 0.0f;
  int waterPercent = 0;
};

SensorData readings;
SemaphoreHandle_t dataMutex = nullptr;

bool mpuReady = false;
bool canReady = false;

// =====================================================
// 5. LDR TASK
// =====================================================

void ldrTask(void *parameter) {
  TickType_t lastWake = xTaskGetTickCount();

  for (;;) {
    int value = analogRead(LDR_PIN);

    xSemaphoreTake(dataMutex, portMAX_DELAY);
    readings.ldrRaw = value;
    xSemaphoreGive(dataMutex);

    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(LDR_INTERVAL_MS));
  }
}

// =====================================================
// 6. MPU6050 TASK
// =====================================================

void mpuTask(void *parameter) {
  TickType_t lastWake = xTaskGetTickCount();

  for (;;) {
    if (mpuReady) {
      mpu.update();

      float ax = mpu.getAccX();
      float ay = mpu.getAccY();
      float az = mpu.getAccZ();

      float gx = mpu.getGyroX();
      float gy = mpu.getGyroY();
      float gz = mpu.getGyroZ();

      float rx = mpu.getAngleX();
      float ry = mpu.getAngleY();
      float rz = mpu.getAngleZ();

      xSemaphoreTake(dataMutex, portMAX_DELAY);

      readings.accX = ax;  readings.accY = ay;  readings.accZ = az;
      readings.gyroX = gx; readings.gyroY = gy; readings.gyroZ = gz;
      readings.angleX = rx; readings.angleY = ry; readings.angleZ = rz;

      xSemaphoreGive(dataMutex);
    }

    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(MPU_INTERVAL_MS));
  }
}

// =====================================================
// 7. ULTRASONIC TASK
// =====================================================

void ultrasonicTask(void *parameter) {
  TickType_t lastWake = xTaskGetTickCount();

  for (;;) {
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);

    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    unsigned long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);

    bool valid = duration > 0;
    float distance = NAN;

    if (valid) {
      distance = duration * 0.0343f / 2.0f;
    }

    xSemaphoreTake(dataMutex, portMAX_DELAY);
    readings.distanceValid = valid;
    readings.distanceCm = distance;
    xSemaphoreGive(dataMutex);

    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(ULTRASONIC_INTERVAL_MS));
  }
}

// =====================================================
// 8. DHT11 TASK
// =====================================================

void dhtTask(void *parameter) {
  TickType_t lastWake = xTaskGetTickCount();

  for (;;) {
    float humidity = dht.readHumidity();
    float temperature = dht.readTemperature();

    if (!isnan(humidity) && !isnan(temperature)) {
      xSemaphoreTake(dataMutex, portMAX_DELAY);
      readings.humidityPct = humidity;
      readings.temperatureC = temperature;
      xSemaphoreGive(dataMutex);
    }

    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(DHT_INTERVAL_MS));
  }
}

// =====================================================
// 9. WATER-LEVEL TASK
// =====================================================

void waterTask(void *parameter) {
  TickType_t lastWake = xTaskGetTickCount();

  for (;;) {
    int rawValue = analogRead(WATER_SENSOR_PIN);
    float voltage = rawValue * 3.3f / 4095.0f;

    int percentage = map(rawValue, DRY_VALUE, WET_VALUE, 0, 100);
    percentage = constrain(percentage, 0, 100);

    xSemaphoreTake(dataMutex, portMAX_DELAY);
    readings.waterRaw = rawValue;
    readings.waterVoltage = voltage;
    readings.waterPercent = percentage;
    xSemaphoreGive(dataMutex);

    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(WATER_INTERVAL_MS));
  }
}

// =====================================================
// 10. CAN (TWAI) HELPERS AND TASK
// =====================================================

// Scale a float to int16. NaN (no data) becomes INT16_MIN as a sentinel.
static int16_t scale16(float v, float k) {
  if (isnan(v)) return INT16_MIN;
  long x = lroundf(v * k);
  if (x > 32767) x = 32767;
  if (x < -32767) x = -32767;
  return (int16_t)x;
}

// Send up to 4 little-endian int16 values in one standard CAN frame (max 8 bytes).
static void sendInts(uint32_t id, const int16_t *v, uint8_t n) {
  twai_message_t m = {};
  m.identifier = id;
  m.data_length_code = n * 2;
  for (uint8_t i = 0; i < n; i++) {
    m.data[2 * i]     = (uint8_t)(v[i] & 0xFF);
    m.data[2 * i + 1] = (uint8_t)(((uint16_t)v[i]) >> 8);
  }
  twai_transmit(&m, pdMS_TO_TICKS(20));
}

static bool setupCan() {
  twai_general_config_t g =
    TWAI_GENERAL_CONFIG_DEFAULT(CAN_TX_PIN, CAN_RX_PIN, TWAI_MODE_NORMAL);
  g.tx_queue_len = 10;

  twai_timing_config_t t = TWAI_TIMING_CONFIG_500KBITS();
  twai_filter_config_t f = TWAI_FILTER_CONFIG_ACCEPT_ALL();

  if (twai_driver_install(&g, &t, &f) != ESP_OK) return false;
  if (twai_start() != ESP_OK) return false;
  return true;
}

// Recover if the bus went off (e.g. no receiver connected yet).
static void canRecover() {
  twai_status_info_t st;
  if (twai_get_status_info(&st) != ESP_OK) return;

  if (st.state == TWAI_STATE_BUS_OFF) {
    twai_initiate_recovery();
  } else if (st.state == TWAI_STATE_STOPPED) {
    twai_start();
  }
}

void canTask(void *parameter) {
  TickType_t lastWake = xTaskGetTickCount();
  SensorData s;

  // Independent transmit timers avoid the uint8_t counter-rollover issue.
  // Unsigned subtraction remains safe when millis() wraps around.
  uint32_t lastMotionTx = millis();
  uint32_t lastLdrTx = lastMotionTx;
  uint32_t lastDhtTx = lastMotionTx;
  uint32_t lastUltrasonicTx = lastMotionTx;
  uint32_t lastWaterTx = lastMotionTx;

  for (;;) {
    const uint32_t now = millis();

    xSemaphoreTake(dataMutex, portMAX_DELAY);
    s = readings;
    xSemaphoreGive(dataMutex);

    canRecover();

    // Accelerometer, gyroscope and angles: 200 ms each, each on its own ID.
    if ((uint32_t)(now - lastMotionTx) >= MPU_INTERVAL_MS) {
      lastMotionTx = now;

      int16_t acc[3] = {
        scale16(s.accX, 1000),
        scale16(s.accY, 1000),
        scale16(s.accZ, 1000)
      };
      int16_t gyro[3] = {
        scale16(s.gyroX, 10),
        scale16(s.gyroY, 10),
        scale16(s.gyroZ, 10)
      };
      int16_t angle[3] = {
        scale16(s.angleX, 10),
        scale16(s.angleY, 10),
        scale16(s.angleZ, 10)
      };

      sendInts(CAN_ID_ACCEL, acc, 3);
      sendInts(CAN_ID_GYRO, gyro, 3);
      sendInts(CAN_ID_ANGLE, angle, 3);
    }

    // LDR: raw 12-bit ADC value, approximately every 500 ms (2 Hz).
    if ((uint32_t)(now - lastLdrTx) >= LDR_INTERVAL_MS) {
      lastLdrTx = now;
      int16_t ldr[1] = { (int16_t)s.ldrRaw };
      sendInts(CAN_ID_LDR, ldr, 1);
    }

    // DHT11: temperature and humidity, approximately every 2 seconds.
    // Each value is scaled by 100; INT16_MIN (bytes 00 80) means unavailable.
    if ((uint32_t)(now - lastDhtTx) >= DHT_INTERVAL_MS) {
      lastDhtTx = now;
      int16_t dhtValues[2] = {
        scale16(s.temperatureC, 100),
        scale16(s.humidityPct, 100)
      };
      sendInts(CAN_ID_DHT11, dhtValues, 2);
    }

    // Ultrasonic sensor: distance x10 cm and a validity flag (0 or 1).
    if ((uint32_t)(now - lastUltrasonicTx) >= ULTRASONIC_INTERVAL_MS) {
      lastUltrasonicTx = now;
      int16_t ultrasonic[2] = {
        s.distanceValid ? scale16(s.distanceCm, 10) : INT16_MIN,
        s.distanceValid ? 1 : 0
      };
      sendInts(CAN_ID_ULTRASONIC, ultrasonic, 2);
    }

    // Water-level sensor: raw ADC, millivolts, and percentage once per second.
    if ((uint32_t)(now - lastWaterTx) >= WATER_INTERVAL_MS) {
      lastWaterTx = now;
      int16_t water[3] = {
        (int16_t)s.waterRaw,
        (int16_t)lroundf(s.waterVoltage * 1000.0f),
        (int16_t)s.waterPercent
      };
      sendInts(CAN_ID_WATER, water, 3);
    }

    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(CAN_SCHEDULER_TICK_MS));
  }
}

// =====================================================
// 11. SERIAL LOGGER
// =====================================================

void serialTask(void *parameter) {
  TickType_t lastWake = xTaskGetTickCount();
  SensorData snapshot;

  Serial.println(
    "TIME_MS,"
    "LDR_RAW,LDR_STATE,"
    "ACC_X_G,ACC_Y_G,ACC_Z_G,"
    "GYRO_X_DPS,GYRO_Y_DPS,GYRO_Z_DPS,"
    "ANGLE_X_DEG,ANGLE_Y_DEG,ANGLE_Z_DEG,"
    "DISTANCE_CM,DISTANCE_STATE,"
    "TEMP_C,HUMIDITY_PCT,"
    "WATER_RAW,WATER_V,WATER_PCT"
  );

  for (;;) {
    xSemaphoreTake(dataMutex, portMAX_DELAY);
    snapshot = readings;
    xSemaphoreGive(dataMutex);

    Serial.print(millis());
    Serial.print(',');

    Serial.print(snapshot.ldrRaw);
    Serial.print(',');
    Serial.print(snapshot.ldrRaw > LIGHT_THRESHOLD ? "BRIGHT" : "DARK");
    Serial.print(',');

    Serial.printf("%.3f,%.3f,%.3f,", snapshot.accX, snapshot.accY, snapshot.accZ);
    Serial.printf("%.3f,%.3f,%.3f,", snapshot.gyroX, snapshot.gyroY, snapshot.gyroZ);
    Serial.printf("%.2f,%.2f,%.2f,", snapshot.angleX, snapshot.angleY, snapshot.angleZ);

    if (snapshot.distanceValid) {
      Serial.print(snapshot.distanceCm, 2);
      Serial.print(',');
      Serial.print(snapshot.distanceCm <= CLOSE_DISTANCE ? "CLOSE" : "FAR");
    } else {
      Serial.print("nan,NO_ECHO");
    }
    Serial.print(',');

    Serial.print(snapshot.temperatureC, 2);
    Serial.print(',');
    Serial.print(snapshot.humidityPct, 2);
    Serial.print(',');

    Serial.print(snapshot.waterRaw);
    Serial.print(',');
    Serial.print(snapshot.waterVoltage, 2);
    Serial.print(',');
    Serial.println(snapshot.waterPercent);

    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(LOG_INTERVAL_MS));
  }
}

// =====================================================
// 12. SETUP
// =====================================================

void setup() {
  Serial.begin(115200);

  analogReadResolution(12);
  pinMode(LDR_PIN, INPUT);
  pinMode(WATER_SENSOR_PIN, INPUT);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  dht.begin();

  Wire.begin(MPU_SDA_PIN, MPU_SCL_PIN);

  byte status = mpu.begin();

  Serial.print("MPU6050 status: ");
  Serial.println(status);

  if (status == 0) {
    Serial.println("Keep MPU6050 still during calibration...");
    delay(1000);
    mpu.calcOffsets();
    mpuReady = true;
    Serial.println("MPU6050 calibration complete.");
  } else {
    Serial.println("WARNING: MPU6050 initialization failed.");
  }

  canReady = setupCan();
  Serial.println(canReady ? "CAN started (500 kbps)." : "ERROR: CAN start failed.");

  dataMutex = xSemaphoreCreateMutex();

  if (dataMutex == nullptr) {
    Serial.println("ERROR: Mutex creation failed.");
    while (true) {
      delay(1000);
    }
  }

  xTaskCreate(ldrTask, "LDR", 2048, nullptr, 2, nullptr);
  xTaskCreate(mpuTask, "MPU6050", 4096, nullptr, 3, nullptr);
  xTaskCreate(ultrasonicTask, "Ultrasonic", 3072, nullptr, 2, nullptr);
  xTaskCreate(dhtTask, "DHT11", 3072, nullptr, 2, nullptr);
  xTaskCreate(waterTask, "Water", 2048, nullptr, 2, nullptr);
  xTaskCreate(serialTask, "SerialLogger", 4096, nullptr, 1, nullptr);

  if (canReady) {
    xTaskCreate(canTask, "CAN", 4096, nullptr, 2, nullptr);
  }
}

// =====================================================
// 13. MAIN LOOP
// =====================================================

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}