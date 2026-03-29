/*
 * SENTINEL — ESP32 Multi-Sensor Hub Firmware
 *
 * Connects to SENTINEL backend via MQTT.
 * Sensors supported:
 *   - PIR Motion Sensor (HC-SR501)
 *   - Ultrasonic Distance Sensor (HC-SR04)
 *   - Magnetic Reed Switch (door/window)
 *   - Temperature + Humidity (DHT22)
 *   - Ambient Light (LDR)
 *   - Sound Level (analog mic module)
 *   - Vibration (SW-420)
 *
 * Wiring:
 *   PIR         → GPIO 27
 *   HC-SR04     → TRIG: GPIO 26, ECHO: GPIO 25
 *   Reed Switch → GPIO 33
 *   DHT22       → GPIO 32
 *   LDR         → GPIO 34 (ADC)
 *   Sound       → GPIO 35 (ADC)
 *   Vibration   → GPIO 14
 *   Status LED  → GPIO 2 (onboard)
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

// ==================== CONFIGURATION ====================
// WiFi
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD  = "YOUR_WIFI_PASSWORD";

// MQTT Broker (SENTINEL backend IP)
const char* MQTT_BROKER    = "192.168.1.100";
const int   MQTT_PORT      = 1883;
const char* MQTT_USER      = "";
const char* MQTT_PASS      = "";

// Device Identity
const char* DEVICE_ID      = "sensor_hub_01";
const char* ZONE           = "front_door";

// ==================== PIN DEFINITIONS ====================
#define PIN_PIR         27
#define PIN_TRIG        26
#define PIN_ECHO        25
#define PIN_REED        33
#define PIN_DHT         32
#define PIN_LDR         34
#define PIN_SOUND       35
#define PIN_VIBRATION   14
#define PIN_LED         2

// ==================== TIMING ====================
#define HEARTBEAT_INTERVAL     30000   // 30s device status
#define SENSOR_READ_INTERVAL   1000    // 1s sensor poll
#define ULTRASONIC_INTERVAL    500     // 500ms distance check
#define DHT_INTERVAL           10000   // 10s temp/humidity
#define MOTION_COOLDOWN        5000    // 5s between motion reports

// ==================== GLOBALS ====================
WiFiClient espClient;
PubSubClient mqtt(espClient);
DHT dht(PIN_DHT, DHT22);

unsigned long lastHeartbeat = 0;
unsigned long lastSensorRead = 0;
unsigned long lastUltrasonic = 0;
unsigned long lastDHT = 0;
unsigned long lastMotionReport = 0;
unsigned long lastVibration = 0;

bool lastPirState = false;
bool lastReedState = false;
int motionCount = 0;

char topicBuf[128];
char payloadBuf[256];

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== SENTINEL Sensor Hub ===");

  // Pin modes
  pinMode(PIN_PIR, INPUT);
  pinMode(PIN_REED, INPUT_PULLUP);
  pinMode(PIN_VIBRATION, INPUT);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_LED, OUTPUT);

  // DHT sensor
  dht.begin();

  // Connect WiFi
  connectWiFi();

  // Setup MQTT
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  mqtt.setBufferSize(512);

  connectMQTT();

  // Initial heartbeat
  sendHeartbeat();

  Serial.println("Sensor hub ready.");
  blinkLED(3, 100);
}

// ==================== MAIN LOOP ====================
void loop() {
  if (!mqtt.connected()) {
    connectMQTT();
  }
  mqtt.loop();

  unsigned long now = millis();

  // Heartbeat
  if (now - lastHeartbeat >= HEARTBEAT_INTERVAL) {
    sendHeartbeat();
    lastHeartbeat = now;
  }

  // PIR Motion
  bool pirState = digitalRead(PIN_PIR);
  if (pirState && !lastPirState && (now - lastMotionReport >= MOTION_COOLDOWN)) {
    motionCount++;
    publishSensor("pir", 1, "", "Motion detected");
    lastMotionReport = now;
    Serial.println("[PIR] Motion detected!");
  }
  lastPirState = pirState;

  // Reed Switch (door/window) — LOW = closed (magnet near), HIGH = open
  bool reedState = digitalRead(PIN_REED);
  if (reedState != lastReedState) {
    publishSensor("reed", reedState ? 1 : 0, "", reedState ? "OPENED" : "CLOSED");
    Serial.printf("[REED] %s\n", reedState ? "OPENED" : "CLOSED");
    lastReedState = reedState;
  }

  // Vibration sensor
  bool vibState = digitalRead(PIN_VIBRATION);
  if (vibState && (now - lastVibration >= 2000)) {
    int intensity = analogRead(PIN_VIBRATION);
    publishSensor("vibration", intensity, "raw", "Vibration detected");
    Serial.printf("[VIB] Intensity: %d\n", intensity);
    lastVibration = now;
  }

  // Ultrasonic distance
  if (now - lastUltrasonic >= ULTRASONIC_INTERVAL) {
    float distance = readUltrasonic();
    if (distance > 0 && distance < 400) {
      publishSensor("ultrasonic", distance, "cm", "");
      if (distance < 50) {
        Serial.printf("[ULTRA] CLOSE: %.1f cm\n", distance);
      }
    }
    lastUltrasonic = now;
  }

  // Analog sensors (LDR, Sound)
  if (now - lastSensorRead >= SENSOR_READ_INTERVAL) {
    int ldrRaw = analogRead(PIN_LDR);
    float lux = map(ldrRaw, 0, 4095, 0, 1000);
    publishSensor("light", lux, "lux", "");

    int soundRaw = analogRead(PIN_SOUND);
    float db = map(soundRaw, 0, 4095, 30, 120);
    publishSensor("sound", db, "dB", "");

    lastSensorRead = now;
  }

  // DHT22 Temperature/Humidity
  if (now - lastDHT >= DHT_INTERVAL) {
    float temp = dht.readTemperature();
    float hum = dht.readHumidity();
    if (!isnan(temp)) publishSensor("temperature", temp, "C", "");
    if (!isnan(hum)) publishSensor("humidity", hum, "%", "");
    lastDHT = now;
  }
}

// ==================== ULTRASONIC ====================
float readUltrasonic() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  long duration = pulseIn(PIN_ECHO, HIGH, 30000); // 30ms timeout
  if (duration == 0) return -1;
  return (duration * 0.0343) / 2.0;
}

// ==================== MQTT ====================
void publishSensor(const char* sensorType, float value, const char* unit, const char* note) {
  StaticJsonDocument<200> doc;
  doc["value"] = value;
  doc["unit"] = unit;
  doc["zone"] = ZONE;
  doc["device_id"] = DEVICE_ID;
  if (strlen(note) > 0) doc["note"] = note;

  snprintf(topicBuf, sizeof(topicBuf), "sentinel/sensors/%s/%s", DEVICE_ID, sensorType);
  serializeJson(doc, payloadBuf, sizeof(payloadBuf));
  mqtt.publish(topicBuf, payloadBuf);
}

void sendHeartbeat() {
  StaticJsonDocument<300> doc;
  doc["type"] = "sensor_hub";
  doc["firmware"] = "1.0.0";
  doc["zone"] = ZONE;
  doc["ip"] = WiFi.localIP().toString();
  doc["rssi"] = WiFi.RSSI();
  doc["uptime"] = millis() / 1000;
  doc["motion_count"] = motionCount;
  doc["free_heap"] = ESP.getFreeHeap();

  JsonArray caps = doc.createNestedArray("capabilities");
  caps.add("pir");
  caps.add("ultrasonic");
  caps.add("reed");
  caps.add("temperature");
  caps.add("humidity");
  caps.add("light");
  caps.add("sound");
  caps.add("vibration");

  snprintf(topicBuf, sizeof(topicBuf), "sentinel/devices/%s/status", DEVICE_ID);
  serializeJson(doc, payloadBuf, sizeof(payloadBuf));
  mqtt.publish(topicBuf, payloadBuf);

  Serial.println("[MQTT] Heartbeat sent");
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  payload[length] = '\0';
  String msg = String((char*)payload);
  Serial.printf("[MQTT] %s: %s\n", topic, msg.c_str());

  // Handle config updates
  String topicStr = String(topic);
  if (topicStr.endsWith("/config")) {
    StaticJsonDocument<256> doc;
    if (deserializeJson(doc, msg) == DeserializationError::Ok) {
      // Apply config changes
      Serial.println("[CONFIG] Configuration updated");
    }
  }
}

// ==================== WIFI ====================
void connectWiFi() {
  Serial.printf("Connecting to WiFi: %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nWiFi connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWiFi FAILED — restarting...");
    ESP.restart();
  }
}

void connectMQTT() {
  while (!mqtt.connected()) {
    Serial.print("MQTT connecting...");
    String clientId = String("sentinel-") + DEVICE_ID;
    bool connected;
    if (strlen(MQTT_USER) > 0) {
      connected = mqtt.connect(clientId.c_str(), MQTT_USER, MQTT_PASS);
    } else {
      connected = mqtt.connect(clientId.c_str());
    }
    if (connected) {
      Serial.println(" connected!");
      // Subscribe to config and alert topics
      snprintf(topicBuf, sizeof(topicBuf), "sentinel/devices/%s/config", DEVICE_ID);
      mqtt.subscribe(topicBuf);
      snprintf(topicBuf, sizeof(topicBuf), "sentinel/alerts/%s/trigger", DEVICE_ID);
      mqtt.subscribe(topicBuf);
      mqtt.subscribe("sentinel/system/threat");
      mqtt.subscribe("sentinel/system/command");
      digitalWrite(PIN_LED, HIGH);
    } else {
      Serial.printf(" failed (rc=%d). Retry in 5s...\n", mqtt.state());
      digitalWrite(PIN_LED, LOW);
      delay(5000);
    }
  }
}

void blinkLED(int times, int ms) {
  for (int i = 0; i < times; i++) {
    digitalWrite(PIN_LED, HIGH);
    delay(ms);
    digitalWrite(PIN_LED, LOW);
    delay(ms);
  }
}
