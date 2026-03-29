/*
 * SENTINEL — ESP32 Siren + LED Alert Controller Firmware
 *
 * Receives alert commands from SENTINEL via MQTT and activates:
 *   - Piezo/siren buzzer
 *   - NeoPixel LED strip (WS2812B) for visual alerts
 *   - Relay for external siren / door lock
 *
 * Wiring:
 *   Piezo Buzzer   → GPIO 25
 *   NeoPixel Strip → GPIO 13 (data)
 *   Relay CH1      → GPIO 26 (siren)
 *   Relay CH2      → GPIO 27 (door lock)
 *   Status LED     → GPIO 2
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Adafruit_NeoPixel.h>

// ==================== CONFIG ====================
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD  = "YOUR_WIFI_PASSWORD";
const char* MQTT_BROKER    = "192.168.1.100";
const int   MQTT_PORT      = 1883;
const char* DEVICE_ID      = "siren_ctrl_01";
const char* ZONE           = "front_door";

// ==================== PINS ====================
#define PIN_BUZZER      25
#define PIN_NEOPIXEL    13
#define PIN_RELAY_SIREN 26
#define PIN_RELAY_LOCK  27
#define PIN_LED         2

#define NUM_PIXELS      30  // Number of LEDs in strip
#define BUZZER_CHANNEL  0

// ==================== GLOBALS ====================
WiFiClient espClient;
PubSubClient mqtt(espClient);
Adafruit_NeoPixel strip(NUM_PIXELS, PIN_NEOPIXEL, NEO_GRB + NEO_KHZ800);

bool sirenActive = false;
bool ledActive = false;
bool lockActive = false;
unsigned long sirenEndTime = 0;
unsigned long ledEndTime = 0;
String currentPattern = "off";
uint32_t alertColor = strip.Color(255, 0, 0);
String currentThreatLevel = "NORMAL";

char topicBuf[128];
char payloadBuf[512];

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== SENTINEL Siren Controller ===");

  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_RELAY_SIREN, OUTPUT);
  pinMode(PIN_RELAY_LOCK, OUTPUT);
  pinMode(PIN_LED, OUTPUT);

  // Start with everything off
  digitalWrite(PIN_RELAY_SIREN, LOW);
  digitalWrite(PIN_RELAY_LOCK, LOW);
  ledcSetup(BUZZER_CHANNEL, 2000, 8);
  ledcAttachPin(PIN_BUZZER, BUZZER_CHANNEL);

  // NeoPixel
  strip.begin();
  strip.setBrightness(150);
  strip.clear();
  strip.show();

  connectWiFi();
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  mqtt.setBufferSize(512);
  connectMQTT();
  sendHeartbeat();

  // Boot animation
  bootAnimation();
  Serial.println("Siren controller ready.");
}

// ==================== MAIN LOOP ====================
void loop() {
  if (!mqtt.connected()) connectMQTT();
  mqtt.loop();

  unsigned long now = millis();

  // Auto-stop siren after duration
  if (sirenActive && sirenEndTime > 0 && now >= sirenEndTime) {
    stopSiren();
  }

  // Auto-stop LED after duration
  if (ledActive && ledEndTime > 0 && now >= ledEndTime) {
    stopLED();
  }

  // LED pattern animation
  if (ledActive) {
    animateLED(now);
  }

  // Siren pattern
  if (sirenActive) {
    animateSiren(now);
  }

  // Heartbeat
  static unsigned long lastHB = 0;
  if (now - lastHB >= 30000) {
    sendHeartbeat();
    lastHB = now;
  }
}

// ==================== MQTT CALLBACK ====================
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  payload[length] = '\0';
  String msg = String((char*)payload);
  Serial.printf("[MQTT] %s: %s\n", topic, msg.c_str());

  String topicStr = String(topic);

  // Alert trigger
  if (topicStr.indexOf("/trigger") >= 0) {
    handleAlertCommand(msg);
  }

  // System threat level
  if (topicStr == "sentinel/system/threat") {
    handleThreatLevel(msg);
  }

  // System commands
  if (topicStr == "sentinel/system/command") {
    StaticJsonDocument<128> doc;
    if (deserializeJson(doc, msg) == DeserializationError::Ok) {
      String cmd = doc["command"].as<String>();
      if (cmd == "lockdown") {
        activateLockdown();
      } else if (cmd == "disarm") {
        stopAll();
      }
    }
  }
}

void handleAlertCommand(String& msg) {
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, msg) != DeserializationError::Ok) return;

  String alertType = doc["alert_type"].as<String>();
  String action = doc["action"].as<String>();
  int duration = doc["duration"] | 10;
  int intensity = doc["intensity"] | 100;
  String color = doc["color"] | "#ff0000";

  Serial.printf("[ALERT] type=%s action=%s duration=%d\n",
    alertType.c_str(), action.c_str(), duration);

  if (alertType == "siren") {
    if (action == "on" || action == "pattern") {
      startSiren(duration, intensity);
    } else if (action == "off") {
      stopSiren();
    }
  }

  if (alertType == "led") {
    if (action == "on" || action == "pulse" || action == "pattern") {
      alertColor = parseColor(color);
      currentPattern = action;
      startLED(duration);
    } else if (action == "off") {
      stopLED();
    }
  }

  if (alertType == "lock") {
    if (action == "on") {
      activateLock();
    } else if (action == "off") {
      deactivateLock();
    }
  }
}

void handleThreatLevel(String& msg) {
  StaticJsonDocument<128> doc;
  if (deserializeJson(doc, msg) != DeserializationError::Ok) return;

  currentThreatLevel = doc["level"].as<String>();

  // Ambient LED based on threat level
  if (!ledActive) {
    if (currentThreatLevel == "CRITICAL") {
      setStripColor(strip.Color(255, 0, 0), 50);
    } else if (currentThreatLevel == "SUSPICIOUS") {
      setStripColor(strip.Color(255, 200, 0), 30);
    } else {
      setStripColor(strip.Color(0, 255, 65), 10);
    }
  }
}

// ==================== SIREN ====================
void startSiren(int durationSec, int intensity) {
  sirenActive = true;
  sirenEndTime = millis() + (durationSec * 1000UL);
  digitalWrite(PIN_RELAY_SIREN, HIGH);
  Serial.printf("[SIREN] ON for %ds\n", durationSec);
}

void stopSiren() {
  sirenActive = false;
  sirenEndTime = 0;
  digitalWrite(PIN_RELAY_SIREN, LOW);
  ledcWriteTone(BUZZER_CHANNEL, 0);
  Serial.println("[SIREN] OFF");
}

void animateSiren(unsigned long now) {
  // Two-tone siren
  int phase = (now / 500) % 2;
  if (phase == 0) {
    ledcWriteTone(BUZZER_CHANNEL, 1200);
  } else {
    ledcWriteTone(BUZZER_CHANNEL, 800);
  }
}

// ==================== LED STRIP ====================
void startLED(int durationSec) {
  ledActive = true;
  ledEndTime = millis() + (durationSec * 1000UL);
  Serial.printf("[LED] ON pattern=%s for %ds\n", currentPattern.c_str(), durationSec);
}

void stopLED() {
  ledActive = false;
  ledEndTime = 0;
  strip.clear();
  strip.show();
  Serial.println("[LED] OFF");
}

void animateLED(unsigned long now) {
  if (currentPattern == "pulse") {
    // Breathing pulse
    int brightness = (sin(now / 500.0) + 1) * 75 + 5;
    strip.setBrightness(brightness);
    setStripColor(alertColor, brightness);
  }
  else if (currentPattern == "pattern") {
    // Red/blue police strobe
    int phase = (now / 100) % 4;
    strip.clear();
    for (int i = 0; i < NUM_PIXELS; i++) {
      if (phase < 2) {
        strip.setPixelColor(i, (i % 2 == phase % 2) ? strip.Color(255, 0, 0) : strip.Color(0, 0, 255));
      } else {
        strip.setPixelColor(i, (i % 2 == phase % 2) ? strip.Color(0, 0, 255) : strip.Color(255, 0, 0));
      }
    }
    strip.show();
  }
  else { // "on"
    setStripColor(alertColor, 150);
  }
}

void setStripColor(uint32_t color, int brightness) {
  strip.setBrightness(brightness);
  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, color);
  }
  strip.show();
}

uint32_t parseColor(String hex) {
  hex.replace("#", "");
  long rgb = strtol(hex.c_str(), NULL, 16);
  return strip.Color((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF);
}

// ==================== LOCK RELAY ====================
void activateLock() {
  lockActive = true;
  digitalWrite(PIN_RELAY_LOCK, HIGH);
  Serial.println("[LOCK] ENGAGED");
}

void deactivateLock() {
  lockActive = false;
  digitalWrite(PIN_RELAY_LOCK, LOW);
  Serial.println("[LOCK] RELEASED");
}

void activateLockdown() {
  startSiren(300, 100);
  alertColor = strip.Color(255, 0, 0);
  currentPattern = "pattern";
  startLED(300);
  activateLock();
  Serial.println("!!! LOCKDOWN !!!");
}

void stopAll() {
  stopSiren();
  stopLED();
  deactivateLock();
}

// ==================== BOOT ANIMATION ====================
void bootAnimation() {
  // Green chase
  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, strip.Color(0, 255, 65));
    strip.show();
    delay(20);
  }
  delay(200);
  strip.clear();
  strip.show();
}

// ==================== HEARTBEAT ====================
void sendHeartbeat() {
  StaticJsonDocument<256> doc;
  doc["type"] = "siren_controller";
  doc["firmware"] = "1.0.0";
  doc["zone"] = ZONE;
  doc["ip"] = WiFi.localIP().toString();
  doc["rssi"] = WiFi.RSSI();
  doc["siren_active"] = sirenActive;
  doc["led_active"] = ledActive;
  doc["lock_active"] = lockActive;

  JsonArray caps = doc.createNestedArray("capabilities");
  caps.add("siren");
  caps.add("led");
  caps.add("lock");
  caps.add("alert");

  snprintf(topicBuf, sizeof(topicBuf), "sentinel/devices/%s/status", DEVICE_ID);
  serializeJson(doc, payloadBuf, sizeof(payloadBuf));
  mqtt.publish(topicBuf, payloadBuf);
}

// ==================== CONNECTIVITY ====================
void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("WiFi: %s", WIFI_SSID);
  int t = 0;
  while (WiFi.status() != WL_CONNECTED && t < 30) { delay(500); Serial.print("."); t++; }
  if (WiFi.status() == WL_CONNECTED) Serial.printf("\nIP: %s\n", WiFi.localIP().toString().c_str());
  else { Serial.println("\nWiFi fail!"); ESP.restart(); }
}

void connectMQTT() {
  while (!mqtt.connected()) {
    Serial.print("MQTT...");
    String cid = String("sentinel-") + DEVICE_ID;
    if (mqtt.connect(cid.c_str())) {
      Serial.println("ok!");
      snprintf(topicBuf, sizeof(topicBuf), "sentinel/alerts/%s/trigger", DEVICE_ID);
      mqtt.subscribe(topicBuf);
      mqtt.subscribe("sentinel/system/threat");
      mqtt.subscribe("sentinel/system/command");
    } else {
      Serial.printf("fail(%d)\n", mqtt.state());
      delay(5000);
    }
  }
}
