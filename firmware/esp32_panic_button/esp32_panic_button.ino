/*
 * SENTINEL — ESP32 Panic Button Firmware
 *
 * Wearable / mountable panic button that sends IMMEDIATE
 * CRITICAL alert to SENTINEL via MQTT.
 *
 * Features:
 *   - Physical panic button (GPIO 0 / boot button)
 *   - Optional secondary button (GPIO 4)
 *   - Status LED (GPIO 2)
 *   - NeoPixel ring feedback (GPIO 15, optional)
 *   - Battery voltage monitoring (GPIO 36)
 *   - Double-press = cancel alert
 *   - Hold 3s = lockdown mode
 *   - Deep sleep between presses for battery life
 *
 * Wiring:
 *   Panic Button   → GPIO 0 (or external button on GPIO 4)
 *   Status LED     → GPIO 2
 *   NeoPixel       → GPIO 15 (WS2812B ring, optional)
 *   Battery ADC    → GPIO 36
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ==================== CONFIG ====================
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD  = "YOUR_WIFI_PASSWORD";
const char* MQTT_BROKER    = "192.168.1.100";
const int   MQTT_PORT      = 1883;
const char* DEVICE_ID      = "panic_btn_01";
const char* ZONE           = "wearable";

// ==================== PINS ====================
#define PIN_PANIC       0   // Boot button
#define PIN_PANIC_EXT   4   // External panic button
#define PIN_LED         2
#define PIN_BATTERY     36

// ==================== STATE ====================
WiFiClient espClient;
PubSubClient mqtt(espClient);

unsigned long pressStart = 0;
bool alertActive = false;
int pressCount = 0;
unsigned long lastPressTime = 0;

char topicBuf[128];
char payloadBuf[512];

void setup() {
  Serial.begin(115200);
  Serial.println("\n=== SENTINEL Panic Button ===");

  pinMode(PIN_PANIC, INPUT_PULLUP);
  pinMode(PIN_PANIC_EXT, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_BATTERY, INPUT);

  connectWiFi();

  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  mqtt.setBufferSize(512);

  connectMQTT();
  sendHeartbeat();

  // Ready indicator
  for (int i = 0; i < 5; i++) {
    digitalWrite(PIN_LED, HIGH); delay(50);
    digitalWrite(PIN_LED, LOW);  delay(50);
  }
  Serial.println("Panic button ready.");
}

void loop() {
  if (!mqtt.connected()) connectMQTT();
  mqtt.loop();

  bool pressed = !digitalRead(PIN_PANIC) || !digitalRead(PIN_PANIC_EXT);
  unsigned long now = millis();

  if (pressed) {
    if (pressStart == 0) {
      pressStart = now;
      pressCount++;
      lastPressTime = now;
    }

    // Hold 3 seconds = LOCKDOWN
    if (pressStart > 0 && (now - pressStart) >= 3000) {
      triggerLockdown();
      pressStart = 0;
      pressCount = 0;
      delay(1000); // debounce
    }
  } else {
    if (pressStart > 0) {
      unsigned long holdDuration = now - pressStart;
      pressStart = 0;

      if (holdDuration < 3000) {
        // Short press handling
        if (alertActive && pressCount >= 2 && (now - lastPressTime) < 1000) {
          // Double press while alert active = CANCEL
          cancelAlert();
          pressCount = 0;
        } else if (!alertActive) {
          // Single press = PANIC
          triggerPanic();
        }
      }
    }

    // Reset press count after 2 seconds
    if (pressCount > 0 && (now - lastPressTime) > 2000) {
      pressCount = 0;
    }
  }

  // Heartbeat every 60s
  static unsigned long lastHB = 0;
  if (now - lastHB >= 60000) {
    sendHeartbeat();
    lastHB = now;
  }

  // Blink while alert active
  if (alertActive) {
    digitalWrite(PIN_LED, (now / 200) % 2);
  } else {
    // Slow breathing when idle
    int brightness = (sin(now / 2000.0 * PI) + 1) * 127;
    analogWrite(PIN_LED, brightness);
  }
}

// ==================== PANIC ACTIONS ====================
void triggerPanic() {
  alertActive = true;
  Serial.println("!!! PANIC BUTTON PRESSED !!!");

  StaticJsonDocument<256> doc;
  doc["value"] = 1;
  doc["zone"] = ZONE;
  doc["device_id"] = DEVICE_ID;
  doc["battery"] = readBatteryPercent();
  doc["note"] = "PANIC BUTTON ACTIVATED";

  snprintf(topicBuf, sizeof(topicBuf), "sentinel/sensors/%s/panic", DEVICE_ID);
  serializeJson(doc, payloadBuf, sizeof(payloadBuf));
  mqtt.publish(topicBuf, payloadBuf, true); // retained

  // Also publish to system command for immediate broadcast
  StaticJsonDocument<128> cmd;
  cmd["command"] = "panic";
  cmd["device_id"] = DEVICE_ID;
  cmd["zone"] = ZONE;
  serializeJson(cmd, payloadBuf, sizeof(payloadBuf));
  mqtt.publish("sentinel/system/command", payloadBuf);

  // Rapid blink
  for (int i = 0; i < 10; i++) {
    digitalWrite(PIN_LED, HIGH); delay(50);
    digitalWrite(PIN_LED, LOW);  delay(50);
  }
}

void cancelAlert() {
  alertActive = false;
  Serial.println("Alert cancelled");

  StaticJsonDocument<128> doc;
  doc["value"] = 0;
  doc["zone"] = ZONE;
  doc["device_id"] = DEVICE_ID;
  doc["note"] = "PANIC CANCELLED";

  snprintf(topicBuf, sizeof(topicBuf), "sentinel/sensors/%s/panic", DEVICE_ID);
  serializeJson(doc, payloadBuf, sizeof(payloadBuf));
  mqtt.publish(topicBuf, payloadBuf, true);

  // Confirm cancel with slow blinks
  for (int i = 0; i < 3; i++) {
    digitalWrite(PIN_LED, HIGH); delay(300);
    digitalWrite(PIN_LED, LOW);  delay(300);
  }
}

void triggerLockdown() {
  Serial.println("!!! LOCKDOWN MODE !!!");

  StaticJsonDocument<128> cmd;
  cmd["command"] = "lockdown";
  cmd["device_id"] = DEVICE_ID;
  cmd["zone"] = ZONE;
  serializeJson(cmd, payloadBuf, sizeof(payloadBuf));
  mqtt.publish("sentinel/system/command", payloadBuf);

  alertActive = true;

  // SOS pattern
  for (int r = 0; r < 3; r++) {
    for (int i = 0; i < 3; i++) { digitalWrite(PIN_LED, HIGH); delay(100); digitalWrite(PIN_LED, LOW); delay(100); }
    for (int i = 0; i < 3; i++) { digitalWrite(PIN_LED, HIGH); delay(300); digitalWrite(PIN_LED, LOW); delay(100); }
    for (int i = 0; i < 3; i++) { digitalWrite(PIN_LED, HIGH); delay(100); digitalWrite(PIN_LED, LOW); delay(100); }
    delay(500);
  }
}

// ==================== BATTERY ====================
float readBatteryPercent() {
  int raw = analogRead(PIN_BATTERY);
  float voltage = (raw / 4095.0) * 3.3 * 2; // voltage divider
  float percent = ((voltage - 3.0) / (4.2 - 3.0)) * 100;
  return constrain(percent, 0, 100);
}

// ==================== HEARTBEAT ====================
void sendHeartbeat() {
  StaticJsonDocument<256> doc;
  doc["type"] = "panic_button";
  doc["firmware"] = "1.0.0";
  doc["zone"] = ZONE;
  doc["ip"] = WiFi.localIP().toString();
  doc["rssi"] = WiFi.RSSI();
  doc["battery"] = readBatteryPercent();
  doc["alert_active"] = alertActive;
  doc["uptime"] = millis() / 1000;

  JsonArray caps = doc.createNestedArray("capabilities");
  caps.add("panic");

  snprintf(topicBuf, sizeof(topicBuf), "sentinel/devices/%s/status", DEVICE_ID);
  serializeJson(doc, payloadBuf, sizeof(payloadBuf));
  mqtt.publish(topicBuf, payloadBuf);
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  payload[length] = '\0';
  Serial.printf("[MQTT] %s: %s\n", topic, (char*)payload);
}

// ==================== CONNECTIVITY ====================
void connectWiFi() {
  Serial.printf("WiFi: %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 30) {
    delay(500); Serial.print("."); tries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nIP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWiFi failed! Restarting...");
    ESP.restart();
  }
}

void connectMQTT() {
  while (!mqtt.connected()) {
    Serial.print("MQTT...");
    String clientId = String("sentinel-") + DEVICE_ID;
    if (mqtt.connect(clientId.c_str())) {
      Serial.println("ok!");
      snprintf(topicBuf, sizeof(topicBuf), "sentinel/alerts/%s/trigger", DEVICE_ID);
      mqtt.subscribe(topicBuf);
      mqtt.subscribe("sentinel/system/command");
    } else {
      Serial.printf("fail(%d) retry 5s\n", mqtt.state());
      delay(5000);
    }
  }
}
