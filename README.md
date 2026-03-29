# SENTINEL

**Autonomous AI Personal Safety System with IoT Integration**

Real-time threat detection powered by computer vision, deep learning, and IoT sensor fusion. SENTINEL monitors your environment through camera AI and connected sensors, classifies threats, tracks repeat visitors across multiple days, and triggers coordinated physical alerts.

---

## Features

**AI Vision Pipeline**
- Real-time pose estimation (MediaPipe) and object/weapon detection (YOLOv8)
- 5 custom PyTorch ML models: threat classifier, temporal predictor, anomaly detector, person re-identification, weapon context analyzer
- Night vision filter with green phosphor, scanlines, and CRT effects
- Persistent multi-day follower tracking with risk scoring

**IoT Sensor Fusion**
- MQTT-based communication with ESP32/Arduino sensor nodes
- PIR motion, ultrasonic distance, door/window reed switches, panic buttons, vibration, sound level
- Physical alert outputs: siren, NeoPixel LED strip, door lock relay
- Camera + IoT sensor correlation for multi-sensor threat escalation

**Smart Home Integration**
- Home Assistant (REST API + events)
- IFTTT Webhooks
- Telegram Bot notifications
- ntfy.sh push notifications
- Custom webhook endpoints

**Security Zones & Lockdown**
- Zone-based perimeter management (arm/disarm individual zones)
- Full lockdown mode: sirens + locks + LED alerts + notifications
- Panic button support with instant critical alert bypass

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              SENTINEL SERVER                  │
                    │              (FastAPI + Python)               │
                    │                                              │
    ┌───────┐       │  ┌───────────┐  ┌──────────┐  ┌──────────┐ │
    │Webcam │──────►│  │ Inference │  │ Threat   │  │ Follower │ │
    │       │       │  │ Engine    │  │ Assessor │  │ Tracker  │ │
    └───────┘       │  │ MediaPipe │  │ ML+Rules │  │ Re-ID DB │ │
                    │  │ YOLOv8    │  └────┬─────┘  └──────────┘ │
                    │  │ PyTorch   │       │                      │
                    │  └─────┬─────┘       │   ┌──────────────┐  │
    ┌───────┐       │        │             ├──►│ Notification │  │
    │ESP32  │──MQTT─│──►┌────┴─────┐       │   │ Service      │  │
    │Sensors│       │   │ Device   │───────┘   │ ntfy/Telegram│  │
    └───────┘       │   │ Manager  │           └──────────────┘  │
                    │   │ Sensor   │                              │
    ┌───────┐       │   │ Fusion   │──►┌──────────────┐          │
    │ESP32  │──MQTT─│──►└──────────┘   │ Smart Home   │          │
    │Siren  │◄─MQTT─│                  │ HA/IFTTT/    │          │
    └───────┘       │                  │ Webhooks     │          │
                    │                  └──────────────┘          │
    ┌───────┐       │                                            │
    │ESP32  │──MQTT─│──►┌──────────┐                             │
    │Panic  │       │   │ Alert    │──►Siren+LED+Lock            │
    └───────┘       │   │ Router   │                             │
                    │   └──────────┘                             │
                    │                                            │
                    │  WebSocket ──► Browser Dashboard            │
                    │  REST API  ──► Mobile / Integrations       │
                    └──────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.10-3.12 (PyTorch requirement)
- Webcam
- Optional: ESP32 boards, MQTT broker (Mosquitto)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/sentinel.git
cd sentinel
bash setup.sh
source venv/bin/activate
```

### Train ML Models

```bash
python run.py generate-data          # Generate synthetic training data
python run.py train                  # Train all 5 models
python run.py train --model threat_mlp  # Train specific model
```

### Run

```bash
python run.py serve                  # Start server on port 8099
```

Open `http://localhost:8099` in your browser.

Access from phone on same WiFi: `http://<your-ip>:8099`

### Standalone Mode (No Backend)

Open `sentinel.html` directly in Chrome — runs entirely client-side with MediaPipe + COCO-SSD.

---

## ML Models

| Model | Architecture | Input | Purpose |
|---|---|---|---|
| ThreatClassifierMLP | 3-layer MLP | 103-dim pose features | Real-time threat classification |
| ThreatSequenceLSTM | Bidirectional LSTM + Attention | 30-frame pose sequences | Temporal threat prediction |
| AnomalyAutoencoder | Autoencoder | 99-dim pose | Unsupervised anomaly detection |
| PersonReIDNet | Embedding net + Triplet Loss | 99-dim pose | Follower re-identification |
| WeaponContextClassifier | Dual-branch fusion | Pose + object features | Weapon + posture context |

### Threat Classification

```
Confidence = proximity(20%) + velocity(20%) + posture(20%) + weapon(30%) + follower(10%)

NORMAL     → No threats detected
SUSPICIOUS → Loitering, slow approach, suspicious objects, repeat presence
CRITICAL   → Weapon detected, rapid approach, aggressive posture, confirmed follower
```

---

## IoT Integration

### MQTT Topic Schema

```
sentinel/sensors/{device_id}/{sensor_type}   → Sensor readings (incoming)
sentinel/alerts/{device_id}/trigger          → Alert commands (outgoing)
sentinel/devices/{device_id}/status          → Device heartbeat
sentinel/system/threat                       → Threat level broadcast
sentinel/system/command                      → System commands (arm/disarm/lockdown)
```

### Supported Sensors

| Sensor | Type | Trigger |
|---|---|---|
| PIR (HC-SR501) | Motion | SUSPICIOUS on detection |
| Ultrasonic (HC-SR04) | Distance | CRITICAL < 50cm, SUSPICIOUS < 150cm |
| Reed Switch | Door/Window | CRITICAL on open |
| Panic Button | Manual | CRITICAL (bypasses cooldown) |
| Vibration (SW-420) | Forced entry | CRITICAL > 500 threshold |
| Sound (Analog mic) | Noise level | SUSPICIOUS > 85dB |
| DHT22 | Temperature/Humidity | Environmental monitoring |
| LDR | Ambient light | Environmental monitoring |

### ESP32 Firmware

Three ready-to-flash firmware sketches in `/firmware/`:

**Sensor Hub** (`esp32_sensor_hub/`)
- Multi-sensor node: PIR + ultrasonic + reed + DHT22 + LDR + sound + vibration
- Auto-reconnect WiFi + MQTT

**Panic Button** (`esp32_panic_button/`)
- Single press = PANIC alert
- Double press = cancel
- Hold 3s = LOCKDOWN
- Battery monitoring for wearable use

**Siren Controller** (`esp32_siren_controller/`)
- Piezo buzzer with two-tone siren
- NeoPixel LED strip (police strobe patterns)
- Relay for external siren + door lock
- Receives threat level broadcasts for ambient lighting

### Wiring: Sensor Hub

```
ESP32           Sensor
─────           ──────
GPIO 27    ───  PIR OUT
GPIO 26    ───  HC-SR04 TRIG
GPIO 25    ───  HC-SR04 ECHO
GPIO 33    ───  Reed Switch
GPIO 32    ───  DHT22 DATA
GPIO 34    ───  LDR (voltage divider)
GPIO 35    ───  Sound module AO
GPIO 14    ───  Vibration DO
3.3V       ───  Sensor VCC
GND        ───  Sensor GND
```

### Wiring: Siren Controller

```
ESP32           Component
─────           ─────────
GPIO 25    ───  Piezo Buzzer +
GPIO 13    ───  NeoPixel DIN
GPIO 26    ───  Relay CH1 (Siren)
GPIO 27    ───  Relay CH2 (Lock)
5V         ───  NeoPixel VCC / Relay VCC
GND        ───  Common GND
```

### MQTT Broker Setup

Install Mosquitto:
```bash
# macOS
brew install mosquitto
brew services start mosquitto

# Ubuntu
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

---

## API Reference

### WebSocket

`ws://localhost:8099/ws/stream` — Real-time video + detection data

### REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | System status |
| GET | `/api/followers` | All tracked followers |
| POST | `/api/followers/{id}/mark-safe` | Mark follower as safe |
| GET | `/api/incidents` | Incident log |
| GET | `/api/recordings` | Recording list |
| POST | `/api/train/{model}` | Trigger model training |
| GET | `/api/iot/status` | IoT system status |
| GET | `/api/iot/devices` | Connected devices |
| POST | `/api/iot/alert` | Manual alert trigger |
| POST | `/api/iot/zones/{name}/arm` | Arm security zone |
| POST | `/api/iot/lockdown` | Activate lockdown |
| POST | `/api/iot/integrations/webhook` | Add webhook |
| GET | `/docs` | Interactive API docs (Swagger) |

---

## Configuration

Environment variables (in `.env`):

```env
SENTINEL_CAMERA_INDEX=0
SENTINEL_SENSITIVITY=medium
SENTINEL_NTFY_TOPIC=sentinel-myname
SENTINEL_MQTT_BROKER_HOST=localhost
SENTINEL_MQTT_BROKER_PORT=1883
SENTINEL_HA_URL=http://homeassistant.local:8123
SENTINEL_HA_TOKEN=your_long_lived_token
SENTINEL_IFTTT_KEY=your_key
SENTINEL_TELEGRAM_BOT_TOKEN=your_bot_token
SENTINEL_TELEGRAM_CHAT_ID=your_chat_id
```

---

## Project Structure

```
sentinel/
├── run.py                    # CLI entry point
├── setup.sh                  # One-command setup
├── sentinel.html             # Standalone frontend (no backend)
├── sentinel/
│   ├── config.py             # Central configuration
│   ├── api/server.py         # FastAPI server + WebSocket + REST API
│   ├── models/database.py    # SQLAlchemy models
│   ├── services/
│   │   ├── inference.py      # Camera + ML inference engine
│   │   ├── threat_assessor.py# Threat state machine
│   │   ├── follower_tracker.py# Multi-day follower tracking
│   │   ├── notifier.py       # Push notifications
│   │   └── recorder.py       # Auto-recording
│   ├── iot/
│   │   ├── mqtt_broker.py    # MQTT communication layer
│   │   ├── device_manager.py # IoT device orchestration + sensor fusion
│   │   └── smart_home.py     # Home Assistant / IFTTT / Telegram bridge
│   ├── training/
│   │   ├── data_generator.py # Synthetic data generation
│   │   ├── models.py         # PyTorch model architectures
│   │   └── train.py          # Training pipeline
│   └── utils/
│       └── visualization.py  # Training plots
├── firmware/
│   ├── esp32_sensor_hub/     # Multi-sensor Arduino sketch
│   ├── esp32_panic_button/   # Panic button firmware
│   └── esp32_siren_controller/# Siren + LED + relay controller
├── static/index.html         # Backend-connected dashboard
└── models/weights/           # Trained model files
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
