"""Central configuration for SENTINEL system."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8099
    DEBUG: bool = False

    # Paths
    DB_PATH: str = str(BASE_DIR / "db" / "sentinel.db")
    MODELS_DIR: str = str(BASE_DIR / "models" / "weights")
    RECORDINGS_DIR: str = str(BASE_DIR / "recordings")
    STATIC_DIR: str = str(BASE_DIR / "static")
    TRAINING_DATA_DIR: str = str(BASE_DIR / "sentinel" / "training" / "data")

    # Camera
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 1280
    CAMERA_HEIGHT: int = 720
    TARGET_FPS: int = 30

    # ML Models
    POSE_MODEL_COMPLEXITY: int = 1
    POSE_MIN_DETECTION_CONFIDENCE: float = 0.6
    POSE_MIN_TRACKING_CONFIDENCE: float = 0.5
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.4
    THREAT_MODEL_PATH: str = ""
    REID_MODEL_PATH: str = ""

    # Threat Assessment
    SENSITIVITY: str = "medium"  # low, medium, high
    LOITER_THRESHOLD_SEC: float = 8.0
    RAPID_APPROACH_GROWTH: float = 0.4
    RAPID_APPROACH_TIME: float = 2.0
    CRITICAL_COVERAGE: float = 0.5
    AGGRESSIVE_POSTURE_SEC: float = 1.5

    # Weights for confidence scoring
    WEIGHT_PROXIMITY: float = 0.20
    WEIGHT_VELOCITY: float = 0.20
    WEIGHT_POSTURE: float = 0.20
    WEIGHT_WEAPON: float = 0.30
    WEIGHT_FOLLOWER: float = 0.10

    # Notifications
    NTFY_TOPIC: str = ""
    NTFY_BASE_URL: str = "https://ntfy.sh"
    COOLDOWN_CRITICAL_SEC: int = 15
    COOLDOWN_SUSPICIOUS_SEC: int = 30

    # Recording
    MAX_RECORDINGS: int = 20
    RECORDING_DURATION_SEC: int = 10
    RECORDING_CODEC: str = "XVID"

    # Follower Tracking
    FOLLOWER_MEMORY_DAYS: int = 14
    FINGERPRINT_MATCH_THRESHOLD: float = 0.70
    FOLLOWER_REPEAT_THRESHOLD: int = 2
    FOLLOWER_POSSIBLE_DAYS: int = 2
    FOLLOWER_CONFIRMED_DAYS: int = 3

    # IoT / MQTT
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    MQTT_USERNAME: str = ""
    MQTT_PASSWORD: str = ""
    IOT_ENABLED: bool = True

    # Smart Home Integrations
    HA_URL: str = ""  # Home Assistant URL
    HA_TOKEN: str = ""  # Home Assistant long-lived access token
    IFTTT_KEY: str = ""  # IFTTT Webhooks key
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Training
    TRAIN_BATCH_SIZE: int = 32
    TRAIN_EPOCHS: int = 50
    TRAIN_LR: float = 0.001
    TRAIN_VAL_SPLIT: float = 0.2

    class Config:
        env_prefix = "SENTINEL_"
        env_file = str(BASE_DIR / ".env")


settings = Settings()

# Ensure directories exist
for d in [settings.RECORDINGS_DIR, settings.MODELS_DIR,
          os.path.dirname(settings.DB_PATH), settings.TRAINING_DATA_DIR]:
    os.makedirs(d, exist_ok=True)
