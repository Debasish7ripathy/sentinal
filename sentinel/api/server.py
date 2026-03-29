"""
FastAPI server for SENTINEL.

Provides:
- WebSocket endpoint for real-time video frame streaming + detection results
- REST API for followers, incidents, recordings, settings
- Static file serving for the frontend
- ML model inference pipeline integration
"""
import asyncio
import base64
import json
import time
import datetime
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sentinel.config import settings
from sentinel.models.database import init_db, get_db, SessionLocal, Follower, Incident, Recording
from sentinel.services.inference import InferenceEngine
from sentinel.services.notifier import NotificationService
from sentinel.services.recorder import RecordingManager
from sentinel.iot.mqtt_broker import MQTTManager, AlertCommand
from sentinel.iot.smart_home import SmartHomeBridge, WebhookTarget
from sentinel.iot.device_manager import DeviceManager


# Global services
engine: InferenceEngine = None
notifier: NotificationService = None
recorder: RecordingManager = None
device_mgr: DeviceManager = None
incident_log: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global engine, notifier, recorder, device_mgr
    print("\n" + "=" * 50)
    print("  SENTINEL SERVER STARTING")
    print("=" * 50)

    init_db()
    notifier = NotificationService()
    recorder = RecordingManager()

    # Inference engine (initialized lazily on first WebSocket connection)
    engine = InferenceEngine()

    # IoT Device Manager
    mqtt_mgr = MQTTManager(
        broker_host=settings.MQTT_BROKER_HOST,
        broker_port=settings.MQTT_BROKER_PORT,
        username=settings.MQTT_USERNAME,
        password=settings.MQTT_PASSWORD,
    )
    smart_home = SmartHomeBridge()

    # Configure smart home integrations from settings
    if settings.HA_URL and settings.HA_TOKEN:
        smart_home.configure_home_assistant(settings.HA_URL, settings.HA_TOKEN)
    if settings.IFTTT_KEY:
        smart_home.configure_ifttt(settings.IFTTT_KEY)
    if settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
        smart_home.configure_telegram(settings.TELEGRAM_BOT_TOKEN, settings.TELEGRAM_CHAT_ID)

    device_mgr = DeviceManager(mqtt_mgr, smart_home)

    if settings.IOT_ENABLED:
        try:
            await device_mgr.start()
            print("[SENTINEL] IoT device manager started")
        except Exception as e:
            print(f"[SENTINEL] IoT startup warning: {e} (continuing without MQTT)")

    yield

    # Shutdown
    print("[SENTINEL] Shutting down...")
    if device_mgr:
        await device_mgr.stop()
    if engine:
        engine.release()
    if recorder:
        recorder.release()
    if notifier:
        await notifier.close()


app = FastAPI(
    title="SENTINEL API",
    version="2.1.0",
    description="Autonomous AI Personal Safety System",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = Path(settings.STATIC_DIR)
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Serve recordings
rec_dir = Path(settings.RECORDINGS_DIR)
rec_dir.mkdir(exist_ok=True)
app.mount("/recordings", StaticFiles(directory=str(rec_dir)), name="recordings")


# ============================================================
# Frontend
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the SENTINEL frontend."""
    frontend_path = Path(__file__).parent.parent.parent / "sentinel.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    # Fallback: serve the integrated version
    integrated_path = Path(__file__).parent.parent.parent / "static" / "index.html"
    if integrated_path.exists():
        return FileResponse(str(integrated_path))
    return HTMLResponse("<h1>SENTINEL - Frontend not found</h1>")


# ============================================================
# WebSocket: Real-time Video Stream + Detection
# ============================================================
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time bidirectional WebSocket:
    - Server → Client: JPEG frames + detection metadata (JSON)
    - Client → Server: settings updates, commands
    """
    await websocket.accept()
    global engine, notifier, recorder

    # Initialize inference engine on first connection
    if not engine.running:
        try:
            engine.initialize()
        except Exception as e:
            await websocket.send_json({"error": f"Camera init failed: {str(e)}"})
            await websocket.close()
            return

    print("[SENTINEL] WebSocket client connected")

    try:
        while True:
            # Process frame
            results = engine.process_frame()
            if results is None:
                await asyncio.sleep(0.01)
                continue

            # Get JPEG frame
            jpeg_bytes = engine.get_jpeg_frame(results)
            frame_b64 = base64.b64encode(jpeg_bytes).decode("utf-8") if jpeg_bytes else ""

            # Recording
            if results.get("annotated_frame") is not None:
                recorder.update(
                    results["annotated_frame"],
                    results["threat_level"],
                    results["weapon_detected"],
                )

            # Process follower alerts
            follower_data = results.get("follower", {})
            alerts = follower_data.get("alerts") if isinstance(follower_data, dict) else []
            if alerts:
                notifier.process_alerts(alerts)

            # Log incidents
            if results["threat_level"] in ("SUSPICIOUS", "CRITICAL"):
                _log_incident(results)

            # Send notifications for threat level changes
            if results["weapon_detected"]:
                await notifier.send(
                    "CRITICAL",
                    f"WEAPON DETECTED: {results['weapon_label']}. Confidence: {results['confidence']}%",
                    weapon_detected=True,
                    bypass_cooldown=True,
                )
            elif results["threat_level"] == "CRITICAL":
                await notifier.send(
                    "CRITICAL",
                    f"CRITICAL threat. Confidence: {results['confidence']}%",
                )
            elif results["threat_level"] == "SUSPICIOUS":
                await notifier.send(
                    "SUSPICIOUS",
                    f"Suspicious activity. Confidence: {results['confidence']}%",
                )

            # Build message
            message = {
                "type": "frame",
                "frame": frame_b64,
                "data": {
                    "fps": results["fps"],
                    "threat_level": results["threat_level"],
                    "confidence": results["confidence"],
                    "persons": len(results["persons"]),
                    "objects": [
                        {"label": o["label"], "confidence": o["confidence"], "level": o["level"]}
                        for o in results["objects"]
                    ],
                    "weapon_detected": results["weapon_detected"],
                    "weapon_label": results.get("weapon_label", ""),
                    "held_weapon": results["held_weapon"],
                    "anomaly_score": results.get("anomaly_score", 0),
                    "ml_probs": results.get("ml_threat_probs"),
                    "assessment": results.get("assessment_details", {}),
                    "recording": recorder.is_recording,
                    "follower": follower_data if isinstance(follower_data, dict) else {},
                },
            }

            await websocket.send_json(message)

            # Check for client messages (non-blocking)
            try:
                client_msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                await _handle_client_message(json.loads(client_msg))
            except (asyncio.TimeoutError, json.JSONDecodeError):
                pass

            # Frame rate control
            await asyncio.sleep(1 / settings.TARGET_FPS)

    except WebSocketDisconnect:
        print("[SENTINEL] WebSocket client disconnected")
    except Exception as e:
        print(f"[SENTINEL] WebSocket error: {e}")


async def _handle_client_message(msg: dict):
    """Handle commands from the frontend client."""
    cmd = msg.get("command")

    if cmd == "set_night_vision":
        engine.night_vision = msg.get("value", True)
    elif cmd == "set_sensitivity":
        engine.threat_assessor.set_sensitivity(msg.get("value", "medium"))
    elif cmd == "set_ntfy_topic":
        settings.NTFY_TOPIC = msg.get("value", "")
    elif cmd == "test_ntfy":
        await notifier.send_test(msg.get("value"))
    elif cmd == "mark_safe":
        engine.follower_tracker.mark_safe(msg.get("follower_id", ""))


def _log_incident(results: dict):
    global incident_log
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "level": results["threat_level"],
        "confidence": results["confidence"],
        "objects": [o["label"] for o in results["objects"]],
        "weapon": results["weapon_detected"],
    }
    incident_log.append(entry)
    if len(incident_log) > 500:
        incident_log = incident_log[-500:]

    # Save to DB
    db = SessionLocal()
    try:
        inc = Incident(
            level=results["threat_level"],
            message=results.get("assessment_details", {}).get("details", ""),
            confidence=results["confidence"],
            objects_detected=[o["label"] for o in results["objects"]],
        )
        db.add(inc)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


# ============================================================
# REST API: Followers
# ============================================================
@app.get("/api/followers")
async def get_followers():
    """Get all tracked followers, sorted by risk score."""
    db = SessionLocal()
    try:
        followers = db.query(Follower).filter(
            Follower.marked_safe == False
        ).order_by(Follower.risk_score.desc()).all()
        return [f.to_dict() for f in followers]
    finally:
        db.close()


@app.get("/api/followers/{follower_id}")
async def get_follower(follower_id: str):
    db = SessionLocal()
    try:
        f = db.query(Follower).get(follower_id)
        if not f:
            raise HTTPException(404, "Follower not found")
        return f.to_dict()
    finally:
        db.close()


@app.post("/api/followers/{follower_id}/mark-safe")
async def mark_follower_safe(follower_id: str):
    db = SessionLocal()
    try:
        f = db.query(Follower).get(follower_id)
        if not f:
            raise HTTPException(404, "Follower not found")
        f.marked_safe = True
        db.commit()
        return {"status": "ok", "id": follower_id}
    finally:
        db.close()


@app.delete("/api/followers")
async def clear_followers():
    db = SessionLocal()
    try:
        db.query(Follower).delete()
        db.commit()
        return {"status": "ok", "message": "All followers cleared"}
    finally:
        db.close()


# ============================================================
# REST API: Incidents
# ============================================================
@app.get("/api/incidents")
async def get_incidents(limit: int = 100, level: str = None):
    db = SessionLocal()
    try:
        query = db.query(Incident).order_by(Incident.timestamp.desc())
        if level:
            query = query.filter(Incident.level == level)
        incidents = query.limit(limit).all()
        return [i.to_dict() for i in incidents]
    finally:
        db.close()


@app.get("/api/incidents/export")
async def export_incidents():
    global incident_log
    return JSONResponse(content=incident_log)


# ============================================================
# REST API: Recordings
# ============================================================
@app.get("/api/recordings")
async def get_recordings():
    if recorder:
        return recorder.get_recordings()
    return []


@app.get("/api/recordings/{filename}")
async def download_recording(filename: str):
    filepath = Path(settings.RECORDINGS_DIR) / filename
    if not filepath.exists():
        raise HTTPException(404, "Recording not found")
    return FileResponse(str(filepath), media_type="video/x-msvideo", filename=filename)


# ============================================================
# REST API: Notifications
# ============================================================
@app.get("/api/notifications")
async def get_notification_log():
    if notifier:
        return notifier.notification_log
    return []


@app.post("/api/notifications/test")
async def test_notification(topic: str = None):
    if notifier:
        success = await notifier.send_test(topic)
        return {"sent": success}
    return {"sent": False}


# ============================================================
# REST API: Settings
# ============================================================
@app.get("/api/settings")
async def get_settings():
    return {
        "sensitivity": settings.SENSITIVITY,
        "ntfyTopic": settings.NTFY_TOPIC,
        "nightVision": engine.night_vision if engine else True,
        "weaponDetection": settings.YOLO_CONFIDENCE_THRESHOLD > 0,
        "followerMemoryDays": settings.FOLLOWER_MEMORY_DAYS,
        "cameraIndex": settings.CAMERA_INDEX,
    }


@app.post("/api/settings")
async def update_settings(data: dict):
    if "sensitivity" in data:
        settings.SENSITIVITY = data["sensitivity"]
        if engine:
            engine.threat_assessor.set_sensitivity(data["sensitivity"])
    if "ntfyTopic" in data:
        settings.NTFY_TOPIC = data["ntfyTopic"]
    if "nightVision" in data and engine:
        engine.night_vision = data["nightVision"]
    if "followerMemoryDays" in data:
        settings.FOLLOWER_MEMORY_DAYS = int(data["followerMemoryDays"])
    return {"status": "ok"}


# ============================================================
# REST API: System Status
# ============================================================
@app.get("/api/status")
async def get_system_status():
    return {
        "running": engine.running if engine else False,
        "fps": engine.fps if engine else 0,
        "currentThreat": engine.current_threat if engine else "NORMAL",
        "confidence": engine.confidence if engine else 0,
        "recording": recorder.is_recording if recorder else False,
        "followerCount": len(engine.follower_tracker.get_all_followers()) if engine else 0,
        "modelsLoaded": {
            "pose": engine.pose_detector is not None if engine else False,
            "yolo": engine.yolo_model is not None if engine else False,
            "threatMLP": engine.threat_model is not None if engine else False,
            "anomaly": engine.anomaly_model is not None if engine else False,
            "reid": engine.reid_model is not None if engine else False,
            "weaponContext": engine.weapon_model is not None if engine else False,
        },
    }


# ============================================================
# REST API: Training trigger
# ============================================================
@app.post("/api/train/{model_name}")
async def trigger_training(model_name: str):
    """Trigger model training (runs in background)."""
    valid = ["all", "threat_mlp", "lstm", "anomaly", "reid", "weapon"]
    if model_name not in valid:
        raise HTTPException(400, f"Invalid model. Choose from: {valid}")

    # Run training in background thread
    import threading
    from sentinel.training.train import (
        train_all, train_threat_mlp, train_threat_lstm,
        train_anomaly_autoencoder, train_reid_net, train_weapon_context
    )

    fn_map = {
        "all": train_all,
        "threat_mlp": train_threat_mlp,
        "lstm": train_threat_lstm,
        "anomaly": train_anomaly_autoencoder,
        "reid": train_reid_net,
        "weapon": train_weapon_context,
    }

    thread = threading.Thread(target=fn_map[model_name], daemon=True)
    thread.start()

    return {"status": "training_started", "model": model_name}


# ============================================================
# REST API: IoT Devices
# ============================================================
@app.get("/api/iot/status")
async def get_iot_status():
    """Get full IoT system status: devices, zones, integrations."""
    if device_mgr:
        return device_mgr.get_status()
    return {"error": "IoT not initialized"}


@app.get("/api/iot/devices")
async def get_iot_devices():
    """List all registered IoT devices."""
    if device_mgr:
        return device_mgr.mqtt.get_all_devices()
    return []


@app.get("/api/iot/sensors/{device_id}/{sensor_type}")
async def get_sensor_history(device_id: str, sensor_type: str, limit: int = 50):
    """Get recent sensor readings for a device."""
    if device_mgr:
        return device_mgr.mqtt.get_sensor_history(device_id, sensor_type, limit)
    return []


@app.post("/api/iot/alert")
async def send_iot_alert(data: dict):
    """Manually trigger an IoT alert."""
    if not device_mgr:
        raise HTTPException(503, "IoT not initialized")

    alert = AlertCommand(
        device_id=data.get("device_id", "all"),
        alert_type=data.get("alert_type", "siren"),
        action=data.get("action", "on"),
        duration_sec=data.get("duration", 10),
        intensity=data.get("intensity", 100),
        color=data.get("color", "#ff0000"),
        message=data.get("message", ""),
        threat_level=data.get("threat_level", "CRITICAL"),
    )
    success = await device_mgr.mqtt.trigger_alert(alert)
    return {"sent": success}


@app.post("/api/iot/zones/{zone_name}/arm")
async def arm_zone(zone_name: str):
    if device_mgr:
        device_mgr.arm_zone(zone_name)
        return {"status": "armed", "zone": zone_name}
    raise HTTPException(503, "IoT not initialized")


@app.post("/api/iot/zones/{zone_name}/disarm")
async def disarm_zone(zone_name: str):
    if device_mgr:
        device_mgr.disarm_zone(zone_name)
        return {"status": "disarmed", "zone": zone_name}
    raise HTTPException(503, "IoT not initialized")


@app.post("/api/iot/lockdown")
async def activate_lockdown():
    if device_mgr:
        await device_mgr.trigger_lockdown()
        return {"status": "lockdown_activated"}
    raise HTTPException(503, "IoT not initialized")


@app.post("/api/iot/lockdown/cancel")
async def cancel_lockdown():
    if device_mgr:
        await device_mgr.cancel_lockdown()
        return {"status": "lockdown_cancelled"}
    raise HTTPException(503, "IoT not initialized")


@app.post("/api/iot/arm-all")
async def arm_all_zones():
    if device_mgr:
        device_mgr.arm_all()
        return {"status": "all_armed"}
    raise HTTPException(503, "IoT not initialized")


@app.post("/api/iot/disarm-all")
async def disarm_all_zones():
    if device_mgr:
        device_mgr.disarm_all()
        return {"status": "all_disarmed"}
    raise HTTPException(503, "IoT not initialized")


@app.get("/api/iot/integrations")
async def get_integrations():
    """Get smart home integration status."""
    if device_mgr:
        return device_mgr.smart_home.get_integration_status()
    return {}


@app.post("/api/iot/integrations/webhook")
async def add_webhook(data: dict):
    """Add a custom webhook integration."""
    if not device_mgr:
        raise HTTPException(503, "IoT not initialized")

    wh = WebhookTarget(
        name=data.get("name", "Custom Webhook"),
        url=data["url"],
        method=data.get("method", "POST"),
        auth_token=data.get("auth_token", ""),
        trigger_on=data.get("trigger_on", ["CRITICAL", "WEAPON", "PANIC"]),
        cooldown_sec=data.get("cooldown_sec", 30),
    )
    device_mgr.smart_home.add_webhook(wh)
    return {"status": "webhook_added", "name": wh.name}


@app.post("/api/iot/device/{device_id}/config")
async def push_device_config(device_id: str, config: dict):
    """Push configuration to an IoT device."""
    if device_mgr:
        success = await device_mgr.mqtt.send_device_config(device_id, config)
        return {"sent": success}
    raise HTTPException(503, "IoT not initialized")
