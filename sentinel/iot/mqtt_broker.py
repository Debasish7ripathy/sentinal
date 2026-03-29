"""
SENTINEL MQTT Integration Layer.

Manages bidirectional MQTT communication between:
- ESP32/Arduino sensor nodes (PIR, ultrasonic, reed switch, panic button)
- Physical alert devices (siren, LED strip, door lock relay)
- Smart home platforms (Home Assistant, IFTTT)
- The SENTINEL AI backend (threat fusion engine)

Topic Schema:
    sentinel/sensors/<device_id>/<sensor_type>    → Incoming sensor data
    sentinel/alerts/<device_id>/trigger            → Outgoing alert commands
    sentinel/devices/<device_id>/status            → Device heartbeat / status
    sentinel/system/threat                         → Current threat level broadcast
    sentinel/system/command                        → System-wide commands
"""
import asyncio
import json
import time
import logging
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict

logger = logging.getLogger("sentinel.iot")


@dataclass
class SensorReading:
    device_id: str
    sensor_type: str  # pir, ultrasonic, reed, panic, temperature, light
    value: float
    unit: str = ""
    zone: str = "default"
    timestamp: float = field(default_factory=time.time)
    raw_payload: dict = field(default_factory=dict)


@dataclass
class DeviceInfo:
    device_id: str
    device_type: str  # sensor_hub, panic_button, siren_controller, camera_node
    firmware_version: str = ""
    zone: str = "default"
    ip_address: str = ""
    last_heartbeat: float = 0
    online: bool = False
    capabilities: list = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class AlertCommand:
    device_id: str  # target device or "all"
    alert_type: str  # siren, led, lock, strobe, voice
    action: str  # on, off, pulse, pattern
    duration_sec: int = 10
    intensity: int = 100  # 0-100
    color: str = "#ff0000"
    message: str = ""
    threat_level: str = "CRITICAL"


class MQTTManager:
    """
    Central MQTT manager for all IoT communications.

    Handles connection, topic subscriptions, message routing,
    device registry, and alert dispatch.
    """

    TOPIC_SENSOR_DATA = "sentinel/sensors/{device_id}/{sensor_type}"
    TOPIC_DEVICE_STATUS = "sentinel/devices/{device_id}/status"
    TOPIC_ALERT_TRIGGER = "sentinel/alerts/{device_id}/trigger"
    TOPIC_SYSTEM_THREAT = "sentinel/system/threat"
    TOPIC_SYSTEM_COMMAND = "sentinel/system/command"
    TOPIC_DEVICE_CONFIG = "sentinel/devices/{device_id}/config"

    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883,
                 username: str = "", password: str = ""):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password

        # Device registry
        self.devices: dict[str, DeviceInfo] = {}

        # Sensor data buffer (last N readings per device/sensor)
        self.sensor_buffer: dict[str, list[SensorReading]] = defaultdict(lambda: [])
        self.buffer_max = 100

        # Callbacks
        self._on_sensor_data: list[Callable] = []
        self._on_device_status: list[Callable] = []
        self._on_alert_event: list[Callable] = []
        self._on_zone_breach: list[Callable] = []

        # State
        self.connected = False
        self.client = None
        self._task: Optional[asyncio.Task] = None

        # Alert cooldowns per device
        self._alert_cooldowns: dict[str, float] = {}

        # Zone threat aggregation
        self.zone_threat_levels: dict[str, str] = {}

    async def start(self):
        """Start MQTT connection and subscription loop."""
        try:
            import aiomqtt
        except ImportError:
            logger.warning("aiomqtt not installed — IoT features disabled. Install with: pip install aiomqtt")
            return

        self._task = asyncio.create_task(self._connection_loop())
        logger.info(f"MQTT manager started — broker: {self.broker_host}:{self.broker_port}")

    async def _connection_loop(self):
        """Maintain persistent MQTT connection with auto-reconnect."""
        import aiomqtt

        while True:
            try:
                async with aiomqtt.Client(
                    hostname=self.broker_host,
                    port=self.broker_port,
                    username=self.username if self.username else None,
                    password=self.password if self.password else None,
                ) as client:
                    self.client = client
                    self.connected = True
                    logger.info("MQTT connected")

                    # Subscribe to all sentinel topics
                    await client.subscribe("sentinel/sensors/#")
                    await client.subscribe("sentinel/devices/#")
                    await client.subscribe("sentinel/system/#")

                    async for message in client.messages:
                        await self._handle_message(
                            str(message.topic),
                            message.payload.decode("utf-8", errors="replace")
                        )

            except Exception as e:
                self.connected = False
                self.client = None
                logger.error(f"MQTT connection error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _handle_message(self, topic: str, payload: str):
        """Route incoming MQTT messages to appropriate handlers."""
        parts = topic.split("/")

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {"raw": payload}

        # sentinel/sensors/<device_id>/<sensor_type>
        if len(parts) >= 4 and parts[1] == "sensors":
            device_id = parts[2]
            sensor_type = parts[3]
            await self._handle_sensor_data(device_id, sensor_type, data)

        # sentinel/devices/<device_id>/status
        elif len(parts) >= 4 and parts[1] == "devices" and parts[3] == "status":
            device_id = parts[2]
            await self._handle_device_status(device_id, data)

        # sentinel/system/command
        elif len(parts) >= 3 and parts[1] == "system" and parts[2] == "command":
            await self._handle_system_command(data)

    async def _handle_sensor_data(self, device_id: str, sensor_type: str, data: dict):
        """Process incoming sensor reading."""
        reading = SensorReading(
            device_id=device_id,
            sensor_type=sensor_type,
            value=float(data.get("value", 0)),
            unit=data.get("unit", ""),
            zone=data.get("zone", self.devices.get(device_id, DeviceInfo(device_id, "unknown")).zone),
            raw_payload=data,
        )

        # Buffer
        key = f"{device_id}/{sensor_type}"
        self.sensor_buffer[key].append(reading)
        if len(self.sensor_buffer[key]) > self.buffer_max:
            self.sensor_buffer[key] = self.sensor_buffer[key][-self.buffer_max:]

        # Check for immediate threat triggers
        threat_event = self._evaluate_sensor_threat(reading)
        if threat_event:
            for cb in self._on_zone_breach:
                await cb(threat_event)

        # Notify subscribers
        for cb in self._on_sensor_data:
            await cb(reading)

    async def _handle_device_status(self, device_id: str, data: dict):
        """Process device heartbeat / status update."""
        if device_id not in self.devices:
            self.devices[device_id] = DeviceInfo(
                device_id=device_id,
                device_type=data.get("type", "unknown"),
                firmware_version=data.get("firmware", ""),
                zone=data.get("zone", "default"),
                ip_address=data.get("ip", ""),
                capabilities=data.get("capabilities", []),
            )

        device = self.devices[device_id]
        device.last_heartbeat = time.time()
        device.online = True
        device.firmware_version = data.get("firmware", device.firmware_version)
        device.ip_address = data.get("ip", device.ip_address)

        if "config" in data:
            device.config.update(data["config"])

        for cb in self._on_device_status:
            await cb(device)

    async def _handle_system_command(self, data: dict):
        """Handle system-wide commands."""
        cmd = data.get("command")
        if cmd == "arm":
            logger.info("System ARMED via MQTT")
        elif cmd == "disarm":
            logger.info("System DISARMED via MQTT")
        elif cmd == "test_alerts":
            await self.trigger_alert(AlertCommand(
                device_id="all", alert_type="led", action="pulse",
                duration_sec=3, color="#00ff41", threat_level="NORMAL"
            ))

    def _evaluate_sensor_threat(self, reading: SensorReading) -> Optional[dict]:
        """Evaluate if a sensor reading constitutes an immediate threat."""
        threat = None

        if reading.sensor_type == "pir" and reading.value == 1:
            threat = {
                "type": "MOTION_DETECTED",
                "level": "SUSPICIOUS",
                "zone": reading.zone,
                "device_id": reading.device_id,
                "message": f"Motion detected in zone: {reading.zone}",
            }

        elif reading.sensor_type == "ultrasonic":
            distance_cm = reading.value
            if distance_cm < 50:
                threat = {
                    "type": "PROXIMITY_BREACH",
                    "level": "CRITICAL",
                    "zone": reading.zone,
                    "device_id": reading.device_id,
                    "message": f"Proximity breach in {reading.zone}: {distance_cm:.0f}cm",
                    "distance_cm": distance_cm,
                }
            elif distance_cm < 150:
                threat = {
                    "type": "PROXIMITY_WARNING",
                    "level": "SUSPICIOUS",
                    "zone": reading.zone,
                    "device_id": reading.device_id,
                    "message": f"Close approach in {reading.zone}: {distance_cm:.0f}cm",
                    "distance_cm": distance_cm,
                }

        elif reading.sensor_type == "reed" and reading.value == 1:
            threat = {
                "type": "DOOR_WINDOW_OPENED",
                "level": "CRITICAL",
                "zone": reading.zone,
                "device_id": reading.device_id,
                "message": f"Door/window opened in zone: {reading.zone}",
            }

        elif reading.sensor_type == "panic" and reading.value == 1:
            threat = {
                "type": "PANIC_BUTTON",
                "level": "CRITICAL",
                "zone": reading.zone,
                "device_id": reading.device_id,
                "message": f"PANIC BUTTON activated in zone: {reading.zone}",
                "bypass_cooldown": True,
            }

        elif reading.sensor_type == "vibration" and reading.value > 500:
            threat = {
                "type": "FORCED_ENTRY",
                "level": "CRITICAL",
                "zone": reading.zone,
                "device_id": reading.device_id,
                "message": f"Forced entry vibration in {reading.zone}: {reading.value}",
            }

        elif reading.sensor_type == "sound" and reading.value > 85:
            threat = {
                "type": "LOUD_SOUND",
                "level": "SUSPICIOUS",
                "zone": reading.zone,
                "device_id": reading.device_id,
                "message": f"Loud sound detected in {reading.zone}: {reading.value:.0f}dB",
            }

        return threat

    # ============================================================
    # Outgoing: Alert Commands
    # ============================================================
    async def trigger_alert(self, alert: AlertCommand):
        """Send alert command to IoT device(s)."""
        if not self.connected or not self.client:
            logger.warning("MQTT not connected — alert not sent")
            return False

        payload = json.dumps({
            "alert_type": alert.alert_type,
            "action": alert.action,
            "duration": alert.duration_sec,
            "intensity": alert.intensity,
            "color": alert.color,
            "message": alert.message,
            "threat_level": alert.threat_level,
            "timestamp": time.time(),
        })

        if alert.device_id == "all":
            # Broadcast to all alert-capable devices
            for dev_id, dev in self.devices.items():
                if alert.alert_type in dev.capabilities or "alert" in dev.capabilities:
                    topic = self.TOPIC_ALERT_TRIGGER.format(device_id=dev_id)
                    await self.client.publish(topic, payload)
        else:
            topic = self.TOPIC_ALERT_TRIGGER.format(device_id=alert.device_id)
            await self.client.publish(topic, payload)

        for cb in self._on_alert_event:
            await cb(alert)

        logger.info(f"Alert sent: {alert.alert_type}/{alert.action} → {alert.device_id}")
        return True

    async def broadcast_threat_level(self, level: str, confidence: int, details: str = ""):
        """Broadcast current threat level to all IoT devices."""
        if not self.connected or not self.client:
            return

        payload = json.dumps({
            "level": level,
            "confidence": confidence,
            "details": details,
            "timestamp": time.time(),
        })
        await self.client.publish(self.TOPIC_SYSTEM_THREAT, payload)

        # Auto-trigger alerts based on threat level
        if level == "CRITICAL":
            await self.trigger_alert(AlertCommand(
                device_id="all", alert_type="siren", action="on",
                duration_sec=30, intensity=100, threat_level="CRITICAL"
            ))
            await self.trigger_alert(AlertCommand(
                device_id="all", alert_type="led", action="pattern",
                duration_sec=30, color="#ff0000", threat_level="CRITICAL"
            ))
        elif level == "SUSPICIOUS":
            await self.trigger_alert(AlertCommand(
                device_id="all", alert_type="led", action="pulse",
                duration_sec=10, color="#ffcc00", threat_level="SUSPICIOUS"
            ))

    async def send_device_config(self, device_id: str, config: dict):
        """Push configuration update to a device."""
        if not self.connected or not self.client:
            return False

        topic = self.TOPIC_DEVICE_CONFIG.format(device_id=device_id)
        await self.client.publish(topic, json.dumps(config))
        return True

    # ============================================================
    # Callback Registration
    # ============================================================
    def on_sensor_data(self, callback: Callable):
        self._on_sensor_data.append(callback)

    def on_device_status(self, callback: Callable):
        self._on_device_status.append(callback)

    def on_alert_event(self, callback: Callable):
        self._on_alert_event.append(callback)

    def on_zone_breach(self, callback: Callable):
        self._on_zone_breach.append(callback)

    # ============================================================
    # Queries
    # ============================================================
    def get_all_devices(self) -> list[dict]:
        """Get all registered devices with online status."""
        now = time.time()
        result = []
        for dev in self.devices.values():
            dev.online = (now - dev.last_heartbeat) < 60
            result.append(dev.to_dict())
        return result

    def get_sensor_history(self, device_id: str, sensor_type: str, limit: int = 50) -> list[dict]:
        key = f"{device_id}/{sensor_type}"
        readings = self.sensor_buffer.get(key, [])[-limit:]
        return [
            {
                "value": r.value,
                "unit": r.unit,
                "zone": r.zone,
                "timestamp": r.timestamp,
            }
            for r in readings
        ]

    def get_zone_status(self) -> dict:
        """Get aggregated status per zone."""
        zones = defaultdict(lambda: {"devices": [], "online": 0, "total": 0, "last_event": None})
        for dev in self.devices.values():
            z = zones[dev.zone]
            z["devices"].append(dev.device_id)
            z["total"] += 1
            if dev.online:
                z["online"] += 1
        return dict(zones)

    async def stop(self):
        if self._task:
            self._task.cancel()
            self.connected = False
            logger.info("MQTT manager stopped")
