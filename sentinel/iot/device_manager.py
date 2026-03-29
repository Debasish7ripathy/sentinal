"""
IoT Device Manager for SENTINEL.

Unified management layer that:
- Bridges MQTT sensor data with the AI threat engine
- Fuses camera AI detections with IoT sensor readings
- Manages zone-based security perimeters
- Coordinates physical alert responses
- Provides device health monitoring
"""
import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from sentinel.iot.mqtt_broker import MQTTManager, AlertCommand, SensorReading
from sentinel.iot.smart_home import SmartHomeBridge

logger = logging.getLogger("sentinel.iot.devices")


@dataclass
class SecurityZone:
    name: str
    armed: bool = True
    devices: list = field(default_factory=list)
    threat_level: str = "NORMAL"
    last_event: Optional[dict] = None
    motion_count_today: int = 0
    breach_count_today: int = 0


class DeviceManager:
    """
    Orchestrates all IoT devices and bridges sensor data
    with the SENTINEL AI threat assessment engine.
    """

    def __init__(self, mqtt_manager: MQTTManager, smart_home: SmartHomeBridge):
        self.mqtt = mqtt_manager
        self.smart_home = smart_home

        # Security zones
        self.zones: dict[str, SecurityZone] = {
            "front_door": SecurityZone(name="Front Door"),
            "back_door": SecurityZone(name="Back Door"),
            "perimeter": SecurityZone(name="Perimeter"),
            "interior": SecurityZone(name="Interior"),
            "garage": SecurityZone(name="Garage"),
        }

        # System state
        self.system_armed = True
        self.lockdown_active = False
        self.last_threat_broadcast = 0

        # Sensor fusion buffer: correlate IoT + camera events
        self.recent_iot_events: list[dict] = []
        self.recent_camera_events: list[dict] = []
        self.fused_events: list[dict] = []

        # Alert state
        self.active_alerts: dict[str, dict] = {}

        # Register MQTT callbacks
        self.mqtt.on_sensor_data(self._on_sensor_data)
        self.mqtt.on_device_status(self._on_device_status)
        self.mqtt.on_zone_breach(self._on_zone_breach)

    async def start(self):
        """Start the device manager and MQTT connection."""
        await self.mqtt.start()
        logger.info("Device manager started")

    # ============================================================
    # MQTT Callbacks
    # ============================================================
    async def _on_sensor_data(self, reading: SensorReading):
        """Handle incoming sensor readings."""
        self.recent_iot_events.append({
            "type": "sensor",
            "device_id": reading.device_id,
            "sensor_type": reading.sensor_type,
            "value": reading.value,
            "zone": reading.zone,
            "timestamp": reading.timestamp,
        })

        # Keep buffer manageable
        if len(self.recent_iot_events) > 200:
            self.recent_iot_events = self.recent_iot_events[-200:]

        # Update zone stats
        zone = self.zones.get(reading.zone)
        if zone:
            if reading.sensor_type == "pir" and reading.value == 1:
                zone.motion_count_today += 1

    async def _on_device_status(self, device):
        """Handle device status updates."""
        # Auto-assign to zone
        zone_name = device.zone
        if zone_name in self.zones:
            zone = self.zones[zone_name]
            if device.device_id not in zone.devices:
                zone.devices.append(device.device_id)

    async def _on_zone_breach(self, threat_event: dict):
        """Handle zone breach events from sensors."""
        zone_name = threat_event.get("zone", "default")
        zone = self.zones.get(zone_name)

        if zone and not zone.armed:
            logger.info(f"Zone {zone_name} is disarmed — ignoring event")
            return

        if zone:
            zone.threat_level = threat_event["level"]
            zone.last_event = threat_event
            zone.breach_count_today += 1

        # Check sensor fusion: if camera also detected threat, escalate
        fused = self._attempt_sensor_fusion(threat_event)

        # Dispatch to smart home
        await self.smart_home.dispatch_event(
            event_type=threat_event["type"],
            level=fused["level"] if fused else threat_event["level"],
            data=fused if fused else threat_event,
        )

        # Trigger physical alerts
        if threat_event["level"] == "CRITICAL" or (fused and fused["level"] == "CRITICAL"):
            await self._trigger_critical_response(threat_event)

        # Log
        self.fused_events.append({
            "timestamp": time.time(),
            "iot_event": threat_event,
            "fused": fused,
        })
        if len(self.fused_events) > 100:
            self.fused_events = self.fused_events[-100:]

    # ============================================================
    # Sensor Fusion
    # ============================================================
    def _attempt_sensor_fusion(self, iot_event: dict) -> Optional[dict]:
        """
        Correlate IoT sensor event with recent camera AI detections.

        If both camera AND IoT trigger within a window, escalate confidence.
        """
        now = time.time()
        window_sec = 10  # correlation window

        recent_camera = [
            e for e in self.recent_camera_events
            if now - e.get("timestamp", 0) < window_sec
        ]

        if not recent_camera:
            return None

        # Find matching zone or general correlation
        best_match = None
        for cam_event in recent_camera:
            cam_level = cam_event.get("threat_level", "NORMAL")
            iot_level = iot_event.get("level", "NORMAL")

            # Both systems detect a threat = high confidence fusion
            if cam_level in ("SUSPICIOUS", "CRITICAL") and iot_level in ("SUSPICIOUS", "CRITICAL"):
                best_match = {
                    "type": "FUSED_THREAT",
                    "level": "CRITICAL",  # Escalate if both agree
                    "confidence": min(100, cam_event.get("confidence", 50) + 30),
                    "camera_threat": cam_level,
                    "iot_threat": iot_level,
                    "iot_event_type": iot_event.get("type"),
                    "zone": iot_event.get("zone"),
                    "message": (
                        f"MULTI-SENSOR THREAT: Camera ({cam_level}) + "
                        f"IoT ({iot_event.get('type')}) in zone {iot_event.get('zone')}"
                    ),
                    "weapon_detected": cam_event.get("weapon_detected", False),
                    "timestamp": now,
                }
                break

        return best_match

    def report_camera_event(self, event: dict):
        """Called by the inference engine to report camera-based detections."""
        event["timestamp"] = time.time()
        self.recent_camera_events.append(event)
        if len(self.recent_camera_events) > 50:
            self.recent_camera_events = self.recent_camera_events[-50:]

    # ============================================================
    # Alert Responses
    # ============================================================
    async def _trigger_critical_response(self, event: dict):
        """Execute coordinated critical threat response."""
        if self.lockdown_active:
            return

        logger.warning(f"CRITICAL RESPONSE: {event.get('message', 'Unknown threat')}")

        # Activate all sirens
        await self.mqtt.trigger_alert(AlertCommand(
            device_id="all", alert_type="siren", action="on",
            duration_sec=60, intensity=100, threat_level="CRITICAL"
        ))

        # Flash all LEDs red
        await self.mqtt.trigger_alert(AlertCommand(
            device_id="all", alert_type="led", action="pattern",
            duration_sec=60, color="#ff0000", threat_level="CRITICAL",
            message="THREAT"
        ))

        # Lock all doors
        await self.mqtt.trigger_alert(AlertCommand(
            device_id="all", alert_type="lock", action="on",
            threat_level="CRITICAL"
        ))

        self.active_alerts["critical_response"] = {
            "started": time.time(),
            "event": event,
        }

    async def trigger_lockdown(self):
        """Full facility lockdown."""
        self.lockdown_active = True
        logger.warning("LOCKDOWN ACTIVATED")

        await self.mqtt.trigger_alert(AlertCommand(
            device_id="all", alert_type="lock", action="on",
            duration_sec=3600, threat_level="CRITICAL"
        ))
        await self.mqtt.trigger_alert(AlertCommand(
            device_id="all", alert_type="siren", action="pattern",
            duration_sec=300, intensity=100, threat_level="CRITICAL"
        ))

        await self.smart_home.dispatch_event("LOCKDOWN", "CRITICAL", {
            "message": "SENTINEL LOCKDOWN ACTIVATED",
            "timestamp": time.time(),
        })

    async def cancel_lockdown(self):
        """Cancel lockdown — restore normal operations."""
        self.lockdown_active = False
        logger.info("Lockdown cancelled")

        await self.mqtt.trigger_alert(AlertCommand(
            device_id="all", alert_type="siren", action="off",
            threat_level="NORMAL"
        ))
        await self.mqtt.trigger_alert(AlertCommand(
            device_id="all", alert_type="led", action="off",
            threat_level="NORMAL"
        ))
        await self.mqtt.trigger_alert(AlertCommand(
            device_id="all", alert_type="lock", action="off",
            threat_level="NORMAL"
        ))

        self.active_alerts.clear()

    # ============================================================
    # Zone Management
    # ============================================================
    def arm_zone(self, zone_name: str):
        if zone_name in self.zones:
            self.zones[zone_name].armed = True
            logger.info(f"Zone armed: {zone_name}")

    def disarm_zone(self, zone_name: str):
        if zone_name in self.zones:
            self.zones[zone_name].armed = False
            logger.info(f"Zone disarmed: {zone_name}")

    def arm_all(self):
        self.system_armed = True
        for z in self.zones.values():
            z.armed = True
        logger.info("All zones armed")

    def disarm_all(self):
        self.system_armed = False
        for z in self.zones.values():
            z.armed = False
        logger.info("All zones disarmed")

    def add_zone(self, name: str):
        if name not in self.zones:
            self.zones[name] = SecurityZone(name=name)

    # ============================================================
    # Status / API
    # ============================================================
    def get_status(self) -> dict:
        """Get full IoT system status."""
        devices = self.mqtt.get_all_devices()
        online = sum(1 for d in devices if d.get("online"))

        return {
            "system_armed": self.system_armed,
            "lockdown_active": self.lockdown_active,
            "mqtt_connected": self.mqtt.connected,
            "devices": {
                "total": len(devices),
                "online": online,
                "list": devices,
            },
            "zones": {
                name: {
                    "name": z.name,
                    "armed": z.armed,
                    "threat_level": z.threat_level,
                    "device_count": len(z.devices),
                    "motion_today": z.motion_count_today,
                    "breaches_today": z.breach_count_today,
                    "last_event": z.last_event,
                }
                for name, z in self.zones.items()
            },
            "active_alerts": self.active_alerts,
            "integrations": self.smart_home.get_integration_status(),
            "recent_fused_events": self.fused_events[-10:],
        }

    async def stop(self):
        await self.mqtt.stop()
        await self.smart_home.close()
        logger.info("Device manager stopped")
