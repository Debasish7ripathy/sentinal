"""
Smart Home Integration for SENTINEL.

Bridges SENTINEL threat events to popular smart home platforms:
- Home Assistant (REST API + webhooks)
- IFTTT (Webhooks service)
- Custom webhook endpoints
- Telegram Bot API (optional)
"""
import asyncio
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger("sentinel.iot.smarthome")


@dataclass
class WebhookTarget:
    name: str
    url: str
    method: str = "POST"
    headers: dict = None
    auth_token: str = ""
    enabled: bool = True
    trigger_on: list = None  # ["CRITICAL", "SUSPICIOUS", "WEAPON", "FOLLOWER", "PANIC"]
    cooldown_sec: int = 30
    last_triggered: float = 0

    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Type": "application/json"}
        if self.trigger_on is None:
            self.trigger_on = ["CRITICAL", "WEAPON", "PANIC"]


class SmartHomeBridge:
    """Manages integrations with external smart home platforms."""

    def __init__(self):
        self.webhooks: list[WebhookTarget] = []
        self.ha_url: str = ""
        self.ha_token: str = ""
        self.ifttt_key: str = ""
        self.telegram_bot_token: str = ""
        self.telegram_chat_id: str = ""
        self.client = httpx.AsyncClient(timeout=10.0)
        self.event_log: list[dict] = []

    def configure_home_assistant(self, url: str, token: str):
        """Configure Home Assistant connection."""
        self.ha_url = url.rstrip("/")
        self.ha_token = token
        logger.info(f"Home Assistant configured: {self.ha_url}")

    def configure_ifttt(self, key: str):
        """Configure IFTTT Webhooks integration."""
        self.ifttt_key = key
        logger.info("IFTTT configured")

    def configure_telegram(self, bot_token: str, chat_id: str):
        """Configure Telegram bot notifications."""
        self.telegram_bot_token = bot_token
        self.telegram_chat_id = chat_id
        logger.info("Telegram configured")

    def add_webhook(self, webhook: WebhookTarget):
        """Add a custom webhook target."""
        self.webhooks.append(webhook)
        logger.info(f"Webhook added: {webhook.name} → {webhook.url}")

    # ============================================================
    # Event Dispatch
    # ============================================================
    async def dispatch_event(self, event_type: str, level: str, data: dict):
        """
        Dispatch a SENTINEL event to all configured integrations.

        event_type: THREAT, WEAPON, FOLLOWER, PANIC, ZONE_BREACH, RECORDING
        level: NORMAL, SUSPICIOUS, CRITICAL
        data: event details dict
        """
        self._log_event(event_type, level, data)

        tasks = []

        # Home Assistant
        if self.ha_url and self.ha_token:
            tasks.append(self._notify_home_assistant(event_type, level, data))

        # IFTTT
        if self.ifttt_key:
            tasks.append(self._notify_ifttt(event_type, level, data))

        # Telegram
        if self.telegram_bot_token and self.telegram_chat_id:
            tasks.append(self._notify_telegram(event_type, level, data))

        # Custom webhooks
        for wh in self.webhooks:
            if wh.enabled and level in wh.trigger_on:
                now = time.time()
                if now - wh.last_triggered > wh.cooldown_sec:
                    tasks.append(self._fire_webhook(wh, event_type, level, data))
                    wh.last_triggered = now

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ============================================================
    # Home Assistant
    # ============================================================
    async def _notify_home_assistant(self, event_type: str, level: str, data: dict):
        """Fire a Home Assistant event and optionally call services."""
        try:
            headers = {
                "Authorization": f"Bearer {self.ha_token}",
                "Content-Type": "application/json",
            }

            # Fire event
            await self.client.post(
                f"{self.ha_url}/api/events/sentinel_{event_type.lower()}",
                headers=headers,
                json={
                    "level": level,
                    "event_type": event_type,
                    **data,
                },
            )

            # Update sensor entity
            await self.client.post(
                f"{self.ha_url}/api/states/sensor.sentinel_threat_level",
                headers=headers,
                json={
                    "state": level,
                    "attributes": {
                        "friendly_name": "SENTINEL Threat Level",
                        "event_type": event_type,
                        "confidence": data.get("confidence", 0),
                        "weapon_detected": data.get("weapon_detected", False),
                        "icon": "mdi:shield-alert" if level == "CRITICAL" else "mdi:shield-check",
                    },
                },
            )

            # Trigger automations based on level
            if level == "CRITICAL":
                # Turn on all lights
                await self.client.post(
                    f"{self.ha_url}/api/services/light/turn_on",
                    headers=headers,
                    json={"entity_id": "all", "brightness": 255, "color_name": "red"},
                )
                # Lock all doors
                await self.client.post(
                    f"{self.ha_url}/api/services/lock/lock",
                    headers=headers,
                    json={"entity_id": "all"},
                )

            logger.info(f"Home Assistant notified: {event_type}/{level}")

        except Exception as e:
            logger.error(f"Home Assistant error: {e}")

    # ============================================================
    # IFTTT
    # ============================================================
    async def _notify_ifttt(self, event_type: str, level: str, data: dict):
        """Trigger IFTTT Webhooks applet."""
        try:
            event_name = f"sentinel_{event_type.lower()}"
            url = f"https://maker.ifttt.com/trigger/{event_name}/with/key/{self.ifttt_key}"

            await self.client.post(url, json={
                "value1": level,
                "value2": data.get("message", event_type),
                "value3": json.dumps({
                    "confidence": data.get("confidence", 0),
                    "weapon": data.get("weapon_detected", False),
                    "zone": data.get("zone", ""),
                }),
            })

            logger.info(f"IFTTT triggered: {event_name}")

        except Exception as e:
            logger.error(f"IFTTT error: {e}")

    # ============================================================
    # Telegram
    # ============================================================
    async def _notify_telegram(self, event_type: str, level: str, data: dict):
        """Send alert via Telegram Bot."""
        try:
            icons = {
                "CRITICAL": "\U0001f6a8",
                "SUSPICIOUS": "\u26a0\ufe0f",
                "NORMAL": "\u2705",
            }
            icon = icons.get(level, "\U0001f514")

            message = data.get("message", f"{event_type}: {level}")
            text = (
                f"{icon} *SENTINEL ALERT*\n\n"
                f"*Level:* {level}\n"
                f"*Event:* {event_type}\n"
                f"*Details:* {message}\n"
            )

            if data.get("weapon_detected"):
                text += f"*Weapon:* {data.get('weapon_label', 'Unknown')}\n"
            if data.get("zone"):
                text += f"*Zone:* {data['zone']}\n"
            if data.get("confidence"):
                text += f"*Confidence:* {data['confidence']}%\n"

            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            await self.client.post(url, json={
                "chat_id": self.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown",
            })

            logger.info(f"Telegram notified: {event_type}/{level}")

        except Exception as e:
            logger.error(f"Telegram error: {e}")

    # ============================================================
    # Custom Webhooks
    # ============================================================
    async def _fire_webhook(self, wh: WebhookTarget, event_type: str, level: str, data: dict):
        """Fire a custom webhook."""
        try:
            headers = dict(wh.headers) if wh.headers else {}
            if wh.auth_token:
                headers["Authorization"] = f"Bearer {wh.auth_token}"

            payload = {
                "source": "SENTINEL",
                "event_type": event_type,
                "level": level,
                "timestamp": time.time(),
                **data,
            }

            if wh.method.upper() == "POST":
                await self.client.post(wh.url, headers=headers, json=payload)
            elif wh.method.upper() == "GET":
                await self.client.get(wh.url, headers=headers, params=payload)

            logger.info(f"Webhook fired: {wh.name}")

        except Exception as e:
            logger.error(f"Webhook error ({wh.name}): {e}")

    # ============================================================
    # Logging / Status
    # ============================================================
    def _log_event(self, event_type: str, level: str, data: dict):
        self.event_log.append({
            "time": time.strftime("%H:%M:%S"),
            "event_type": event_type,
            "level": level,
            "message": data.get("message", ""),
        })
        if len(self.event_log) > 200:
            self.event_log = self.event_log[-200:]

    def get_integration_status(self) -> dict:
        return {
            "home_assistant": {"configured": bool(self.ha_url), "url": self.ha_url},
            "ifttt": {"configured": bool(self.ifttt_key)},
            "telegram": {"configured": bool(self.telegram_bot_token)},
            "webhooks": [
                {"name": wh.name, "url": wh.url, "enabled": wh.enabled}
                for wh in self.webhooks
            ],
            "event_log": self.event_log[-20:],
        }

    async def close(self):
        await self.client.aclose()
