"""
Notification service for SENTINEL.
Sends push notifications via ntfy.sh with cooldown management.
"""
import time
import asyncio
from typing import Optional

import httpx

from sentinel.config import settings


class NotificationService:
    """Manages push notifications with cooldowns and priority routing."""

    def __init__(self):
        self.cooldowns = {
            "CRITICAL": 0,
            "SUSPICIOUS": 0,
        }
        self.notification_log = []
        self.client = httpx.AsyncClient(timeout=10.0)

    async def send(self, level: str, message: str,
                   weapon_detected: bool = False,
                   follower: bool = False,
                   bypass_cooldown: bool = False) -> bool:
        """
        Send a notification via ntfy.sh.

        Returns True if sent, False if skipped (cooldown/no topic).
        """
        topic = settings.NTFY_TOPIC
        if not topic:
            return False

        now = time.time()

        # Check cooldown (weapon bypasses)
        if not bypass_cooldown and not weapon_detected:
            if level == "CRITICAL":
                if now - self.cooldowns["CRITICAL"] < settings.COOLDOWN_CRITICAL_SEC:
                    return False
                self.cooldowns["CRITICAL"] = now
            elif level == "SUSPICIOUS":
                if now - self.cooldowns["SUSPICIOUS"] < settings.COOLDOWN_SUSPICIOUS_SEC:
                    return False
                self.cooldowns["SUSPICIOUS"] = now

        # Build headers
        headers = {
            "Title": "SENTINEL CRITICAL" if level == "CRITICAL" else "SENTINEL ALERT",
            "Priority": "urgent" if level == "CRITICAL" else "high",
            "Tags": "rotating_light,sos" if level == "CRITICAL" else "warning",
        }

        if weapon_detected:
            headers["Tags"] += ",knife"
            headers["Title"] = "WEAPON DETECTED — SENTINEL"
            headers["Priority"] = "urgent"

        if follower:
            headers["Tags"] += ",eyes"
            if not weapon_detected:
                headers["Title"] = "KNOWN FOLLOWER DETECTED"

        url = f"{settings.NTFY_BASE_URL}/{topic}"

        try:
            response = await self.client.post(url, headers=headers, content=message)
            success = response.status_code == 200

            self.notification_log.append({
                "time": time.strftime("%H:%M:%S"),
                "level": level,
                "message": message,
                "sent": success,
            })

            # Keep log manageable
            if len(self.notification_log) > 100:
                self.notification_log = self.notification_log[-100:]

            return success
        except Exception as e:
            self.notification_log.append({
                "time": time.strftime("%H:%M:%S"),
                "level": "ERROR",
                "message": f"Failed: {str(e)}",
                "sent": False,
            })
            return False

    async def send_test(self, topic: Optional[str] = None) -> bool:
        """Send a test notification."""
        t = topic or settings.NTFY_TOPIC
        if not t:
            return False

        try:
            response = await self.client.post(
                f"{settings.NTFY_BASE_URL}/{t}",
                headers={
                    "Title": "SENTINEL TEST",
                    "Priority": "default",
                    "Tags": "white_check_mark",
                },
                content="Test notification from SENTINEL system.",
            )
            return response.status_code == 200
        except Exception:
            return False

    def process_alerts(self, alerts: list):
        """Process alert list from follower tracker asynchronously."""
        for alert in alerts:
            asyncio.create_task(self.send(
                level=alert["level"],
                message=alert["message"],
                weapon_detected="ARMED" in alert.get("type", ""),
                follower="FOLLOWER" in alert.get("type", ""),
                bypass_cooldown=alert.get("bypass_cooldown", False),
            ))

    async def close(self):
        await self.client.aclose()
