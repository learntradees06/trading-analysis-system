# Notifications module
# src/notifications.py
"""Notifications Module for Discord Alerts"""

import requests
import json
from datetime import datetime
from typing import Dict, Optional

class NotificationManager:
    def __init__(self, webhook_url: str = None):
        """
        Initialize Notification Manager

        Args:
            webhook_url: Discord webhook URL
        """
        self.webhook_url = webhook_url
        self._enabled = bool(webhook_url) # Enabled by default if URL is provided

    def is_enabled(self) -> bool:
        """Check if notifications are enabled."""
        return self._enabled

    def enable(self):
        """Enable notifications."""
        if self.webhook_url:
            self._enabled = True
        else:
            print("⚠️ Cannot enable notifications: no webhook URL is configured.")

    def disable(self):
        """Disable notifications."""
        self._enabled = False

    def send_to_discord(self, message: str, embed: Optional[Dict] = None) -> bool:
        """
        Send message to Discord webhook

        Args:
            message: Text message to send
            embed: Optional embed dictionary for rich formatting

        Returns:
            Boolean indicating success
        """
        if not self.is_enabled() or not self.webhook_url:
            if not self.webhook_url:
                print("Warning: No Discord webhook URL configured")
            return False

        payload = {"content": message}

        if embed:
            payload["embeds"] = [embed]

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            return response.status_code == 204
        except Exception as e:
            print(f"Error sending Discord notification: {e}")
            return False

    def send_signal_alert(self, ticker: str, signal: Dict) -> bool:
        """
        Send trading signal alert to Discord

        Args:
            ticker: Symbol
            signal: Signal dictionary with details

        Returns:
            Boolean indicating success
        """
        if not self.is_enabled():
            return False

        signal_type = signal.get('signal', 'NEUTRAL')
        score = signal.get('score', 0)
        confidence = signal.get('confidence', 'LOW')
        evidence = signal.get('evidence', [])

        # Determine color based on signal
        if 'LONG' in signal_type:
            color = 0x00ff00  # Green
            emoji = "🟢"
        elif 'SHORT' in signal_type:
            color = 0xff0000  # Red
            emoji = "🔴"
        else:
            color = 0xffff00  # Yellow
            emoji = "🟡"

        # Build embed
        embed = {
            "title": f"{emoji} {ticker} - {signal_type} Signal",
            "description": f"**Confidence:** {confidence} ({score:.1f}/100)",
            "color": color,
            "fields": [
                {
                    "name": "📊 Key Evidence",
                    "value": '\n'.join([f"• {e}" for e in evidence[:5]]),
                    "inline": False
                }
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "Advanced Trading System Alert"
            }
        }

        # Add component scores if available
        if 'component_scores' in signal:
            scores_text = '\n'.join([
                f"• {k.replace('_', ' ').title()}: {v:.1f}"
                for k, v in signal['component_scores'].items()
            ])
            embed["fields"].append({
                "name": "🎯 Component Scores",
                "value": scores_text,
                "inline": True
            })

        message = f"**Trading Alert for {ticker}**"

        return self.send_to_discord(message, embed)

    def send_market_update(self, ticker: str, update_type: str, data: Dict) -> bool:
        """
        Send market update notification

        Args:
            ticker: Symbol
            update_type: Type of update (e.g., 'IB_BREAK', 'NEW_HIGH')
            data: Relevant data for the update

        Returns:
            Boolean indicating success
        """
        if not self.is_enabled():
            return False

        updates = {
            'IB_BREAK': {
                'title': f"📈 {ticker} - Initial Balance Break",
                'color': 0x0099ff,
                'description': "Price has broken outside the Initial Balance range"
            },
            'NEW_HIGH': {
                'title': f"🚀 {ticker} - New Session High",
                'color': 0x00ff00,
                'description': "Price has made a new high for the session"
            },
            'NEW_LOW': {
                'title': f"📉 {ticker} - New Session Low",
                'color': 0xff0000,
                'description': "Price has made a new low for the session"
            },
            'POC_TOUCH': {
                'title': f"🎯 {ticker} - POC Touch",
                'color': 0x9900ff,
                'description': "Price has touched the Point of Control"
            }
        }

        if update_type not in updates:
            return False

        update_info = updates[update_type]

        embed = {
            "title": update_info['title'],
            "description": update_info['description'],
            "color": update_info['color'],
            "fields": [],
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add data fields
        if data:
            for key, value in data.items():
                embed["fields"].append({
                    "name": key.replace('_', ' ').title(),
                    "value": str(value),
                    "inline": True
                })

        return self.send_to_discord("", embed)
