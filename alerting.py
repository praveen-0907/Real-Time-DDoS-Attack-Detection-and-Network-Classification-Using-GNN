"""
alerting.py — Windows Desktop Toast Notifications
===================================================
Sends Windows toast notifications when a DDoS attack is detected.
Uses winotify (preferred) with fallback to win10toast.

Install:  pip install winotify
"""

import time
import threading
from config import ALERTING, ATTACK_CLASSES, ATTACK_COLORS
from logger import get_logger

log = get_logger(__name__)

_last_alert_time: float = 0.0
_lock = threading.Lock()


def _send_winotify(title: str, body: str, icon_path: str = None):
    from winotify import Notification, audio
    toast = Notification(
        app_id="GNN DDoS Detection",
        title=title,
        msg=body,
        duration="short",
        icon=icon_path or "",
    )
    toast.set_audio(audio.Default, loop=False)
    toast.show()


def _send_win10toast(title: str, body: str):
    from win10toast import ToastNotifier
    toaster = ToastNotifier()
    toaster.show_toast(
        title,
        body,
        duration=ALERTING["toast_duration_sec"],
        threaded=True,
    )


def send_attack_alert(attack_class: int, confidence: float, top_ips: list = None):
    """
    Send a Windows toast notification for a detected attack.
    Respects cooldown period to avoid notification spam.
    """
    global _last_alert_time

    if not ALERTING["enabled"]:
        return

    with _lock:
        now = time.time()
        if now - _last_alert_time < ALERTING["cooldown_sec"]:
            log.debug("Alert suppressed — cooldown active")
            return
        _last_alert_time = now

    label = ATTACK_CLASSES.get(attack_class, "Unknown Attack")
    title = ALERTING["toast_title"]
    body_lines = [
        f"Type: {label}",
        f"Confidence: {confidence:.1%}",
    ]
    if top_ips:
        body_lines.append(f"Top attacker: {top_ips[0]}")

    body = "\n".join(body_lines)

    def _dispatch():
        try:
            _send_winotify(title, body)
            log.info(f"Toast alert sent: {label} ({confidence:.1%})")
        except ImportError:
            try:
                _send_win10toast(title, body)
                log.info(f"Toast alert sent via win10toast: {label}")
            except ImportError:
                log.warning("No toast library found. Install winotify: pip install winotify")
            except Exception as e:
                log.error(f"win10toast error: {e}")
        except Exception as e:
            log.error(f"Toast alert failed: {e}")

    threading.Thread(target=_dispatch, daemon=True).start()


def send_info_toast(title: str, body: str):
    """Send a generic informational toast."""
    def _dispatch():
        try:
            _send_winotify(title, body)
        except Exception:
            pass
    threading.Thread(target=_dispatch, daemon=True).start()
