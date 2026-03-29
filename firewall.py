"""
firewall.py — Windows Firewall IP Block/Unblock via netsh
==========================================================
Adds / removes inbound block rules using Windows Firewall (netsh advfirewall).
Requires the process to be running with Administrator privileges.
"""

import subprocess
import re
from typing import Tuple
from logger import get_logger

log = get_logger(__name__)

RULE_PREFIX = "GNN_DDoS_Block_"


def _rule_name(ip: str) -> str:
    safe_ip = ip.replace(".", "_")
    return f"{RULE_PREFIX}{safe_ip}"


def _is_valid_ip(ip: str) -> bool:
    pattern = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
    if not pattern.match(ip):
        return False
    return all(0 <= int(o) <= 255 for o in ip.split("."))


def block_ip(ip: str) -> Tuple[bool, str]:
    """
    Add a Windows Firewall inbound block rule for the given IP.
    Returns (success, message).
    """
    if not _is_valid_ip(ip):
        return False, f"Invalid IP address: {ip}"

    # Check if already blocked to avoid duplicate rules
    if is_blocked(ip):
        return True, f"{ip} is already blocked"

    rule = _rule_name(ip)
    cmd = [
        "netsh", "advfirewall", "firewall", "add", "rule",
        f"name={rule}",
        "dir=in",
        "action=block",
        f"remoteip={ip}",
        "enable=yes",
        "description=Blocked by GNN DDoS Detection System",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            msg = f"Blocked {ip} via Windows Firewall (rule: {rule})"
            log.info(msg)
            return True, msg
        else:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            error_msg = stderr or stdout
            
            # Check for common permission errors
            if "access is denied" in error_msg.lower() or "requested operation requires elevation" in error_msg.lower():
                msg = f"Administrator privileges required. Right-click terminal and 'Run as Administrator', then restart dashboard."
                log.error(f"Permission denied blocking {ip}: {error_msg}")
                return False, msg
            
            msg = f"Failed to block {ip}: {error_msg}"
            log.error(msg)
            return False, msg
    except subprocess.TimeoutExpired:
        return False, f"Timeout while blocking {ip}"
    except PermissionError:
        msg = "Administrator privileges required. Right-click terminal and 'Run as Administrator', then restart dashboard."
        log.error(f"Permission denied: {msg}")
        return False, msg
    except FileNotFoundError:
        return False, "netsh not found. Make sure you are on Windows."
    except Exception as e:
        error_str = str(e)
        if "access is denied" in error_str.lower():
            return False, "Administrator privileges required. Right-click terminal and 'Run as Administrator', then restart dashboard."
        return False, error_str


def unblock_ip(ip: str) -> Tuple[bool, str]:
    """Remove the Windows Firewall block rule for the given IP."""
    if not _is_valid_ip(ip):
        return False, f"Invalid IP address: {ip}"

    rule = _rule_name(ip)
    cmd = [
        "netsh", "advfirewall", "firewall", "delete", "rule",
        f"name={rule}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            msg = f"Unblocked {ip} — firewall rule removed."
            log.info(msg)
            return True, msg
        else:
            msg = f"Could not remove rule for {ip}: {result.stderr.strip() or result.stdout.strip()}"
            log.warning(msg)
            return False, msg
    except Exception as e:
        return False, str(e)


def is_blocked(ip: str) -> bool:
    """Check whether a block rule exists for the given IP."""
    rule = _rule_name(ip)
    cmd = [
        "netsh", "advfirewall", "firewall", "show", "rule",
        f"name={rule}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return "No rules match" not in result.stdout and result.returncode == 0
    except Exception:
        return False


def list_blocked_ips() -> list:
    """Return list of IPs currently blocked by this system."""
    cmd = [
        "netsh", "advfirewall", "firewall", "show", "rule",
        f"name={RULE_PREFIX}*",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        ips = []
        for line in result.stdout.splitlines():
            if "RemoteIP:" in line:
                ip = line.split("RemoteIP:")[-1].strip()
                if _is_valid_ip(ip):
                    ips.append(ip)
        return ips
    except Exception:
        return []


def block_multiple_ips(ip_list: list) -> Tuple[int, int, list]:
    """
    Block multiple IPs efficiently.
    Returns (success_count, fail_count, failed_ips).
    """
    success_count = 0
    fail_count = 0
    failed_ips = []
    
    for ip in ip_list:
        ok, msg = block_ip(ip)
        if ok:
            success_count += 1
        else:
            fail_count += 1
            failed_ips.append(ip)
    
    return success_count, fail_count, failed_ips
