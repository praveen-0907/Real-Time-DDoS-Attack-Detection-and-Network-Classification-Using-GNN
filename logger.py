"""
logger.py — Centralised Logging Setup
======================================
Writes logs to both console and a rotating session log file.
"""

import os
import logging
import glob
from datetime import datetime
from config import LOGGING, LOGS_DIR


def _setup_root_logger():
    level = getattr(logging, LOGGING.get("log_level", "INFO"), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.handlers:
        return  # Already configured

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    if LOGGING.get("log_to_file", True):
        os.makedirs(LOGS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOGS_DIR, f"session_{ts}.log")

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Rotate old logs
        _rotate_logs(LOGS_DIR, LOGGING.get("max_log_files", 10))


def _rotate_logs(log_dir: str, max_files: int):
    logs = sorted(glob.glob(os.path.join(log_dir, "session_*.log")), key=os.path.getmtime)
    for old in logs[:-max_files]:
        try:
            os.remove(old)
        except OSError:
            pass


_setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
