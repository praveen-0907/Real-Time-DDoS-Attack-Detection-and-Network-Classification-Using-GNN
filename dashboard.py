"""
dashboard.py — Flask + SocketIO Web Server
==========================================
All HTTP routes and SocketIO events. No HTML here — templates/ handles the UI.

Run:  python dashboard.py
"""

import os
import sys
import glob
import threading
import webbrowser
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from functools import wraps
import time
from collections import defaultdict

from config import SERVER, MODELS_DIR, PCAP, CHECKPOINTING, ATTACK_CLASSES, BASE_DIR
from detector import DetectionSystem
from firewall import block_ip, unblock_ip, is_blocked, list_blocked_ips
from logger import get_logger
from model import load_checkpoint

log = get_logger(__name__)

# ─── Rate Limiting ────────────────────────────────────────────────────────────

_rate_limit_store = defaultdict(list)
_rate_limit_lock = threading.Lock()

def rate_limit(max_calls: int = 10, window_sec: int = 60):
    """Simple rate limiting decorator for API endpoints."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            client_ip = request.remote_addr
            now = time.time()
            
            with _rate_limit_lock:
                # Clean old entries
                _rate_limit_store[client_ip] = [
                    t for t in _rate_limit_store[client_ip] 
                    if now - t < window_sec
                ]
                
                # Check limit
                if len(_rate_limit_store[client_ip]) >= max_calls:
                    return jsonify({
                        "success": False, 
                        "error": f"Rate limit exceeded. Max {max_calls} requests per {window_sec}s."
                    }), 429
                
                # Record this request
                _rate_limit_store[client_ip].append(now)
            
            return f(*args, **kwargs)
        return wrapped
    return decorator

# ─── App Setup ────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.config["SECRET_KEY"]    = SERVER["secret_key"]
app.config["MAX_CONTENT_LENGTH"] = PCAP["max_file_size_mb"] * 1024 * 1024

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Shared detection system (initialised on first start)
_system: DetectionSystem = None


def get_system() -> DetectionSystem:
    global _system
    if _system is None:
        _system = DetectionSystem(socketio)
        # Auto-load best or latest model if available
        for candidate in [CHECKPOINTING["best_model_file"], CHECKPOINTING["latest_model_file"]]:
            if os.path.exists(candidate):
                ok, msg = _system.load_model(candidate)
                if ok:
                    log.info(f"Auto-loaded model: {candidate}")
                    break
    return _system


# ─── Page Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html", attack_classes=ATTACK_CLASSES)


# ─── API: Monitoring Control ──────────────────────────────────────────────────

@app.route("/api/start", methods=["POST"])
def api_start():
    data             = request.json or {}
    interface        = data.get("interface")
    capture_interval = float(data.get("capture_interval", 5))
    train_every      = int(data.get("train_every", 3))

    if not interface:
        return jsonify({"success": False, "error": "No interface specified."})

    sys_ = get_system()
    ok   = sys_.start(interface, capture_interval, train_every)
    return jsonify({"success": ok})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    sys_ = get_system()
    ok   = sys_.stop()
    return jsonify({"success": ok})


@app.route("/api/pause", methods=["POST"])
def api_pause():
    return jsonify({"success": get_system().pause()})


@app.route("/api/resume", methods=["POST"])
def api_resume():
    return jsonify({"success": get_system().resume()})


@app.route("/api/toggle_training", methods=["POST"])
def api_toggle_training():
    enabled = get_system().toggle_training()
    return jsonify({"success": True, "training_enabled": enabled})


@app.route("/api/status", methods=["GET"])
def api_status():
    sys_  = get_system()
    stats = sys_.stats.copy()
    return jsonify({
        "is_running":       sys_.is_running,
        "is_paused":        sys_.is_paused,
        "training_enabled": sys_.training_enabled,
        "interface":        sys_.interface,
        "stats":            stats,
    })


# ─── API: Network Interfaces ──────────────────────────────────────────────────

@app.route("/api/interfaces", methods=["GET"])
def api_interfaces():
    try:
        from scapy.all import get_if_list
        interfaces = get_if_list()
    except Exception as e:
        log.error(f"Interface list failed: {e}")
        interfaces = []

    sys_ = get_system()
    return jsonify({
        "interfaces":        interfaces,
        "current_interface": sys_.interface,
    })

# ───  API: Network Interface Suggest ───────────────────────────────────────

@app.route("/api/interfaces/suggest", methods=["GET"])
def api_suggest_interface():
    from scapy.all import sniff, get_if_list
    try:
        interfaces = get_if_list()
    except Exception:
        return jsonify({"suggested": None, "counts": {}})

    counts = {}
    lock   = threading.Lock()

    def test_iface(iface):
        try:
            counter = []
            sniff(iface=iface, timeout=1, store=False, count=200, prn=lambda p: counter.append(1))
            with lock:
                counts[iface] = len(counter)
        except Exception:
            with lock:
                counts[iface] = 0

    threads = [threading.Thread(target=test_iface, args=(i,), daemon=True) for i in interfaces]
    for t in threads: t.start()
    for t in threads: t.join(timeout=2.5)

    active = {k: v for k, v in counts.items() if v > 0}
    suggested = max(active, key=active.get) if active else None
    if suggested and ("Loopback" in suggested or suggested in ("lo", "lo0")):
        non_loop = {k: v for k, v in counts.items() if "Loopback" not in k and k not in ("lo", "lo0")}
        suggested = max(non_loop, key=non_loop.get) if non_loop else None

    log.info(f"Interface suggestion: {suggested} | counts: {counts}")
    return jsonify({"suggested": suggested, "counts": counts})


# ─── API: Model Management ────────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def api_models():
    import torch
    files = sorted(
        glob.glob(os.path.join(MODELS_DIR, "*.pth")),
        key=os.path.getmtime,
        reverse=True,
    )
    result = []
    for f in files:
        try:
            ckpt = torch.load(f, map_location="cpu")
            result.append({
                "filename":   os.path.basename(f),
                "path":       f,
                "size_mb":    round(os.path.getsize(f) / 1_048_576, 2),
                "modified":   datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S"),
                "iterations": ckpt.get("stats", {}).get("total_iterations", "—"),
                "val_acc":    ckpt.get("stats", {}).get("val_acc", None),
            })
        except Exception:
            result.append({
                "filename": os.path.basename(f),
                "path":     f,
                "size_mb":  round(os.path.getsize(f) / 1_048_576, 2),
                "modified": datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S"),
                "iterations": "—",
                "val_acc":    None,
            })

    sys_ = get_system()
    return jsonify({
        "models":        result,
        "current_model": sys_.stats.get("loaded_model", "None"),
    })


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    data = request.json or {}
    path = data.get("model_path", "")
    if not path or not os.path.exists(path):
        return jsonify({"success": False, "error": "Model file not found."})
    ok, msg = get_system().load_model(path)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/save_model", methods=["POST"])
def api_save_model():
    try:
        path = get_system().manual_save()
        return jsonify({"success": True, "filename": os.path.basename(path)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ─── API: Firewall ────────────────────────────────────────────────────────────

@app.route("/api/firewall/block", methods=["POST"])
@rate_limit(max_calls=20, window_sec=60)  # Max 20 blocks per minute
def api_block():
    ip = (request.json or {}).get("ip", "").strip()
    if not ip:
        return jsonify({"success": False, "error": "No IP provided."})
    ok, msg = block_ip(ip)
    if ok:
        sys_ = get_system()
        if ip in sys_.attacker_ips:
            sys_.attacker_ips[ip]["blocked"] = True
        socketio.emit("attacker_ips_update", list(sys_.attacker_ips.values()))
    return jsonify({"success": ok, "message": msg})


@app.route("/api/firewall/unblock", methods=["POST"])
def api_unblock():
    ip = (request.json or {}).get("ip", "").strip()
    if not ip:
        return jsonify({"success": False, "error": "No IP provided."})
    ok, msg = unblock_ip(ip)
    if ok:
        sys_ = get_system()
        if ip in sys_.attacker_ips:
            sys_.attacker_ips[ip]["blocked"] = False
        socketio.emit("attacker_ips_update", list(sys_.attacker_ips.values()))
    return jsonify({"success": ok, "message": msg})


@app.route("/api/firewall/blocked", methods=["GET"])
def api_blocked_list():
    return jsonify({"blocked_ips": list_blocked_ips()})


@app.route("/api/mark_false_positive", methods=["POST"])
def api_mark_false_positive():
    """Mark a detection event as false positive for threshold tuning."""
    import json
    from datetime import datetime as dt
    
    data = request.json or {}
    timestamp = data.get("timestamp")
    label = data.get("label")
    confidence = data.get("confidence")
    
    if not timestamp or not label:
        return jsonify({"success": False, "error": "Missing timestamp or label"})
    
    # Store false positive
    fp_file = os.path.join(BASE_DIR, "false_positives.json")
    fp_entry = {
        "timestamp": timestamp,
        "marked_at": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "confidence": confidence,
        "packet_count": data.get("packet_count", 0),
        "probs": data.get("probs", {}),
    }
    
    try:
        # Load existing FPs
        if os.path.exists(fp_file):
            with open(fp_file, "r") as f:
                fps = json.load(f)
        else:
            fps = []
        
        fps.append(fp_entry)
        
        # Save updated FPs
        with open(fp_file, "w") as f:
            json.dump(fps, f, indent=2)
        
        log.info(f"Marked false positive: {label} at {timestamp}")
        return jsonify({"success": True, "message": f"Marked {label} as false positive", "total_fps": len(fps)})
    except Exception as e:
        log.error(f"Failed to mark false positive: {e}")
        return jsonify({"success": False, "error": str(e)})


# ─── API: PCAP Upload & Analysis ──────────────────────────────────────────────

@app.route("/api/pcap/upload", methods=["POST"])
def api_pcap_upload():
    from pcap_analyzer import allowed_file, analyse_pcap

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request."})

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"success": False, "error": "No file selected."})

    if not allowed_file(f.filename):
        return jsonify({"success": False, "error": "Only .pcap / .pcapng / .cap files allowed."})

    filename = secure_filename(f.filename)
    save_path = os.path.join(PCAP["upload_folder"], filename)
    f.save(save_path)
    log.info(f"PCAP uploaded: {save_path}")

    sys_ = get_system()
    sys_._ensure_model()

    def _run_analysis():
        socketio.emit("pcap_progress", {"status": "started", "filename": filename})
        try:
            def progress(done, total):
                pct = int(done / total * 100)
                socketio.emit("pcap_progress", {"status": "running", "pct": pct, "done": done, "total": total})

            result = analyse_pcap(
                save_path,
                sys_.model,
                device=sys_.device,
                progress_cb=progress,
            )
            socketio.emit("pcap_result", result)
            socketio.emit("pcap_progress", {"status": "done"})
        except Exception as e:
            log.error(f"PCAP analysis error: {e}")
            socketio.emit("pcap_progress", {"status": "error", "message": str(e)})

    threading.Thread(target=_run_analysis, daemon=True).start()
    return jsonify({"success": True, "filename": filename})


# ─── SocketIO Events ──────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    sys_ = get_system()
    socketio.emit("stats_update", sys_.stats)
    socketio.emit("attacker_ips_update", list(sys_.attacker_ips.values()))
    log.info("Dashboard client connected")


@socketio.on("disconnect")
def on_disconnect():
    log.info("Dashboard client disconnected")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import platform

    print("=" * 65)
    print("  GNN DDoS Detection System")
    print("=" * 65)
    print(f"  Platform : {platform.system()} {platform.release()}")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  URL      : http://localhost:{SERVER['port']}")
    print("=" * 65)

    if SERVER.get("auto_open_browser"):
        threading.Timer(
            1.5,
            lambda: webbrowser.open(f"http://localhost:{SERVER['port']}")
        ).start()

    socketio.run(
        app,
        host=SERVER["host"],
        port=SERVER["port"],
        debug=SERVER["debug"],
        allow_unsafe_werkzeug=True,
    )
