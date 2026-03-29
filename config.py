"""
config.py — Central configuration for GNN DDoS Detection System
================================================================
Edit this file to change any system-wide settings.
"""

import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
DATA_DIR        = os.path.join(BASE_DIR, "data")
PROCESSED_DIR   = os.path.join(BASE_DIR, "processed_data")
TEMPLATES_DIR   = os.path.join(BASE_DIR, "templates")
STATIC_DIR      = os.path.join(BASE_DIR, "static")

for _dir in [MODELS_DIR, LOGS_DIR, DATA_DIR, PROCESSED_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL = {
    # 15 edge features per flow edge:
    # [0] packet count          [1] total bytes         [2] mean pkt size
    # [3] std pkt size          [4] max pkt size        [5] duration ms
    # [6] packet rate           [7] SYN flag count      [8] ACK flag count
    # [9] FIN flag count        [10] RST flag count     [11] fwd/bwd ratio
    # [12] IAT mean             [13] IAT std            [14] unique src IPs
    "edge_feature_dim": 15,
    "hidden_dim": 64,
    "num_layers": 3,
    "dropout": 0.3,
    "num_classes": 6,            # Normal + 5 attack types
}

# ─── Attack Classes ───────────────────────────────────────────────────────────
# Index → label mapping (0 = Normal)
ATTACK_CLASSES = {
    0: "Normal",
    1: "SYN Flood",
    2: "UDP Flood",
    3: "HTTP Flood",
    4: "ICMP Flood",
    5: "DNS Amplification",
}

ATTACK_COLORS = {
    0: "#22c55e",   # green  – Normal
    1: "#ef4444",   # red    – SYN Flood
    2: "#f97316",   # orange – UDP Flood
    3: "#eab308",   # yellow – HTTP Flood
    4: "#a855f7",   # purple – ICMP Flood
    5: "#06b6d4",   # cyan   – DNS Amplification
}

# CICIDS2017/19 column → our attack class mapping
CICIDS_LABEL_MAP = {
    "BENIGN":               0,
    "benign":                  0,
    "DoS Hulk":             3,   # HTTP Flood
    "PortScan":             0,   # treat as normal for now
    "DDoS":                 1,   # default to SYN Flood
    "DoS GoldenEye":        3,
    "FTP-Patator":          0,
    "SSH-Patator":          0,
    "DoS slowloris":        3,
    "DoS Slowhttptest":     3,
    "Bot":                  0,
    "Web Attack \x96 Brute Force": 0,
    "Web Attack \x96 XSS":  0,
    "Infiltration":         0,
    "Web Attack \x96 Sql Injection": 0,
    "Heartbleed":           0,
    "SYN":                  1,
    "UDP":                  2,
    "ICMP":                 4,
    "UDP-lag":              2,
    "DrDoS_DNS":            5,
    "DrDoS_LDAP":           5,
    "DrDoS_MSSQL":          5,
    "DrDoS_NetBIOS":        5,
    "DrDoS_NTP":            5,
    "DrDoS_SNMP":           5,
    "DrDoS_SSDP":           5,
    "DrDoS_UDP":            2,
    "Syn":                  1,
    "TFTP":                 2,
    "UDPLag":               2,   # from here variant spelling in some 2019 files
    "WebDDoS":              3,
    "PortMap":              5,
    "NetBIOS":              5,
    "LDAP":                 5,
    "MSSQL":                5,
    "NTP":                  5,
    "DNS":                  5,
    "SNMP":                 5,
}

# ─── Live Detection ───────────────────────────────────────────────────────────
DETECTION = {
    "capture_interval_sec": 3,       # seconds per capture window
    "train_every_n_iter": 5,         # fine-tune every N iterations (increased for stability)
    "min_packets_for_detection": 10, # skip if fewer packets captured (raised threshold)
    "heuristic_attack_threshold": 8, # score >= this → label as attack (more conservative)
    "confidence_threshold": 0.75,    # GNN confidence to trigger alert/block prompt (raised)
    "attack_confirmation_window": 3, # require N consecutive attack detections before alerting
    "baseline_learning_iterations": 20, # learn normal traffic baseline for first N iterations
    "max_normal_packet_rate": 200,  # packets/sec — above this is suspicious
    "max_normal_unique_ips": 50,    # unique IPs per window — above this is suspicious
}

# ─── Checkpointing ────────────────────────────────────────────────────────────
CHECKPOINTING = {
    "save_every_n_iter": 10,    # save checkpoint every N iterations
    "max_checkpoints": 5,       # rolling window — delete oldest beyond this
    "best_model_file": os.path.join(MODELS_DIR, "best_model.pth"),
    "latest_model_file": os.path.join(MODELS_DIR, "latest_model.pth"),
}

# ─── Training (Offline Pre-training) ──────────────────────────────────────────
TRAINING = {
    "batch_size": 128,           # RTX 3050 6GB — safe at 128; drop to 64 if OOM
    "learning_rate": 1e-3,
    "epochs": 50,
    "val_split": 0.2,
    "early_stop_patience": 10,
    "num_workers": 4,            # Ryzen 5 7000 — 4 parallel loader workers (set to 0 if errors occur)
    "graph_window_size": 100,    # packets per graph during pre-training
    "calibrate_temperature": True,  # Enable temperature scaling for better confidence estimates
}

# ─── Alerting ─────────────────────────────────────────────────────────────────
ALERTING = {
    "enabled": True,
    "toast_duration_sec": 8,
    "toast_title": "⚠ DDoS Attack Detected",
    "cooldown_sec": 30,          # minimum seconds between repeated alerts
}

# ─── Logging ──────────────────────────────────────────────────────────────────
LOGGING = {
    "log_to_file": True,
    "log_dir": LOGS_DIR,
    "max_log_files": 10,         # rotate after this many session logs
    "log_level": "INFO",
}

# ─── Dashboard / Server ───────────────────────────────────────────────────────
SERVER = {
    "host": "0.0.0.0",
    "port": 5000,
    "secret_key": "gnn-ddos-secret-2024",
    "debug": False,
    "auto_open_browser": True,
}

# ─── PCAP Analysis ────────────────────────────────────────────────────────────
PCAP = {
    "max_file_size_mb": 500,
    "graph_window_size": 200,    # packets per graph slice
    "upload_folder": os.path.join(BASE_DIR, "uploads"),
}
os.makedirs(PCAP["upload_folder"], exist_ok=True)
