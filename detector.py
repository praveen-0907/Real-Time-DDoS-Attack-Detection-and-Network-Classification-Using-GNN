"""
detector.py — Live Detection Engine
====================================
Handles: packet capture → graph construction → GNN inference →
         heuristic labelling → incremental fine-tuning → checkpoint management
"""

import os
import sys
import time
import socket
import threading
import signal
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from torch_geometric.data import Data

from scapy.all import conf as scapy_conf
scapy_conf.use_pcap = True     # force Npcap backend on Windows
scapy_conf.sniff_promisc = 1

from config import (
    ATTACK_CLASSES, CHECKPOINTING, DETECTION, LOGGING, MODELS_DIR
)
from model import DDoSDetectionGNN, save_checkpoint, load_checkpoint, rotate_checkpoints
from logger import get_logger

log = get_logger(__name__)


# ─── Edge Feature Indices ─────────────────────────────────────────────────────
# [0]  packet count        [1]  total bytes         [2]  mean pkt size
# [3]  std pkt size        [4]  max pkt size        [5]  duration ms
# [6]  packet rate         [7]  SYN flag count      [8]  ACK flag count
# [9]  FIN flag count      [10] RST flag count      [11] fwd/bwd ratio
# [12] IAT mean            [13] IAT std             [14] unique src IPs
EDGE_DIM = 15


def build_graph(packets_df: pd.DataFrame, window_start: float, window_end: float) -> Optional[Data]:
    """
    Convert a DataFrame of raw packets into a PyG graph.

    Nodes  = unique IP addresses
    Edges  = directed flows (src_ip → dst_ip)
    Edge features (15):
      [0]  packet count        [1]  total bytes         [2]  mean pkt size
      [3]  std pkt size        [4]  max pkt size        [5]  duration ms
      [6]  packet rate         [7]  SYN flag count      [8]  ACK flag count
      [9]  FIN flag count      [10] RST flag count      [11] fwd/bwd ratio
      [12] IAT mean            [13] IAT std             [14] unique src IPs
    """
    if packets_df is None or len(packets_df) == 0:
        return None

    ip_to_id: Dict[str, int] = {}
    node_counter = 0

    def get_id(ip: str) -> int:
        nonlocal node_counter
        if ip not in ip_to_id:
            ip_to_id[ip] = node_counter
            node_counter += 1
        return ip_to_id[ip]

    edge_dict = defaultdict(lambda: {
        "count": 0, "bytes": 0, "sizes": [],
        "timestamps": [],
        "first_ts": None, "last_ts": None,
        "syn": 0, "ack": 0, "fin": 0, "rst": 0,
        "src_ips": set(),
    })

    duration = max(window_end - window_start, 1e-3)

    for _, pkt in packets_df.iterrows():
        src = get_id(str(pkt["src_ip"]))
        dst = get_id(str(pkt["dst_ip"]))
        e   = (src, dst)

        sz        = int(pkt["packet_size"])
        ts        = float(pkt.get("timestamp", window_start))
        tcp_flags = int(pkt.get("tcp_flags", 0))

        edge_dict[e]["count"]  += 1
        edge_dict[e]["bytes"]  += sz
        edge_dict[e]["sizes"].append(sz)
        edge_dict[e]["timestamps"].append(ts)
        edge_dict[e]["src_ips"].add(str(pkt["src_ip"]))

        if edge_dict[e]["first_ts"] is None or ts < edge_dict[e]["first_ts"]:
            edge_dict[e]["first_ts"] = ts
        if edge_dict[e]["last_ts"] is None or ts > edge_dict[e]["last_ts"]:
            edge_dict[e]["last_ts"] = ts

        # TCP flag bitmask decomposition (use counts, not binary)
        if tcp_flags & 0x02:  # SYN
            edge_dict[e]["syn"] += 1
        if tcp_flags & 0x10:  # ACK
            edge_dict[e]["ack"] += 1
        if tcp_flags & 0x01:  # FIN
            edge_dict[e]["fin"] += 1
        if tcp_flags & 0x04:  # RST
            edge_dict[e]["rst"] += 1

    if node_counter == 0:
        return None

    # Node-level fwd/bwd counts for directionality ratio
    fwd_counts: Dict[int, int] = defaultdict(int)
    bwd_counts: Dict[int, int] = defaultdict(int)
    for (src, dst), d in edge_dict.items():
        fwd_counts[src] += d["count"]
        bwd_counts[dst] += d["count"]

    edge_list, edge_features = [], []
    for (src, dst), d in edge_dict.items():
        sizes = d["sizes"]
        tss   = sorted(d["timestamps"])

        dur_ms   = ((d["last_ts"] or window_end) - (d["first_ts"] or window_start)) * 1000
        pkt_rate = d["count"] / max(duration, 1e-6)

        if len(tss) > 1:
            iats     = [tss[i+1] - tss[i] for i in range(len(tss)-1)]
            iat_mean = float(np.mean(iats))
            iat_std  = float(np.std(iats))
        else:
            iat_mean = 0.0
            iat_std  = 0.0

        fwd_bwd = fwd_counts[src] / max(bwd_counts[src], 1)

        edge_list.append([src, dst])
        edge_features.append([
            d["count"],                                               # 0
            d["bytes"],                                               # 1
            float(np.mean(sizes)),                                    # 2
            float(np.std(sizes)) if len(sizes) > 1 else 0.0,         # 3
            float(np.max(sizes)),                                     # 4
            dur_ms,                                                   # 5
            pkt_rate,                                                 # 6
            float(d["syn"]),                                          # 7
            float(d["ack"]),                                          # 8
            float(d["fin"]),                                          # 9
            float(d["rst"]),                                          # 10
            fwd_bwd,                                                  # 11
            iat_mean,                                                 # 12
            iat_std,                                                  # 13
            float(len(d["src_ips"])),                                 # 14
        ])

    if not edge_list:
        return None

    x          = torch.ones((node_counter, 1), dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_features, dtype=torch.float)

    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
    edge_attr = torch.log1p(torch.abs(edge_attr))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.ip_mapping = ip_to_id
    return data


# ─── Heuristic Labeller ───────────────────────────────────────────────────────

def heuristic_label(packets_df: pd.DataFrame, baseline_stats: Optional[dict] = None, duration: float = 3.0) -> int:
    """
    Multi-factor heuristic to label a traffic window.
    Returns class index (0–5) matching ATTACK_CLASSES.
    
    Now includes:
    - Baseline-aware thresholds (adapts to user's normal traffic)
    - QUIC protocol detection (UDP port 443 = legitimate streaming)
    - More conservative scoring to reduce false positives
    - Configurable duration for accurate packet rate calculation
    """
    if packets_df is None or len(packets_df) == 0:
        return 0

    total        = len(packets_df)
    pkt_rate     = total / duration
    unique_src   = packets_df["src_ip"].nunique()
    unique_dst   = packets_df["dst_ip"].nunique()
    avg_size     = packets_df["packet_size"].mean()
    proto_counts = packets_df["protocol"].value_counts()
    dominant_proto = proto_counts.index[0] if len(proto_counts) else -1
    dom_ratio    = proto_counts.iloc[0] / total if total else 0

    # Identify our own IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        my_ip = s.getsockname()[0]
        s.close()
    except Exception:
        my_ip = None

    incoming      = (packets_df["dst_ip"] == my_ip).sum() if my_ip else 0
    outgoing      = (packets_df["src_ip"] == my_ip).sum() if my_ip else 0
    incoming_ratio = incoming / total if total else 0

    # dst_port analysis
    dst_ports = packets_df.get("dst_port", pd.Series(dtype=int))
    http_traffic  = ((dst_ports == 80) | (dst_ports == 8080)).sum()
    https_traffic = (dst_ports == 443).sum()
    dns_traffic   = (dst_ports == 53).sum()
    
    # QUIC detection (UDP + port 443 = YouTube/Netflix/modern streaming)
    quic_traffic = 0
    if dominant_proto == 17:  # UDP
        quic_traffic = ((dst_ports == 443) | (dst_ports == 80)).sum()
    quic_ratio = quic_traffic / max(total, 1)

    # Baseline-aware thresholds
    if baseline_stats:
        baseline_rate = baseline_stats.get("avg_packet_rate", 50)
        baseline_ips  = baseline_stats.get("avg_unique_ips", 10)
        rate_multiplier = pkt_rate / max(baseline_rate, 10)  # how many times normal?
        ip_multiplier   = unique_src / max(baseline_ips, 5)
    else:
        rate_multiplier = pkt_rate / 50  # fallback: assume 50 pkt/s is normal
        ip_multiplier   = unique_src / 10

    # Scoring per attack type
    scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # SYN, UDP, HTTP, ICMP, DNS

    # ─── SYN Flood ───
    # Many sources, small packets, TCP proto=6, high rate, incoming-heavy
    if dominant_proto == 6 and avg_size < 100 and rate_multiplier > 3 and unique_src > 20:
        scores[1] += 6
    if unique_src > 30 and unique_dst < 3 and incoming_ratio > 0.7:
        scores[1] += 4
    if dominant_proto == 6 and pkt_rate > 100 and avg_size < 80:
        scores[1] += 3

    # ─── UDP Flood ───
    # UDP proto=17, high rate, many sources, NOT QUIC
    if dominant_proto == 17 and quic_ratio < 0.3:  # NOT legitimate streaming
        if rate_multiplier > 4 and dom_ratio > 0.85:
            scores[2] += 6
        if unique_src > 25 and pkt_rate > 80:
            scores[2] += 4
        if avg_size < 100 and pkt_rate > 100:  # small packets, high rate
            scores[2] += 3

    # ─── HTTP Flood ───
    # TCP port 80/443, many sources, medium-large packets, high rate
    http_ratio = (http_traffic + https_traffic) / max(total, 1)
    if http_ratio > 0.6 and unique_src > 15 and rate_multiplier > 3:
        scores[3] += 6
    if dominant_proto == 6 and avg_size > 200 and pkt_rate > 50 and unique_src > 20:
        scores[3] += 3

    # ─── ICMP Flood ───
    # proto=1, high rate, dominant
    if dominant_proto == 1 and pkt_rate > 50:
        scores[4] += 6
    if dominant_proto == 1 and dom_ratio > 0.9 and rate_multiplier > 3:
        scores[4] += 4

    # ─── DNS Amplification ───
    # UDP port 53, large response packets, many sources
    dns_ratio = dns_traffic / max(total, 1)
    if dns_ratio > 0.5 and avg_size > 500:
        scores[5] += 6
    if dominant_proto == 17 and dns_traffic > 30 and avg_size > 400 and unique_src > 10:
        scores[5] += 4

    # General attack gate: if no scores high, return Normal
    best_class = max(scores, key=scores.get)
    if scores[best_class] >= DETECTION["heuristic_attack_threshold"]:
        return best_class

    return 0  # Normal


# ─── Detection System ─────────────────────────────────────────────────────────

class DetectionSystem:
    """
    Core live detection engine. Runs in a background thread.
    Emits events via a callback dict for the dashboard to consume.
    """

    def __init__(self, socketio, device: str = "cpu"):
        self.socketio   = socketio
        self.device     = device
        self.model: Optional[DDoSDetectionGNN] = None
        self.optimizer  = None

        self.is_running = False
        self.is_paused  = False
        self.interface  = None

        self.training_buffer: deque = deque(maxlen=200)
        self.training_enabled       = True
        self.best_loss              = float("inf")

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Baseline learning for normal traffic
        self.baseline_stats = {
            "avg_packet_rate": 0.0,
            "avg_unique_ips": 0.0,
            "avg_packet_size": 0.0,
            "samples": 0,
        }
        self.baseline_learning_mode = True
        self.baseline_history = deque(maxlen=10)  # Rolling window for stability check
        
        # Attack confirmation window (temporal smoothing)
        self.recent_predictions: deque = deque(maxlen=DETECTION["attack_confirmation_window"])

        self.stats = {
            "total_packets":    0,
            "total_iterations": 0,
            "training_updates": 0,
            "current_confidence": 0.0,
            "current_label":    "Normal",
            "current_class_id": 0,
            "class_counts":     {v: 0 for v in ATTACK_CLASSES.values()},
            "suspicious_detections": 0,
            "confirmed_attacks": 0,
            "start_time":       None,
            "loaded_model":     "None",
            "network_speed_bps": 0,
            "packet_rate":      0.0,
        }

        self.detection_history: deque = deque(maxlen=100)
        self.attacker_ips:      Dict[str, dict] = {}
        self._monitor_thread:   Optional[threading.Thread] = None
        self._last_bytes        = 0
        self._last_speed_time   = time.time()

    # ── Model management ──────────────────────────────────────────────────────

    def load_model(self, filepath: str) -> Tuple[bool, str]:
        if self.is_running:
            return False, "Stop monitoring before loading a model."
        try:
            m, opt_state, stats = load_checkpoint(filepath, self.device)
            self.model = m
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            if opt_state:
                self.optimizer.load_state_dict(opt_state)
            self.stats["loaded_model"] = os.path.basename(filepath)
            msg = f"Loaded {os.path.basename(filepath)}"
            self._emit_log(msg, "success")
            log.info(msg)
            return True, msg
        except Exception as e:
            log.error(f"Model load failed: {e}")
            return False, str(e)

    def _ensure_model(self):
        """Initialise a fresh model if none loaded."""
        if self.model is None:
            self.model = DDoSDetectionGNN().to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            # Neutral classifier bias
            with torch.no_grad():
                self.model.classifier[-1].bias.zero_()
            self.stats["loaded_model"] = "Fresh (untrained)"
            self._emit_log("⚠ No model loaded — starting fresh.", "warning")

    # ── Packet capture ────────────────────────────────────────────────────────

    def capture_packets(self, duration: float = 5.0) -> Tuple[pd.DataFrame, float, float]:
        from scapy.all import sniff, IP, TCP, UDP, ICMP, conf

        conf.use_pcap = True
        buffer = []
        t_start = time.time()

        def on_packet(pkt):
            try:
                if IP not in pkt:
                    return
                entry = {
                    "src_ip":      pkt[IP].src,
                    "dst_ip":      pkt[IP].dst,
                    "protocol":    pkt[IP].proto,
                    "packet_size": len(pkt),
                    "src_port":    0,
                    "dst_port":    0,
                    "timestamp":   time.time(),
                    "tcp_flags":   0,
                }
                if TCP in pkt:
                    entry["src_port"]  = pkt[TCP].sport
                    entry["dst_port"]  = pkt[TCP].dport
                    entry["tcp_flags"] = int(pkt[TCP].flags)
                elif UDP in pkt:
                    entry["src_port"] = pkt[UDP].sport
                    entry["dst_port"] = pkt[UDP].dport
                buffer.append(entry)
            except Exception:
                pass

        try:
            sniff(
                iface=self.interface,
                prn=on_packet,
                timeout=duration,
                store=False,
            )
        except Exception as e:
            log.error(f"Capture error: {e}")
            self._emit_log(f"Capture error: {e}", "error")

        t_end = time.time()
        return pd.DataFrame(buffer), t_start, t_end

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(self, graph: Data) -> Tuple[int, str, float, dict]:
        self.model.eval()
        graph = graph.to(self.device)
        if not hasattr(graph, "batch") or graph.batch is None:
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
        pred, label, conf, probs = self.model.predict(graph, self.device)
        return pred, label, conf, probs

    # ── Incremental training ──────────────────────────────────────────────────

    def incremental_train(self, graph: Data, label: int):
        if not self.training_enabled or graph is None:
            return None
        graph = graph.to(self.device)
        graph.y = torch.tensor([label], dtype=torch.long, device=self.device)
        self.training_buffer.append(graph)

        if len(self.training_buffer) < 5:
            return None

        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        for g in self.training_buffer:
            if not hasattr(g, "batch") or g.batch is None:
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=self.device)
            out  = self.model(g)
            loss = F.cross_entropy(out, g.y)
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        avg_loss = total_loss / len(self.training_buffer)
        self.stats["training_updates"] += 1

        self._emit_log(f"Training update — loss: {avg_loss:.4f}", "info")
        self.socketio.emit("training_update", {"loss": round(avg_loss, 4), "iteration": self.stats["total_iterations"]})
        return avg_loss

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str):
        path = os.path.join(MODELS_DIR, filename)
        save_checkpoint(self.model, self.optimizer, self.stats, path)
        rotate_checkpoints(MODELS_DIR, CHECKPOINTING["max_checkpoints"])

    def _maybe_save_best(self, loss: Optional[float]):
        if loss is not None and loss < self.best_loss:
            self.best_loss = loss
            save_checkpoint(self.model, self.optimizer, self.stats, CHECKPOINTING["best_model_file"])
            self._emit_log(f"✓ New best model saved (loss {loss:.4f})", "success")

    # ── Attacker IP tracking ──────────────────────────────────────────────────

    def _update_attacker_ips(self, packets_df: pd.DataFrame, attack_class: int):
        if attack_class == 0 or packets_df is None or len(packets_df) == 0:
            return

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            my_ip = s.getsockname()[0]
            s.close()
        except Exception:
            my_ip = None

        attackers = packets_df[packets_df["dst_ip"] == my_ip]["src_ip"].value_counts()
        for ip, count in attackers.items():
            if ip == my_ip:
                continue
            if ip not in self.attacker_ips:
                self.attacker_ips[ip] = {
                    "ip": ip, "count": 0,
                    "attack_type": ATTACK_CLASSES[attack_class],
                    "first_seen": datetime.now().strftime("%H:%M:%S"),
                    "last_seen":  datetime.now().strftime("%H:%M:%S"),
                    "blocked": False,
                }
            self.attacker_ips[ip]["count"] += int(count)
            self.attacker_ips[ip]["last_seen"] = datetime.now().strftime("%H:%M:%S")
            self.attacker_ips[ip]["attack_type"] = ATTACK_CLASSES[attack_class]

        self.socketio.emit("attacker_ips_update", list(self.attacker_ips.values()))

    # ── Main monitoring loop ──────────────────────────────────────────────────

    def _monitoring_loop(self, capture_interval: float, train_every: int):
        self.stats["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        iteration = 0
        self._emit_log("Monitoring started", "success")
        log.info("Monitoring started")

        while self.is_running:
            if self.is_paused:
                time.sleep(0.5)
                continue

            iteration += 1
            self.stats["total_iterations"] = iteration
            
            # Capture packets first
            if iteration == 1:
                self._emit_log(f"Starting packet capture on interface: {self.interface}", "info")
            self._emit_log(f"Capturing packets (iteration {iteration})…", "info")

            packets_df, t_start, t_end = self.capture_packets(capture_interval)
            n_packets = len(packets_df)
            self.stats["total_packets"] += n_packets

            # Network speed (bytes/sec)
            if n_packets > 0:
                total_bytes = packets_df["packet_size"].sum()
                elapsed = max(t_end - t_start, 1e-3)
                self.stats["network_speed_bps"] = int(total_bytes / elapsed)
                self.stats["packet_rate"] = round(n_packets / elapsed, 1)
                
                # Update baseline during learning phase
                if self.baseline_learning_mode:
                    unique_ips = packets_df["src_ip"].nunique() + packets_df["dst_ip"].nunique()
                    avg_size = packets_df["packet_size"].mean()
                    n = self.baseline_stats["samples"]
                    self.baseline_stats["avg_packet_rate"] = (self.baseline_stats["avg_packet_rate"] * n + self.stats["packet_rate"]) / (n + 1)
                    self.baseline_stats["avg_unique_ips"] = (self.baseline_stats["avg_unique_ips"] * n + unique_ips) / (n + 1)
                    self.baseline_stats["avg_packet_size"] = (self.baseline_stats["avg_packet_size"] * n + avg_size) / (n + 1)
                    self.baseline_stats["samples"] += 1
                    
                    # Add to history for stability check
                    self.baseline_history.append({
                        "rate": self.stats["packet_rate"],
                        "ips": unique_ips,
                    })
                    
                    # Check baseline stability after updating
                    if iteration <= 5:
                        self._emit_log(f"Learning baseline [{iteration}/5 minimum]...", "info")
                    elif len(self.baseline_history) >= 5:
                        rates = [h["rate"] for h in self.baseline_history]
                        ips = [h["ips"] for h in self.baseline_history]
                        rate_std = np.std(rates)
                        ip_std = np.std(ips)
                        
                        # Stability thresholds: std < 20% of mean
                        rate_mean = np.mean(rates)
                        ip_mean = np.mean(ips)
                        rate_stable = rate_std < (rate_mean * 0.2) if rate_mean > 0 else False
                        ip_stable = ip_std < (ip_mean * 0.2) if ip_mean > 0 else False
                        
                        if rate_stable and ip_stable:
                            self.baseline_learning_mode = False
                            self._emit_log(f"✓ Baseline stabilized after {iteration} iterations: {self.baseline_stats['avg_packet_rate']:.1f} pkt/s, {self.baseline_stats['avg_unique_ips']:.0f} IPs", "success")
                        else:
                            self._emit_log(f"Learning baseline [{iteration}] - waiting for stability...", "info")
                    else:
                        self._emit_log(f"Learning baseline [{iteration}]...", "info")
            else:
                self.stats["network_speed_bps"] = 0
                self.stats["packet_rate"] = 0.0

            if n_packets < DETECTION["min_packets_for_detection"]:
                self._emit_log(f"⚠ Too few packets ({n_packets}) — skipping iteration", "warning")
                self._emit_stats()
                # Still check baseline stability even with low traffic
                if self.baseline_learning_mode and iteration > 5:
                    self.baseline_learning_mode = False
                    self._emit_log(f"✓ Baseline complete after {iteration} iterations (low traffic)", "success")
                continue

            graph = build_graph(packets_df, t_start, t_end)
            if graph is None:
                self._emit_log("⚠ Could not build graph", "warning")
                continue

            # GNN prediction
            pred_class, pred_label, gnn_confidence, probs = self.detect(graph)
            
            # Heuristic label for training supervision (now baseline-aware)
            actual_duration = max(t_end - t_start, 1e-3)
            h_label = heuristic_label(packets_df, self.baseline_stats if not self.baseline_learning_mode else None, actual_duration)

            # During baseline learning, force label to Normal
            if self.baseline_learning_mode:
                display_class = 0
                display_label = "Normal"
                attack_confidence = 0.0
            else:
                # Use GNN prediction as primary, heuristic as training signal
                display_class = pred_class
                display_label = pred_label
                # Attack confidence = GNN confidence for attack classes
                if pred_class == 0:
                    attack_confidence = 0.0  # Normal traffic
                else:
                    attack_confidence = round(gnn_confidence, 4)  # Use GNN confidence directly

            # Temporal smoothing: require N consecutive attack detections
            self.recent_predictions.append(display_class)
            confirmed_attack = (
                not self.baseline_learning_mode and
                len(self.recent_predictions) == DETECTION["attack_confirmation_window"] and
                all(c != 0 for c in self.recent_predictions)  # all recent windows are attacks
            )

            self.stats["current_confidence"] = attack_confidence
            self.stats["current_label"]       = display_label
            self.stats["current_class_id"]    = display_class
            self.stats["class_counts"][display_label] = \
                self.stats["class_counts"].get(display_label, 0) + 1

            # Detection event
            is_attack = confirmed_attack  # only trigger on confirmed attacks
            detection_event = {
                "timestamp":   datetime.now().strftime("%H:%M:%S"),
                "is_attack":   is_attack,
                "class_id":    display_class,
                "label":       display_label,
                "confidence":  attack_confidence,
                "packet_count": n_packets,
                "node_count":  graph.x.size(0),
                "probs":       probs,
                "capture_interval": capture_interval,
                "window_duration": round(actual_duration, 3),
            }
            self.detection_history.append(detection_event)
            
            # Emit updates less frequently (every iteration, but throttled by frontend)
            if iteration % 2 == 0 or is_attack:  # emit every 2 iterations or immediately on attack
                self.socketio.emit("detection_update", detection_event)
                self.socketio.emit("chart_update", list(self.detection_history)[-20:])  # only last 20 points

            if is_attack:
                self.stats["confirmed_attacks"] += 1
                self._emit_log(f"⚠ CONFIRMED {display_label} — {attack_confidence:.1%} attack probability", "danger")
                self._update_attacker_ips(packets_df, display_class)
                from alerting import send_attack_alert
                top_ips = list(self.attacker_ips.keys())[:3]
                send_attack_alert(display_class, attack_confidence, top_ips)
            elif display_class != 0 and not confirmed_attack:
                self.stats["suspicious_detections"] += 1
                self._emit_log(f"Possible {display_label} (unconfirmed) — {attack_confidence:.1%}", "warning")
            else:
                if iteration % 5 == 0:  # log normal traffic less frequently
                    self._emit_log(f"Normal traffic — {attack_confidence:.1%} attack probability", "success")

            # Incremental training (skip during baseline learning)
            loss = None
            if not self.baseline_learning_mode and iteration % train_every == 0:
                loss = self.incremental_train(graph, h_label)
                self._maybe_save_best(loss)

            # Checkpointing
            if iteration % CHECKPOINTING["save_every_n_iter"] == 0:
                ckpt_name = f"checkpoint_iter_{iteration}.pth"
                self._save_checkpoint(ckpt_name)
                self._save_checkpoint("latest_model.pth")
                self._emit_log(f"Checkpoint saved (iter {iteration})", "info")

            self._emit_stats()

        self._emit_log("⏹ Monitoring stopped", "info")
        log.info("Monitoring stopped")

    # ── Public control API ────────────────────────────────────────────────────

    def start(self, interface: str, capture_interval: float = 5, train_every: int = 3) -> bool:
        if self.is_running:
            return False
        self._ensure_model()
        self.interface  = interface
        self.is_running = True
        self.is_paused  = False
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(capture_interval, train_every),
            daemon=True,
        )
        self._monitor_thread.start()
        return True

    def stop(self) -> bool:
        if not self.is_running:
            return False
        self.is_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)  # Increased timeout for graceful shutdown
        if self.model and self.stats["total_iterations"] > 0:
            try:
                self._save_checkpoint("latest_model.pth")
                log.info("Model saved on shutdown")
            except Exception as e:
                log.error(f"Failed to save model on shutdown: {e}")
        return True

    def _setup_signal_handlers(self):
        """Setup graceful shutdown on CTRL+C or system signals."""
        def signal_handler(signum, frame):
            log.info(f"Received signal {signum}, shutting down gracefully...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def pause(self) -> bool:
        self.is_paused = True
        self._emit_log("⏸ Monitoring paused", "info")
        return True

    def resume(self) -> bool:
        self.is_paused = False
        self._emit_log("▶ Monitoring resumed", "info")
        return True

    def toggle_training(self) -> bool:
        self.training_enabled = not self.training_enabled
        state = "ON" if self.training_enabled else "OFF"
        self._emit_log(f"Training toggled {state}", "info")
        return self.training_enabled

    def manual_save(self) -> str:
        self._ensure_model()
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"manual_save_{ts}.pth"
        path = os.path.join(MODELS_DIR, name)
        save_checkpoint(self.model, self.optimizer, self.stats, path)
        self._emit_log(f"Manual save: {name}", "success")
        return path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _emit_log(self, msg: str, level: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.socketio.emit("log_message", {"timestamp": ts, "message": msg, "level": level})
        log.info(f"[{level.upper()}] {msg}")

    def _emit_stats(self):
        self.socketio.emit("stats_update", self.stats)
