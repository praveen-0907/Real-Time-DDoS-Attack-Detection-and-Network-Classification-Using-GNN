"""
pcap_analyzer.py — Offline PCAP File Analysis
==============================================
Analyses uploaded PCAP files using the trained GNN model.
Returns per-window results: class, confidence, attacker IPs.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Callable

from config import ATTACK_CLASSES, ATTACK_COLORS, PCAP
from model import DDoSDetectionGNN, load_checkpoint
from detector import build_graph, heuristic_label
from logger import get_logger

log = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pcap", ".pcapng", ".cap"}


def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def parse_pcap(filepath: str, max_packets: int = None) -> Optional[pd.DataFrame]:
    """Read a PCAP file into a DataFrame using scapy.
    
    Args:
        filepath: Path to PCAP file
        max_packets: Maximum packets to read (None = all). Use for memory efficiency.
    """
    try:
        from scapy.all import rdpcap, IP, TCP, UDP, ICMP, PcapReader
    except ImportError:
        log.error("Scapy not installed.")
        return None

    rows = []
    packet_count = 0
    
    try:
        # Use PcapReader for memory-efficient streaming
        with PcapReader(filepath) as pcap_reader:
            for pkt in pcap_reader:
                try:
                    if IP not in pkt:
                        continue
                    ts = float(pkt.time)
                    entry = {
                        "src_ip":      pkt[IP].src,
                        "dst_ip":      pkt[IP].dst,
                        "protocol":    pkt[IP].proto,
                        "packet_size": len(pkt),
                        "src_port":    0,
                        "dst_port":    0,
                        "timestamp":   ts,
                        "tcp_flags":   0,
                    }
                    if TCP in pkt:
                        entry["src_port"] = pkt[TCP].sport
                        entry["dst_port"] = pkt[TCP].dport
                        entry["tcp_flags"] = int(pkt[TCP].flags)
                    elif UDP in pkt:
                        entry["src_port"] = pkt[UDP].sport
                        entry["dst_port"] = pkt[UDP].dport
                    rows.append(entry)
                    
                    packet_count += 1
                    if max_packets and packet_count >= max_packets:
                        log.info(f"Reached max_packets limit ({max_packets})")
                        break
                except Exception:
                    continue
    except Exception as e:
        log.error(f"Failed to read PCAP {filepath}: {e}")
        return None

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info(f"Parsed PCAP {filepath}: {len(df)} IP packets")
    return df


def analyse_pcap(
    filepath: str,
    model: DDoSDetectionGNN,
    device: str = "cpu",
    window_size: int = PCAP["graph_window_size"],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """
    Slice the PCAP into time windows, run GNN inference on each, return results.

    Returns:
        {
          "total_packets": int,
          "duration_sec": float,
          "windows": [
            {
              "window": int,
              "t_start": float, "t_end": float,
              "packet_count": int,
              "pred_class": int,
              "pred_label": str,
              "confidence": float,
              "heuristic_class": int,
              "probs": {label: float},
              "top_src_ips": [str],
            }, ...
          ],
          "summary": {
            "class_counts": {label: int},
            "attack_ratio": float,
            "top_attacker_ips": [str],
          }
        }
    """
    df = parse_pcap(filepath)
    if df is None or len(df) == 0:
        return {"error": "Failed to parse PCAP or no IP packets found."}

    total_packets = len(df)
    duration_sec  = float(df["timestamp"].max() - df["timestamp"].min())

    # Slide window over packets (include partial last window)
    windows_result = []
    n_windows      = (total_packets + window_size - 1) // window_size  # Ceiling division to include partial window
    class_counts   = {v: 0 for v in ATTACK_CLASSES.values()}
    attacker_counts: Dict[str, int] = {}

    model.eval()

    for w_idx in range(n_windows):
        start_i = w_idx * window_size
        end_i   = min(start_i + window_size, total_packets)
        window  = df.iloc[start_i:end_i]

        t_start = float(window["timestamp"].min())
        t_end   = float(window["timestamp"].max())
        window_duration = max(t_end - t_start, 1e-3)

        graph = build_graph(window, t_start, t_end)
        h_cls = heuristic_label(window, None, window_duration)  # Pass actual window duration

        pred_class, pred_label, confidence, probs = 0, "Normal", 0.0, {}

        if graph is not None:
            graph = graph.to(device)
            if not hasattr(graph, "batch") or graph.batch is None:
                graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
            try:
                pred_class, pred_label, confidence, probs = model.predict(graph, device)
            except Exception as e:
                log.warning(f"Inference failed on window {w_idx}: {e}")

        # Use GNN prediction as primary, heuristic for comparison
        display_class = pred_class
        display_label = pred_label
        class_counts[display_label] = class_counts.get(display_label, 0) + 1

        # Top source IPs
        top_srcs = window["src_ip"].value_counts().head(5).index.tolist()
        if display_class != 0:
            for ip in top_srcs:
                attacker_counts[ip] = attacker_counts.get(ip, 0) + 1

        windows_result.append({
            "window":         w_idx + 1,
            "t_start":        round(t_start, 2),
            "t_end":          round(t_end, 2),
            "packet_count":   len(window),
            "pred_class":     display_class,
            "pred_label":     display_label,
            "confidence":     round(confidence, 4),
            "heuristic_class": h_cls,
            "heuristic_label": ATTACK_CLASSES[h_cls],
            "agreement":      (pred_class == h_cls),
            "probs":           probs,
            "top_src_ips":    top_srcs,
            "color":           ATTACK_COLORS.get(display_class, "#22c55e"),
        })

        if progress_cb:
            progress_cb(w_idx + 1, n_windows)

    total_attack = sum(v for k, v in class_counts.items() if k != "Normal")
    attack_ratio = total_attack / max(len(windows_result), 1)

    top_attackers = sorted(attacker_counts, key=attacker_counts.get, reverse=True)[:10]

    return {
        "filename":      os.path.basename(filepath),
        "total_packets": total_packets,
        "duration_sec":  round(duration_sec, 2),
        "n_windows":     len(windows_result),
        "windows":       windows_result,
        "summary": {
            "class_counts":      class_counts,
            "attack_ratio":      round(attack_ratio, 4),
            "top_attacker_ips":  top_attackers,
        },
    }
