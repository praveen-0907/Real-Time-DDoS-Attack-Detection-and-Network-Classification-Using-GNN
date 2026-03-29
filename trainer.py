"""
trainer.py — Offline Pre-training on CICIDS2017 / CICIDS2018
=============================================================
Optimised for:
  - CUDA (RTX 3050 / any NVIDIA GPU)
  - Mixed precision training (FP16 via torch.cuda.amp)
  - Parallel data loading (num_workers > 0)
  - Incremental CSV caching (only reprocess new files)

Usage:
    python trainer.py                        # auto-detects GPU, uses config defaults
    python trainer.py --epochs 50            # override epochs
    python trainer.py --batch 128            # override batch size
    python trainer.py --rebuild-cache        # force re-process CSVs even if cache exists
    python trainer.py --device cpu           # force CPU if needed

Place CICIDS CSV files inside the data/ folder before running.
Download from: https://www.unb.ca/cic/datasets/ids-2017.html
"""

import os
import sys
import glob
import time
import hashlib
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Optional

from config import (
    CICIDS_LABEL_MAP, TRAINING, CHECKPOINTING, MODEL,
    DATA_DIR, PROCESSED_DIR, MODELS_DIR, ATTACK_CLASSES, BASE_DIR
)
from model import DDoSDetectionGNN, save_checkpoint, rotate_checkpoints, calibrate_temperature
from logger import get_logger

log = get_logger(__name__)

# ─── CICIDS Column Name Variants ─────────────────────────────────────────────
_LABEL_COLS   = ["Label", "label", " Label"]
_SRC_IP_COLS  = ["Source IP", "Src IP", " Source IP"]
_DST_IP_COLS  = ["Destination IP", "Dst IP", " Destination IP"]
_PROTO_COLS   = ["Protocol", "protocol", " Protocol"]
_PKT_LEN_COLS = [
    "Total Length of Fwd Packets", "Packet Length Mean",
    "Flow Bytes/s", " Total Length of Fwd Packets",
    "Total Fwd Packets", "Fwd Packet Length Mean",
]
_PORT_COLS      = ["Destination Port", " Destination Port", "Dst Port"]
_SYN_COLS       = ["SYN Flag Cnt", " SYN Flag Cnt", "SYN Flag Count"]
_ACK_COLS       = ["ACK Flag Cnt", " ACK Flag Cnt", "ACK Flag Count"]
_FIN_COLS       = ["FIN Flag Cnt", " FIN Flag Cnt", "FIN Flag Count"]
_RST_COLS       = ["RST Flag Cnt", " RST Flag Cnt", "RST Flag Count"]
_FWD_PKT_COLS   = ["Tot Fwd Pkts", " Tot Fwd Pkts", "Total Fwd Packets", "Fwd Packet Count"]
_BWD_PKT_COLS = ["Tot Bwd Pkts", " Tot Bwd Pkts", "Total Bwd Packets","Total Backward Packets", "Bwd Packet Count"]
_IAT_MEAN_COLS  = ["Flow IAT Mean", " Flow IAT Mean"]
_IAT_STD_COLS   = ["Flow IAT Std",  " Flow IAT Std"]
_DURATION_COLS  = ["Flow Duration", " Flow Duration"]
_PKT_RATE_COLS  = ["Flow Pkts/s",   " Flow Pkts/s",  "Flow Packets/s"]


def _pick_col(df: pd.DataFrame, candidates: list, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _csv_hash(filepath: str) -> str:
    """MD5 of first 64KB — fast fingerprint to detect changed files."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


# ─── CSV Loading ──────────────────────────────────────────────────────────────

def load_cicids_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Two-pass CSV loader — loads only required columns at parser level.
    Pass 1: header only to resolve actual column names (~zero RAM).
    Pass 2: usecols targeted load (~85% less RAM than full load).

    Extracts all columns needed to build 15 edge features.
    Missing columns (e.g. flag counts absent in some files) default to 0.
    """
    try:
        # ── Pass 1: header only ───────────────────────────────────────────────
        for enc in ["utf-8", "latin-1", "windows-1252"]:
            try:
                header_df = pd.read_csv(filepath, nrows=0, encoding=enc)
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            log.warning(f"Could not determine encoding for {filepath} — skipping")
            return None
        all_cols  = [c.strip() for c in header_df.columns]
        col_map   = {c.strip(): c for c in header_df.columns}

        def resolve(candidates):
            for c in candidates:
                if c in all_cols:
                    return col_map[c]
            return None

        label_col    = resolve(_LABEL_COLS)
        src_ip_col   = resolve(_SRC_IP_COLS)
        dst_ip_col   = resolve(_DST_IP_COLS)
        proto_col    = resolve(_PROTO_COLS)
        pkt_len_col  = resolve(_PKT_LEN_COLS)
        port_col     = resolve(_PORT_COLS)
        syn_col      = resolve(_SYN_COLS)
        ack_col      = resolve(_ACK_COLS)
        fin_col      = resolve(_FIN_COLS)
        rst_col      = resolve(_RST_COLS)
        fwd_pkt_col  = resolve(_FWD_PKT_COLS)
        bwd_pkt_col  = resolve(_BWD_PKT_COLS)
        iat_mean_col = resolve(_IAT_MEAN_COLS)
        iat_std_col  = resolve(_IAT_STD_COLS)
        duration_col = resolve(_DURATION_COLS)
        pkt_rate_col = resolve(_PKT_RATE_COLS)

        if label_col is None:
            log.warning(f"No label column in {filepath} — skipping")
            return None

        usecols = list({c for c in [
            label_col, src_ip_col, dst_ip_col, proto_col, pkt_len_col, port_col,
            syn_col, ack_col, fin_col, rst_col,
            fwd_pkt_col, bwd_pkt_col,
            iat_mean_col, iat_std_col,
            duration_col, pkt_rate_col,
        ] if c is not None})

        # ── Pass 2: load only required columns ────────────────────────────────
        df = pd.read_csv(filepath, usecols=usecols, low_memory=False, encoding=enc, on_bad_lines="skip")
        df.columns = [c.strip() for c in df.columns]

        # Re-resolve after strip
        label_col    = _pick_col(df, _LABEL_COLS)
        src_ip_col   = _pick_col(df, _SRC_IP_COLS)
        dst_ip_col   = _pick_col(df, _DST_IP_COLS)
        proto_col    = _pick_col(df, _PROTO_COLS)
        pkt_len_col  = _pick_col(df, _PKT_LEN_COLS)
        port_col     = _pick_col(df, _PORT_COLS)
        syn_col      = _pick_col(df, _SYN_COLS)
        ack_col      = _pick_col(df, _ACK_COLS)
        fin_col      = _pick_col(df, _FIN_COLS)
        rst_col      = _pick_col(df, _RST_COLS)
        fwd_pkt_col  = _pick_col(df, _FWD_PKT_COLS)
        bwd_pkt_col  = _pick_col(df, _BWD_PKT_COLS)
        iat_mean_col = _pick_col(df, _IAT_MEAN_COLS)
        iat_std_col  = _pick_col(df, _IAT_STD_COLS)
        duration_col = _pick_col(df, _DURATION_COLS)
        pkt_rate_col = _pick_col(df, _PKT_RATE_COLS)

        def _num(col, fill=0.0):
            if col and col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(fill)
            return pd.Series(fill, index=df.index, dtype=float)

        out = pd.DataFrame()
        out["label"]       = df[label_col].astype(str).str.strip()
        out["src_ip"]      = df[src_ip_col].astype(str) if src_ip_col else "0.0.0.0"
        out["dst_ip"]      = df[dst_ip_col].astype(str) if dst_ip_col else "0.0.0.0"
        out["protocol"]    = _num(proto_col,    0).astype(int)
        out["packet_size"] = _num(pkt_len_col,  64.0).clip(0, 65535)
        out["dst_port"]    = _num(port_col,     0).astype(int)
        out["syn_cnt"]     = _num(syn_col,      0.0)
        out["ack_cnt"]     = _num(ack_col,      0.0)
        out["fin_cnt"]     = _num(fin_col,      0.0)
        out["rst_cnt"]     = _num(rst_col,      0.0)
        out["fwd_pkts"]    = _num(fwd_pkt_col,  1.0).clip(lower=0)
        out["bwd_pkts"]    = _num(bwd_pkt_col,  1.0).clip(lower=0)
        out["iat_mean"]    = _num(iat_mean_col,  0.0).clip(lower=0)
        out["iat_std"]     = _num(iat_std_col,   0.0).clip(lower=0)
        out["duration"]    = _num(duration_col,  0.0).clip(lower=0)
        out["pkt_rate"]    = _num(pkt_rate_col,  0.0).clip(lower=0)
        out["timestamp"]   = 0.0
        out["class_id"]    = out["label"].map(CICIDS_LABEL_MAP).fillna(0).astype(int)

        # Clean Infinity/NaN (common in CICIDS2019 Flow Bytes/s)
        out.replace([float("inf"), float("-inf")], 0.0, inplace=True)
        out.fillna(0.0, inplace=True)
        out.dropna(subset=["src_ip", "dst_ip"], inplace=True)
        out.reset_index(drop=True, inplace=True)

        dist = out["class_id"].value_counts().to_dict()
        log.info(f"  Loaded {os.path.basename(filepath)}: {len(out):,} rows | {dist}")
        return out

    except Exception as e:
        log.error(f"Failed to load {filepath}: {e}")
        return None


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_graph_from_window(window_df: pd.DataFrame, label: int) -> Optional[Data]:
    """
    Convert a fixed-size flow window into a PyG graph with 15 edge features.
    Must match detector.py build_graph() feature layout exactly.

    Edge features:
      [0]  packet count        [1]  total bytes         [2]  mean pkt size
      [3]  std pkt size        [4]  max pkt size        [5]  duration ms
      [6]  packet rate         [7]  SYN flag count      [8]  ACK flag count
      [9]  FIN flag count      [10] RST flag count      [11] fwd/bwd ratio
      [12] IAT mean            [13] IAT std             [14] unique src IPs
    """
    if len(window_df) == 0:
        return None

    ip_to_id: dict = {}
    counter = 0

    def get_id(ip):
        nonlocal counter
        if ip not in ip_to_id:
            ip_to_id[ip] = counter
            counter += 1
        return ip_to_id[ip]

    edge_dict = defaultdict(lambda: {
        "count": 0, "bytes": 0.0, "sizes": [],
        "syn": 0.0, "ack": 0.0, "fin": 0.0, "rst": 0.0,
        "fwd": 0, "bwd": 0,
        "iat_mean": 0.0, "iat_std": 0.0,
        "duration": 0.0, "pkt_rate": 0.0,
        "src_ips": set(),
    })

    for _, r in window_df.iterrows():
        src = get_id(str(r["src_ip"]))
        dst = get_id(str(r["dst_ip"]))
        sz  = float(r["packet_size"])
        e   = (src, dst)

        edge_dict[e]["count"]    += 1
        edge_dict[e]["bytes"]    += sz
        edge_dict[e]["sizes"].append(sz)
        edge_dict[e]["src_ips"].add(str(r["src_ip"]))

        # Per-row flag counts from CSV columns (already aggregated at flow level)
        edge_dict[e]["syn"]      += float(r.get("syn_cnt",  0))
        edge_dict[e]["ack"]      += float(r.get("ack_cnt",  0))
        edge_dict[e]["fin"]      += float(r.get("fin_cnt",  0))
        edge_dict[e]["rst"]      += float(r.get("rst_cnt",  0))

        # Fwd/bwd packet counts
        edge_dict[e]["fwd"]      += int(r.get("fwd_pkts",  1))
        edge_dict[e]["bwd"]      += int(r.get("bwd_pkts",  1))

        # IAT and duration — accumulate then average
        edge_dict[e]["iat_mean"] += float(r.get("iat_mean", 0))
        edge_dict[e]["iat_std"]  += float(r.get("iat_std",  0))
        edge_dict[e]["duration"] += float(r.get("duration", 0))
        edge_dict[e]["pkt_rate"] += float(r.get("pkt_rate", 0))

    if not edge_dict:
        return None

    edge_list, edge_feats = [], []
    for (s, d), v in edge_dict.items():
        sizes = v["sizes"]
        n     = max(v["count"], 1)

        fwd_bwd_ratio = v["fwd"] / max(v["bwd"], 1)
        iat_mean_avg  = v["iat_mean"]  / n
        iat_std_avg   = v["iat_std"]   / n
        dur_avg       = v["duration"]  / n
        rate_avg      = v["pkt_rate"]  / n

        edge_list.append([s, d])
        edge_feats.append([
            v["count"],                                               # 0
            v["bytes"],                                               # 1
            float(np.mean(sizes)),                                    # 2
            float(np.std(sizes)) if len(sizes) > 1 else 0.0,         # 3
            float(np.max(sizes)),                                     # 4
            dur_avg,                                                  # 5
            rate_avg,                                                 # 6
            v["syn"],                                                 # 7
            v["ack"],                                                 # 8
            v["fin"],                                                 # 9
            v["rst"],                                                 # 10
            fwd_bwd_ratio,                                            # 11
            iat_mean_avg,                                             # 12
            iat_std_avg,                                              # 13
            float(len(v["src_ips"])),                                 # 14
        ])

    x  = torch.ones((counter, 1), dtype=torch.float)
    ei = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    ea = torch.tensor(edge_feats, dtype=torch.float)
    ea = torch.nan_to_num(ea, nan=0.0, posinf=0.0, neginf=0.0)
    ea = torch.log1p(torch.abs(ea))   

    data       = Data(x=x, edge_index=ei, edge_attr=ea)
    data.y     = torch.tensor([label], dtype=torch.long)
    data.batch = torch.zeros(counter, dtype=torch.long)
    return data


# ─── Dataset Builder with Incremental Cache ──────────────────────────────────

def build_dataset(
    csv_files:     List[str],
    window_size:   int  = TRAINING["graph_window_size"],
    max_per_class: int  = 3000,  # increased from 2000
    rebuild:       bool = False,
) -> List[Data]:
    """
    Build a balanced graph dataset from CICIDS CSVs.

    Incremental caching strategy:
      - Each CSV file gets a per-file shard .pkl keyed by its MD5 hash.
      - Only new or changed CSVs are reprocessed.
      - The merged dataset is also cached separately.
      - --rebuild-cache forces full reprocessing.
      
    Class balancing:
      - Normal class gets 3x more samples (9000) to reflect real-world distribution
      - Attack classes capped at 3000 each for balanced learning
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    shard_dir   = os.path.join(PROCESSED_DIR, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    merged_path = os.path.join(PROCESSED_DIR, "cicids_graph_dataset.pkl")

    if not rebuild and os.path.exists(merged_path):
        log.info(f"Loading merged cache from {merged_path}")
        with open(merged_path, "rb") as f:
            dataset = pickle.load(f)
        log.info(f"  {len(dataset):,} graphs loaded from cache")
        return dataset

    all_graphs: List[Data]   = []
    class_counts              = defaultdict(int)
    total_files               = len(csv_files)
    
    # Class-specific caps: Normal gets 3x more samples
    class_caps = {0: max_per_class * 3}  # Normal: 9000
    for i in range(1, 6):  # Attacks: 3000 each
        class_caps[i] = max_per_class

    for file_idx, filepath in enumerate(csv_files):
        fname      = os.path.basename(filepath)
        fhash      = _csv_hash(filepath)
        shard_path = os.path.join(shard_dir, f"{fhash}.pkl")

        print(f"\n[{file_idx+1}/{total_files}] {fname}")

        if not rebuild and os.path.exists(shard_path):
            print(f"  ↳ Cache hit — loading shard")
            with open(shard_path, "rb") as f:
                shard_graphs = pickle.load(f)
            print(f"  ↳ {len(shard_graphs):,} graphs from cache")
        else:
            print(f"  ↳ Processing CSV (first time)…")
            t0 = time.time()
            df = load_cicids_csv(filepath)
            if df is None:
                continue

            shard_graphs: List[Data] = []
            step = window_size // 2   # 50% overlap → more samples per file

            for label_id in df["class_id"].unique():
                subset = df[df["class_id"] == label_id].reset_index(drop=True)
                existing = sum(1 for g in shard_graphs if int(g.y.item()) == int(label_id))
                cap = class_caps.get(int(label_id), max_per_class)
                
                for start in range(0, len(subset) - window_size + 1, step):
                    if existing >= cap:
                        break
                    window = subset.iloc[start: start + window_size]
                    graph  = build_graph_from_window(window, int(label_id))
                    if graph is not None:
                        shard_graphs.append(graph)
                        existing += 1
                        
            with open(shard_path, "wb") as f:
                pickle.dump(shard_graphs, f)
            print(f"  ↳ Built {len(shard_graphs):,} graphs in {time.time()-t0:.1f}s — shard cached")

        # Apply per-class cap when merging
        for g in shard_graphs:
            lbl = int(g.y.item())
            cap = class_caps.get(lbl, max_per_class)
            if class_counts[lbl] < cap:
                all_graphs.append(g)
                class_counts[lbl] += 1

    dist_str = " | ".join(
        f"{ATTACK_CLASSES.get(k, str(k))}: {v}"
        for k, v in sorted(class_counts.items())
    )
    print(f"\n✓ Dataset ready: {len(all_graphs):,} graphs total")
    print(f"  Distribution : {dist_str}\n")
    log.info(f"Dataset built: {len(all_graphs)} graphs | {dist_str}")

    with open(merged_path, "wb") as f:
        pickle.dump(all_graphs, f)

    return all_graphs


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_epoch(
    model:     DDoSDetectionGNN,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    use_amp:   bool,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                out  = model(batch)
                loss = F.cross_entropy(out, batch.y.squeeze())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(batch)
            loss = F.cross_entropy(out, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def eval_epoch(
    model:   DDoSDetectionGNN,
    loader:  DataLoader,
    device:  torch.device,
    use_amp: bool,
) -> Tuple[float, float, dict]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0
    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)

            if use_amp:
                with autocast():
                    out  = model(batch)
                    loss = F.cross_entropy(out, batch.y.squeeze())
            else:
                out  = model(batch)
                loss = F.cross_entropy(out, batch.y.squeeze())

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            lbls  = batch.y.squeeze()
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)

            for p, l in zip(preds.cpu().tolist(), lbls.cpu().tolist()):
                per_class_total[l]   += 1
                per_class_correct[l] += int(p == l)

    acc     = correct / max(total, 1)
    per_cls = {
        ATTACK_CLASSES.get(k, str(k)): round(per_class_correct[k] / max(per_class_total[k], 1), 3)
        for k in per_class_total
    }
    return total_loss / max(len(loader), 1), acc, per_cls


# ─── Main ─────────────────────────────────────────────────────────────────────

def pretrain(
    data_dir:   str  = DATA_DIR,
    epochs:     int  = TRAINING["epochs"],
    batch_size: int  = TRAINING["batch_size"],
    device_str: str  = "auto",
    rebuild:    bool = False,
):
    # ── Device ───────────────────────────────────────────────────────────────
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device  = torch.device(device_str)
    use_amp = (device.type == "cuda")

    print("\n" + "="*65)
    print("  GNNShield — Pre-Trainer")
    print("="*65)
    print(f"  Device     : {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if use_amp else ""))
    if use_amp:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM       : {vram:.1f} GB")
    print(f"  AMP (FP16) : {'✓ enabled' if use_amp else '✗ CPU mode'}")
    print(f"  Batch size : {batch_size}")
    print(f"  Workers    : {TRAINING['num_workers']}")
    print(f"  Epochs     : {epochs}")
    print("="*65)

    # ── CSV files ─────────────────────────────────────────────────────────────
    # Search in data/ and cicids2019/ folders
    csv_files = sorted(
        glob.glob(os.path.join(data_dir, "*.csv")) +
        glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True) +
        glob.glob(os.path.join(BASE_DIR, "cicids2019", "*.csv"))
    )
    if not csv_files:
        print(f"\n⚠  No CSV files found in '{data_dir}'.")
        print("   Download CICIDS2017 from:")
        print("   https://www.unb.ca/cic/datasets/ids-2017.html\n")
        return

    print(f"\n  CSV files found: {len(csv_files)}")
    for f in csv_files:
        print(f"    • {os.path.basename(f)}  ({os.path.getsize(f)/1e6:.0f} MB)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\n[1/3] Building graph dataset…")
    t0      = time.time()
    dataset = build_dataset(csv_files, rebuild=rebuild)
    print(f"  ↳ Ready in {time.time()-t0:.1f}s")

    if not dataset:
        print("\n✗ Empty dataset — check CSV files.")
        return

    # ── Splits + Loaders ──────────────────────────────────────────────────────
    val_size   = int(len(dataset) * TRAINING["val_split"])
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    loader_kw = dict(
        batch_size         = batch_size,
        num_workers        = TRAINING["num_workers"],
        pin_memory         = use_amp,
        persistent_workers = (TRAINING["num_workers"] > 0),
    )
    train_loader = DataLoader(train_set, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kw)
    print(f"\n  Train: {train_size:,}  |  Val: {val_size:,}  |  Batches/epoch: {len(train_loader):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/3] Initialising model…")
    model     = DDoSDetectionGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6,
    )
    scaler    = GradScaler(enabled=use_amp)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[3/4] Training…\n")
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val Acc':>8}  {'LR':>10}  {'Time':>6}")
    print("-" * 65)

    best_val_loss  = float("inf")
    patience_count = 0
    stats          = {}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss              = train_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        val_loss, val_acc, pcls = eval_epoch(model, val_loader, device, use_amp)
        scheduler.step(val_loss)

        lr      = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>10.4f}  {val_acc:>7.2%}  {lr:>10.2e}  {elapsed:>5.1f}s")

        if epoch % 10 == 0:
            cls_str = "  ".join(f"{k}: {v:.0%}" for k, v in pcls.items())
            print(f"         Per-class → {cls_str}")

        stats = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_checkpoint(model, optimizer, stats, CHECKPOINTING["best_model_file"])
            print(f"         ★ Best model saved (val_loss {val_loss:.4f})")
        else:
            patience_count += 1

        if epoch % 10 == 0:
            ckpt = os.path.join(MODELS_DIR, f"pretrain_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, stats, ckpt)
            rotate_checkpoints(MODELS_DIR, max_keep=5)

        if use_amp and epoch % 5 == 0:
            used  = torch.cuda.memory_allocated(0) / 1e9
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"         GPU mem: {used:.2f}/{total_vram:.1f} GB")

        if patience_count >= TRAINING["early_stop_patience"]:
            print(f"\n  Early stopping triggered (no improvement for {patience_count} epochs).")
            break
    
    stats["best_val_loss"] = best_val_loss
    stats["best_val_acc"]  = val_acc  # Use current val_acc since best_val_acc not tracked
    
    # Calibrate temperature on validation set for better confidence estimates
    print("\n[4/4] Calibrating confidence (temperature scaling)...")
    try:
        optimal_temp = calibrate_temperature(model, val_loader, device, max_iter=50)
        print(f"  ✓ Optimal temperature: {optimal_temp:.4f}")
        stats["temperature"] = optimal_temp
    except Exception as e:
        log.warning(f"Temperature calibration failed: {e}")
        print(f"  ⚠ Calibration skipped: {e}")
    
    save_checkpoint(model, optimizer, stats, CHECKPOINTING["latest_model_file"])

    print("\n" + "="*65)
    print(f"  ✓ Done.  Best val_loss: {best_val_loss:.4f}  |  Val acc: {stats.get('val_acc',0):.2%}")
    print(f"  Saved to: {MODELS_DIR}")
    print("="*65 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-train GNNShield on CICIDS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",          default=DATA_DIR,               help="Folder with CICIDS CSV files")
    parser.add_argument("--epochs",        default=TRAINING["epochs"],     type=int)
    parser.add_argument("--batch",         default=TRAINING["batch_size"], type=int,
                        help="Batch size. Recommended: 128 for RTX 3050")
    parser.add_argument("--device",        default="auto",
                        help="auto | cuda | cpu")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Reprocess all CSVs even if cache exists")
    args = parser.parse_args()

    pretrain(
        data_dir   = args.data,
        epochs     = args.epochs,
        batch_size = args.batch,
        device_str = args.device,
        rebuild    = args.rebuild_cache,
    )
