"""
model.py — GNN Model Architecture for DDoS Detection
=====================================================
6-class classifier: Normal + SYN Flood + UDP Flood + HTTP Flood + ICMP Flood + DNS Amplification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
import os
from config import MODEL, ATTACK_CLASSES, CHECKPOINTING


class DDoSDetectionGNN(nn.Module):
    """
    Graph Neural Network for multi-class DDoS attack detection.

    Architecture:
        - Edge encoder: raw edge features → hidden_dim embeddings
        - Node encoder: degree feature → hidden_dim
        - 3x GCNConv layers with BatchNorm + ReLU + Dropout
        - Graph readout: mean + max + sum pooling concatenated
        - MLP classifier → 6 classes
    """

    def __init__(
        self,
        edge_feature_dim: int = MODEL["edge_feature_dim"],
        hidden_dim: int = MODEL["hidden_dim"],
        num_layers: int = MODEL["num_layers"],
        dropout: float = MODEL["dropout"],
        num_classes: int = MODEL["num_classes"],
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Temperature scaling for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Node feature encoder (degree-based, single scalar input)
        self.node_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        )

        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Readout: mean + max + sum → hidden_dim * 3
        readout_dim = hidden_dim * 3

        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Encode node features
        x = self.node_encoder(x)

        # Inject edge information into node representations
        if edge_attr is not None and edge_attr.size(0) > 0:
            edge_emb = self.edge_encoder(edge_attr)
            src, dst = edge_index[0], edge_index[1]
            edge_contrib = torch.zeros_like(x)
            edge_contrib.index_add_(0, src, edge_emb)
            edge_contrib.index_add_(0, dst, edge_emb)
            x = x + edge_contrib

        # Graph convolutions
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if x.size(0) > 1:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level readout (mean + max + sum pooling)
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x_sum  = global_add_pool(x, batch)
        graph_emb = torch.cat([x_mean, x_max, x_sum], dim=1)

        return self.classifier(graph_emb)

    def predict(self, data: Data, device: str = "cpu"):
        """Return (class_index, class_name, confidence, all_probs_dict)."""
        self.eval()
        data = data.to(device)
        with torch.no_grad():
            if not hasattr(data, "batch") or data.batch is None:
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            logits = self(data)
            # Apply temperature scaling for calibrated confidence
            scaled_logits = logits / self.temperature
            probs  = F.softmax(scaled_logits, dim=1)[0]
            pred   = probs.argmax().item()
            conf   = probs[pred].item()
            probs_dict = {ATTACK_CLASSES[i]: round(probs[i].item(), 4) for i in range(self.num_classes)}
        return pred, ATTACK_CLASSES[pred], conf, probs_dict


# ─── Checkpoint Utilities ─────────────────────────────────────────────────────

def save_checkpoint(model: DDoSDetectionGNN, optimizer, stats: dict, filepath: str):
    """Save model + optimizer state to a checkpoint file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stats":                stats,
        "model_config": {
            "edge_feature_dim": model.edge_encoder[0].in_features,  # reads actual input dim
            "hidden_dim":       model.hidden_dim,
            "num_layers":       model.num_layers,
            "dropout":          model.dropout,
            "num_classes":      model.num_classes,
        },
    }, filepath)


def load_checkpoint(filepath: str, device: str = "cpu"):
    """Load checkpoint. Returns (model, optimizer_state_dict, stats)."""
    ckpt = torch.load(filepath, map_location=device)

    cfg = ckpt.get("model_config", {})
    model = DDoSDetectionGNN(
        edge_feature_dim = cfg.get("edge_feature_dim", MODEL["edge_feature_dim"]),  # ← ADD THIS
        hidden_dim  = cfg.get("hidden_dim",   MODEL["hidden_dim"]),
        num_layers  = cfg.get("num_layers",   MODEL["num_layers"]),
        dropout     = cfg.get("dropout",      MODEL["dropout"]),
        num_classes = cfg.get("num_classes",  MODEL["num_classes"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt.get("optimizer_state_dict"), ckpt.get("stats", {})


def rotate_checkpoints(models_dir: str, max_keep: int = 5):
    """Delete oldest rolling checkpoints, keep only max_keep."""
    import glob
    checkpoints = sorted(
        glob.glob(os.path.join(models_dir, "checkpoint_iter_*.pth")),
        key=os.path.getmtime,
    )
    for old in checkpoints[:-max_keep]:
        try:
            os.remove(old)
        except OSError:
            pass


def calibrate_temperature(model: DDoSDetectionGNN, val_loader, device: str = "cpu", max_iter: int = 50):
    """
    Calibrate model temperature using validation set for better confidence estimates.
    Uses simple temperature scaling - finds optimal T that minimizes NLL on validation set.
    
    Returns: optimal temperature value
    """
    from torch.utils.data import DataLoader
    import torch.optim as optim
    
    model.eval()
    nll_criterion = nn.CrossEntropyLoss()
    
    # Collect all logits and labels from validation set
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch)
            all_logits.append(logits)
            all_labels.append(batch.y.squeeze())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Optimize temperature
    temperature = nn.Parameter(torch.ones(1, device=device))
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    
    def eval_loss():
        optimizer.zero_grad()
        loss = nll_criterion(all_logits / temperature, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    optimal_temp = temperature.item()
    
    # Update model's temperature parameter
    model.temperature.data = torch.tensor([optimal_temp])
    
    return optimal_temp
