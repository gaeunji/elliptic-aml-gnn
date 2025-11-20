# MKL 라이브러리 오류 해결을 위한 환경 변수 설정
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.data import HeteroData

# Conditionally import NeighborLoader (requires pyg-lib or torch-sparse)
try:
    from torch_geometric.loader import NeighborLoader
    NEIGHBOR_LOADER_AVAILABLE = True
except ImportError as e:
    NEIGHBOR_LOADER_AVAILABLE = False
    NeighborLoader = None
    IMPORT_ERROR_MSG = str(e)

from model import get_model


# ============================================================
# Seed Setting
# ============================================================
def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Configuration
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(description="Heterogeneous GNN for Elliptic++")
    
    # Data
    parser.add_argument("--data_path", type=str, default="elliptic_hetero_static.pt",
                       help="Path to preprocessed data")
    
    # Model
    parser.add_argument("--model", type=str, default="sage", 
                       choices=["sage", "gat", "gcn", "gtnlite"],
                       help="Model architecture")
    parser.add_argument("--hidden_channels", type=int, default=64,
                       help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.5,
                       help="Dropout rate")
    parser.add_argument("--aggr", type=str, default="sum",
                       choices=["sum", "mean", "max"],
                       help="Aggregation method")
    
    # GAT specific
    parser.add_argument("--heads", type=int, default=4,
                       help="Number of attention heads (for GAT)")
    
    # Training
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                       help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Log every N epochs")
    
    # Neighbor Sampling
    parser.add_argument("--batch_size", type=int, default=1024,
                       help="Batch size for neighbor sampling")
    parser.add_argument("--num_neighbors", type=int, nargs="+", default=[10, 5],
                       help="Number of neighbors to sample per layer (e.g., [10, 5] for 2 layers)")
    parser.add_argument("--use_neighbor_sampling", action="store_true",
                       help="Use neighbor sampling for training (default: full graph)")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cuda/cpu)")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Save
    parser.add_argument("--save_model", action="store_true",
                       help="Save best model")
    parser.add_argument("--model_save_path", type=str, default="best_model.pt",
                       help="Path to save model")
    
    return parser.parse_args()


# ============================================================
# Data Loading
# ============================================================
def load_data(data_path, use_neighbor_sampling=False):
    """Load preprocessed heterogeneous graph data
    
    Returns:
        If use_neighbor_sampling=True: (hetero_data, y_tx, y_tx_raw)
        If use_neighbor_sampling=False: (x_dict, edge_index_dict, y_tx, y_tx_raw, train_mask, val_mask, test_mask)
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # weights_only=False is required for torch_geometric data structures
    data = torch.load(data_path, weights_only=False)
    
    if use_neighbor_sampling:
        # Create HeteroData object for NeighborLoader
        hetero_data = HeteroData()
        
        # Add node features
        hetero_data["tx"].x = data["tx"].x
        hetero_data["addr"].x = data["addr"].x
        
        # Add node labels and masks (tx only)
        hetero_data["tx"].y = data["tx"].y  # {1, 2, -1}
        hetero_data["tx"].train_mask = data["tx"].train_mask
        hetero_data["tx"].val_mask = data["tx"].val_mask
        hetero_data["tx"].test_mask = data["tx"].test_mask
        
        # Add edges
        hetero_data["tx", "to", "tx"].edge_index = data["tx", "to", "tx"].edge_index
        hetero_data["addr", "to", "addr"].edge_index = data["addr", "to", "addr"].edge_index
        hetero_data["addr", "to", "tx"].edge_index = data["addr", "to", "tx"].edge_index
        hetero_data["tx", "to", "addr"].edge_index = data["tx", "to", "addr"].edge_index
        
        # Map labels: 1(illicit/불법) → 1, 2(licit/정상) → 0
        y_tx_raw = data["tx"].y.clone()
        y_tx = y_tx_raw.clone()
        label_map = {1: 1, 2: 0}  # illicit → 1, licit → 0
        for old, new in label_map.items():
            y_tx[y_tx_raw == old] = new
        
        return hetero_data, y_tx, y_tx_raw
    else:
        # Original format for full graph training
        # Extract features
        x_dict = {
            "tx": data["tx"].x,
            "addr": data["addr"].x,
        }
        
        # Extract labels (tx only)
        y_tx_raw = data["tx"].y  # {1, 2, -1}
        
        # Extract edges
        edge_index_dict = {
            ("tx", "to", "tx"):     data["tx", "to", "tx"].edge_index,
            ("addr", "to", "addr"): data["addr", "to", "addr"].edge_index,
            ("addr", "to", "tx"):   data["addr", "to", "tx"].edge_index,
            ("tx", "to", "addr"):   data["tx", "to", "addr"].edge_index,
        }
        
        # Extract masks
        train_mask = data["tx"].train_mask
        val_mask = data["tx"].val_mask
        test_mask = data["tx"].test_mask
        
        # Map labels: 1(illicit/불법) → 1, 2(licit/정상) → 0
        y_tx = y_tx_raw.clone()
        label_map = {1: 1, 2: 0}  # illicit → 1, licit → 0
        for old, new in label_map.items():
            y_tx[y_tx_raw == old] = new
        
        return x_dict, edge_index_dict, y_tx, y_tx_raw, train_mask, val_mask, test_mask


# ============================================================
# Class Weight Calculation
# ============================================================
def compute_class_weights(y_tx, train_mask, device):
    """
    Compute class weights for imbalanced dataset (binary classification).
    y_tx: tensor with {0=licit, 1=illicit, -1=unknown}
    """
    # Filter labeled train samples
    train_labels = y_tx[train_mask & (y_tx != -1)]

    # Safety check
    if train_labels.numel() == 0:
        return None
    
    # Count labels
    num_pos = (train_labels == 1).sum().item()
    num_neg = (train_labels == 0).sum().item()

    # Avoid divide-by-zero
    if num_pos == 0 or num_neg == 0:
        return None

    # Inverse frequency
    total = num_pos + num_neg
    w_neg = total / num_neg
    w_pos = total / num_pos

    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)

    # (Optional) clipping to prevent explosion
    class_weights = torch.clamp(class_weights, max=10.0)

    return class_weights



# ============================================================
# Training & Evaluation
# ============================================================
def train_epoch_full_graph(model, x_dict, edge_index_dict, y_tx, y_tx_raw, train_mask, optimizer, class_weights=None):
    """Train for one epoch using full graph"""
    model.train()
    optimizer.zero_grad()
    
    out_dict = model(x_dict, edge_index_dict)
    logits_tx = out_dict["tx"]
    
    # Only labeled nodes (exclude unknown=-1)
    mask = train_mask & (y_tx_raw != -1)
    
    if class_weights is not None:
        # Ensure class_weights is on the same device as logits
        if class_weights.device != logits_tx.device:
            class_weights = class_weights.to(logits_tx.device)
        loss = F.cross_entropy(logits_tx[mask], y_tx[mask], weight=class_weights)
    else:
        loss = F.cross_entropy(logits_tx[mask], y_tx[mask])
    
    loss.backward()
    optimizer.step()
    
    return float(loss.item())


def train_epoch_neighbor_sampling(model, train_loader, device, optimizer, class_weights=None):
    """Train for one epoch using neighbor sampling"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Get node features and edge indices from batch
        x_dict = {node_type: batch[node_type].x for node_type in batch.node_types}
        edge_index_dict = {
            edge_type: batch[edge_type].edge_index 
            for edge_type in batch.edge_types
        }
        
        # Forward pass
        out_dict = model(x_dict, edge_index_dict)
        logits_tx = out_dict["tx"]
        
        # Get labels and mask for training nodes in this batch
        y_batch = batch["tx"].y
        train_mask_batch = batch["tx"].train_mask
        
        # Only labeled nodes (exclude unknown=-1)
        mask = train_mask_batch & (y_batch != -1)
        
        if mask.sum() > 0:
            # Map labels: 1(illicit/불법) → 1, 2(licit/정상) → 0
            y_batch_mapped = y_batch.clone()
            label_map = {1: 1, 2: 0}  # illicit → 1, licit → 0
            for old, new in label_map.items():
                y_batch_mapped[y_batch == old] = new
            
            if class_weights is not None:
                # Ensure class_weights is on the same device as logits
                if class_weights.device != logits_tx.device:
                    class_weights = class_weights.to(logits_tx.device)
                loss = F.cross_entropy(logits_tx[mask], y_batch_mapped[mask], weight=class_weights)
            else:
                loss = F.cross_entropy(logits_tx[mask], y_batch_mapped[mask])
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss.item())
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, x_dict, edge_index_dict, y_tx, y_tx_raw, mask):
    """Evaluate on given mask using full graph"""
    model.eval()
    
    out_dict = model(x_dict, edge_index_dict)
    logits_tx = out_dict["tx"]
    
    eval_mask = mask & (y_tx_raw != -1)
    pred = logits_tx[eval_mask].argmax(dim=-1)
    y_true = y_tx[eval_mask]
    
    # Calculate loss
    loss = F.cross_entropy(logits_tx[eval_mask], y_true).item()
    
    correct = (pred == y_true).sum().item()
    total = int(eval_mask.sum())
    
    acc = correct / total if total > 0 else 0.0
    
    # Calculate precision, recall, f1, confusion matrix
    if total > 0:
        pred_np = pred.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        precision = precision_score(y_true_np, pred_np, average='binary', zero_division=0)
        recall = recall_score(y_true_np, pred_np, average='binary', zero_division=0)
        f1 = f1_score(y_true_np, pred_np, average='binary', zero_division=0)
        cm = confusion_matrix(y_true_np, pred_np, labels=[0, 1])
    else:
        precision = recall = f1 = 0.0
        cm = np.array([[0, 0], [0, 0]])
    
    return acc, precision, recall, f1, loss, cm


def print_relation_weights(model, edge_index_dict=None):
    """
    GTNLite 모델의 relation weight를 출력
    """
    if not hasattr(model, 'layers') or not hasattr(model, 'edge_types'):
        return
    
    print("\n" + "=" * 60)
    print("Learned Relation Weights (GTNLite)")
    print("=" * 60)
    
    edge_types = model.edge_types
    
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'rel_logits'):
            rel_weights = F.softmax(layer.rel_logits, dim=0)
            print(f"\nLayer {layer_idx + 1}:")
            for idx, (src, rel, dst) in enumerate(edge_types):
                if idx < len(rel_weights):
                    weight = rel_weights[idx].item()
                    print(f"  {src:4s} -> {dst:4s} ({rel:2s}): {weight:7.4f} ({weight*100:5.2f}%)")
    print("=" * 60 + "\n")


@torch.no_grad()
def evaluate_from_hetero_data(model, hetero_data, y_tx, y_tx_raw, mask_name, device):
    """Evaluate on given mask using full graph from HeteroData"""
    model.eval()
    
    # Use full graph for evaluation
    x_dict = {node_type: hetero_data[node_type].x.to(device) for node_type in hetero_data.node_types}
    edge_index_dict = {
        edge_type: hetero_data[edge_type].edge_index.to(device)
        for edge_type in hetero_data.edge_types
    }
    
    out_dict = model(x_dict, edge_index_dict)
    logits_tx = out_dict["tx"].cpu()
    
    # Get mask
    mask = hetero_data["tx"][mask_name].cpu()
    eval_mask = mask & (y_tx_raw != -1)
    
    pred = logits_tx[eval_mask].argmax(dim=-1)
    y_true = y_tx[eval_mask]
    
    # Calculate loss
    loss = F.cross_entropy(logits_tx[eval_mask], y_true).item()
    
    correct = (pred == y_true).sum().item()
    total = int(eval_mask.sum())
    
    acc = correct / total if total > 0 else 0.0
    
    # Calculate precision, recall, f1, confusion matrix
    if total > 0:
        pred_np = pred.numpy()
        y_true_np = y_true.numpy()
        precision = precision_score(y_true_np, pred_np, average='binary', zero_division=0)
        recall = recall_score(y_true_np, pred_np, average='binary', zero_division=0)
        f1 = f1_score(y_true_np, pred_np, average='binary', zero_division=0)
        cm = confusion_matrix(y_true_np, pred_np, labels=[0, 1])
    else:
        precision = recall = f1 = 0.0
        cm = np.array([[0, 0], [0, 0]])
    
    return acc, precision, recall, f1, loss, cm


# ============================================================
# Main Training Loop
# ============================================================
def main():
    args = get_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}\n")
    
    # Device setup
    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA is not available. Possible reasons:")
        print("  - PyTorch was installed without CUDA support")
        print("  - No NVIDIA GPU detected")
        print("  - CUDA drivers not installed")
    print()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"WARNING: Requested CUDA device but CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print()
    
    # Check Neighbor Sampling availability
    if args.use_neighbor_sampling and not NEIGHBOR_LOADER_AVAILABLE:
        print("=" * 60)
        print("ERROR: Neighbor Sampling requires 'pyg-lib' or 'torch-sparse'")
        print("=" * 60)
        print("\nTo install, run one of the following:")
        print("  pip install pyg-lib")
        print("  or")
        print("  pip install torch-sparse")
        print("\nFor CUDA support, see:")
        print("  https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
        print(f"\nOriginal error: {IMPORT_ERROR_MSG}")
        raise ImportError("NeighborLoader requires 'pyg-lib' or 'torch-sparse'. Please install one of them.")
    
    # Load data
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    if args.use_neighbor_sampling:
        hetero_data, y_tx, y_tx_raw = load_data(args.data_path, use_neighbor_sampling=True)
        
        print(f"TX features: {hetero_data['tx'].x.shape}")
        print(f"ADDR features: {hetero_data['addr'].x.shape}")
        print(f"Label distribution: {torch.bincount(y_tx[y_tx >= 0])}")
        print(f"Train/Val/Test: {hetero_data['tx'].train_mask.sum()}/{hetero_data['tx'].val_mask.sum()}/{hetero_data['tx'].test_mask.sum()}\n")
        
        # Get training node indices
        train_idx = hetero_data["tx"].train_mask.nonzero(as_tuple=False).squeeze()
        
        # Setup neighbor sampling
        num_neighbors = args.num_neighbors
        if len(num_neighbors) != args.num_layers:
            print(f"Warning: num_neighbors length ({len(num_neighbors)}) != num_layers ({args.num_layers})")
            if len(num_neighbors) == 1:
                num_neighbors = num_neighbors * args.num_layers
            else:
                num_neighbors = num_neighbors[:args.num_layers]
        
        # Create neighbor loader for training
        train_loader = NeighborLoader(
            hetero_data,
            num_neighbors=num_neighbors,
            input_nodes=("tx", train_idx),
            batch_size=args.batch_size,
            shuffle=True,
        )
        
        print(f"Neighbor sampling config:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Num neighbors per layer: {num_neighbors}")
        print(f"  Training nodes: {len(train_idx)}\n")
        
        # For evaluation, prepare full graph data on device
        x_dict_eval = {node_type: hetero_data[node_type].x for node_type in hetero_data.node_types}
        edge_index_dict_eval = {
            edge_type: hetero_data[edge_type].edge_index
            for edge_type in hetero_data.edge_types
        }
        train_mask_eval = hetero_data["tx"].train_mask
        val_mask_eval = hetero_data["tx"].val_mask
        test_mask_eval = hetero_data["tx"].test_mask
        
        # Move to device
        x_dict_eval = {k: v.to(device) for k, v in x_dict_eval.items()}
        edge_index_dict_eval = {k: v.to(device) for k, v in edge_index_dict_eval.items()}
        y_tx = y_tx.to(device)
        y_tx_raw = y_tx_raw.to(device)
        train_mask_eval = train_mask_eval.to(device)
        val_mask_eval = val_mask_eval.to(device)
        test_mask_eval = test_mask_eval.to(device)
        
        # Store for model creation
        tx_feat_dim = hetero_data["tx"].x.size(1)
        addr_feat_dim = hetero_data["addr"].x.size(1)
    else:
        x_dict, edge_index_dict, y_tx, y_tx_raw, train_mask, val_mask, test_mask = load_data(args.data_path, use_neighbor_sampling=False)
        
        print(f"TX features: {x_dict['tx'].shape}")
        print(f"ADDR features: {x_dict['addr'].shape}")
        print(f"Label distribution: {torch.bincount(y_tx[y_tx >= 0])}")
        print(f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}\n")
        
        # Move to device
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
        y_tx = y_tx.to(device)
        y_tx_raw = y_tx_raw.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        
        # Store for model creation
        tx_feat_dim = x_dict["tx"].size(1)
        addr_feat_dim = x_dict["addr"].size(1)
        
        # For compatibility
        train_loader = None
        hetero_data = None
        x_dict_eval = x_dict
        edge_index_dict_eval = edge_index_dict
        train_mask_eval = train_mask
        val_mask_eval = val_mask
        test_mask_eval = test_mask
    
    # Model setup
    print("=" * 60)
    print(f"Building model: {args.model.upper()}")
    print("=" * 60)
    
    if args.model == "gat":
        # GAT model requires in_channels_dict and doesn't use aggr
        model_kwargs = {
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "heads": args.heads,
        }
        model = get_model(
            model_name=args.model,
            in_channels_dict={
                "tx": tx_feat_dim,
                "addr": addr_feat_dim
            },
            hidden_channels=args.hidden_channels,
            out_channels=2,  # Binary classification
            **model_kwargs
        ).to(device)
    else:
        # SAGE, GCN, and GTNLite use in_channels and aggr (GTNLite ignores aggr)
        model_kwargs = {
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "aggr": args.aggr,
        }
        model = get_model(
            model_name=args.model,
            in_channels=tx_feat_dim,
            hidden_channels=args.hidden_channels,
            out_channels=2,  # Binary classification
            **model_kwargs
        ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Compute class weights for imbalanced dataset
    print("=" * 60)
    print("Computing class weights for imbalanced dataset")
    print("=" * 60)
    
    # Initialize class_weights to None (will be set if computation succeeds)
    class_weights = None
    
    if args.use_neighbor_sampling:
        train_labeled_mask = train_mask_eval & (y_tx_raw != -1)
        train_labels = y_tx[train_labeled_mask]
        train_mask_for_weights = train_mask_eval
    else:
        train_labeled_mask = train_mask & (y_tx_raw != -1)
        train_labels = y_tx[train_labeled_mask]
        train_mask_for_weights = train_mask
    
    if len(train_labels) > 0:
        num_pos = (train_labels == 1).sum().item()
        num_neg = (train_labels == 0).sum().item()
        total = num_pos + num_neg
        
        print(f"Training set class distribution:")
        print(f"  Class 0 (Licit/정상):   {num_neg:,} samples ({num_neg/total*100:.2f}%)")
        print(f"  Class 1 (Illicit/불법): {num_pos:,} samples ({num_pos/total*100:.2f}%)")
        
        class_weights = compute_class_weights(y_tx, train_mask_for_weights, device)
        if class_weights is not None:
            print(f"\nComputed class weights (will be applied to loss):")
            print(f"  Class 0 (Licit):  {class_weights[0].item():.4f}")
            print(f"  Class 1 (Illicit): {class_weights[1].item():.4f}")
            print(f"  Weight ratio (Class1/Class0): {class_weights[1].item()/class_weights[0].item():.4f}")
            print(f"  Device: {class_weights.device}")
        else:
            print("\nWarning: Could not compute class weights. Using uniform weights.")
    else:
        print("Warning: No labeled training samples found. Using uniform weights.")
    
    print()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print("=" * 60)
    print("Training...")
    print("=" * 60)
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    
    best_val_metrics = None
    best_test_metrics = None
    
    for epoch in range(1, args.epochs + 1):
        # Training
        if args.use_neighbor_sampling:
            train_loss = train_epoch_neighbor_sampling(model, train_loader, device, optimizer, class_weights)
            # Evaluation uses full graph
            train_acc, train_prec, train_rec, train_f1, _, train_cm = evaluate(model, x_dict_eval, edge_index_dict_eval, y_tx, y_tx_raw, train_mask_eval)
            val_acc, val_prec, val_rec, val_f1, val_loss, val_cm = evaluate(model, x_dict_eval, edge_index_dict_eval, y_tx, y_tx_raw, val_mask_eval)
            test_acc, test_prec, test_rec, test_f1, _, test_cm = evaluate(model, x_dict_eval, edge_index_dict_eval, y_tx, y_tx_raw, test_mask_eval)
        else:
            train_loss = train_epoch_full_graph(model, x_dict, edge_index_dict, y_tx, y_tx_raw, train_mask, optimizer, class_weights)
            train_acc, train_prec, train_rec, train_f1, _, train_cm = evaluate(model, x_dict, edge_index_dict, y_tx, y_tx_raw, train_mask)
            val_acc, val_prec, val_rec, val_f1, val_loss, val_cm = evaluate(model, x_dict, edge_index_dict, y_tx, y_tx_raw, val_mask)
            test_acc, test_prec, test_rec, test_f1, _, test_cm = evaluate(model, x_dict, edge_index_dict, y_tx, y_tx_raw, test_mask)
        
        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            best_val_metrics = (val_acc, val_prec, val_rec, val_f1, val_cm)
            best_test_metrics = (test_acc, test_prec, test_rec, test_f1, test_cm)
            
            # Save model
            if args.save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_precision': val_prec,
                    'val_recall': val_rec,
                    'val_f1': val_f1,
                    'test_acc': test_acc,
                    'test_precision': test_prec,
                    'test_recall': test_rec,
                    'test_f1': test_f1,
                    'args': vars(args),
                }, args.model_save_path)
        
        # Log - 모든 epoch에 대해 출력
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train: Prec={train_prec:.4f} Rec={train_rec:.4f} F1={train_f1:.4f} | "
            f"Val: Prec={val_prec:.4f} Rec={val_rec:.4f} F1={val_f1:.4f} | "
            f"Test: Prec={test_prec:.4f} Rec={test_rec:.4f} F1={test_f1:.4f}"
        )
        if best_val_metrics:
            print(
                f"  Best @E{best_epoch}: "
                f"Val(Prec={best_val_metrics[1]:.4f} Rec={best_val_metrics[2]:.4f} F1={best_val_metrics[3]:.4f}) | "
                f"Test(Prec={best_test_metrics[1]:.4f} Rec={best_test_metrics[2]:.4f} F1={best_test_metrics[3]:.4f})"
            )
    
    # Final results
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    if best_val_metrics and best_test_metrics:
        print(f"Best epoch: {best_epoch}")
        print(f"\nBest Validation Metrics:")
        print(f"  Precision: {best_val_metrics[1]:.4f}")
        print(f"  Recall:    {best_val_metrics[2]:.4f}")
        print(f"  F1-Score:  {best_val_metrics[3]:.4f}")
        print(f"\nValidation Confusion Matrix:")
        val_cm = best_val_metrics[4]
        print(f"                Predicted")
        print(f"              Class 0  Class 1")
        print(f"  Actual 0    {val_cm[0,0]:6d}  {val_cm[0,1]:6d}")
        print(f"  Actual 1    {val_cm[1,0]:6d}  {val_cm[1,1]:6d}")
        print(f"\nCorresponding Test Metrics:")
        print(f"  Precision: {best_test_metrics[1]:.4f}")
        print(f"  Recall:    {best_test_metrics[2]:.4f}")
        print(f"  F1-Score:  {best_test_metrics[3]:.4f}")
        print(f"\nTest Confusion Matrix:")
        test_cm = best_test_metrics[4]
        print(f"                Predicted")
        print(f"              Class 0  Class 1")
        print(f"  Actual 0    {test_cm[0,0]:6d}  {test_cm[0,1]:6d}")
        print(f"  Actual 1    {test_cm[1,0]:6d}  {test_cm[1,1]:6d}")
    else:
        print(f"Best epoch: {best_epoch}")
    
    # Print relation weights for GTNLite model
    if args.model == "gtnlite":
        print_relation_weights(model)
    
    if args.save_model:
        print(f"\nModel saved to: {args.model_save_path}")


if __name__ == "__main__":
    main()