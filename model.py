import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Dropout
from torch_geometric.nn import SAGEConv, HeteroConv, GATConv, GCNConv


class HeteroSAGE(torch.nn.Module):
    """
    Heterogeneous GraphSAGE model
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension (number of classes)
        num_layers: Number of graph convolution layers (default: 2)
        dropout: Dropout rate (default: 0.5)
        aggr: Aggregation method for HeteroConv (default: "sum")
    """
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels,
        num_layers=2,
        dropout=0.5,
        aggr="sum"
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(
            HeteroConv({
                ("tx", "to", "tx"):     SAGEConv(in_channels, hidden_channels),
                ("addr", "to", "addr"): SAGEConv(in_channels, hidden_channels),
                ("addr", "to", "tx"):   SAGEConv(in_channels, hidden_channels),
                ("tx", "to", "addr"):   SAGEConv(in_channels, hidden_channels),
            }, aggr=aggr)
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                HeteroConv({
                    ("tx", "to", "tx"):     SAGEConv(hidden_channels, hidden_channels),
                    ("addr", "to", "addr"): SAGEConv(hidden_channels, hidden_channels),
                    ("addr", "to", "tx"):   SAGEConv(hidden_channels, hidden_channels),
                    ("tx", "to", "addr"):   SAGEConv(hidden_channels, hidden_channels),
                }, aggr=aggr)
            )
        
        # Last layer
        self.convs.append(
            HeteroConv({
                ("tx", "to", "tx"):     SAGEConv(hidden_channels, out_channels),
                ("addr", "to", "addr"): SAGEConv(hidden_channels, out_channels),
                ("addr", "to", "tx"):   SAGEConv(hidden_channels, out_channels),
                ("tx", "to", "addr"):   SAGEConv(hidden_channels, out_channels),
            }, aggr=aggr)
        )

    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation and dropout except for the last layer
            if i < self.num_layers - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                         for k, v in x_dict.items()}
        
        return x_dict


class HeteroGAT(torch.nn.Module):
    """
    Heterogeneous GAT with core improvements only:
      - Type-specific input projections
      - Self-loops on homogeneous relations (tx->tx, addr->addr)
      - LayerNorm + Residual (hidden * heads 구간)

    Args:
        in_channels_dict: dict, e.g. {"tx": 183, "addr": 56}
        hidden_channels: common hidden dim after projection
        out_channels: number of classes
        num_layers: number of GNN layers
        heads: number of attention heads
        dropout: dropout rate
    """
    def __init__(
        self,
        in_channels_dict,
        hidden_channels=32,
        out_channels=2,
        num_layers=2,
        heads=2,
        dropout=0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout

        # 1. 타입별 입력 투영 (각 노드 타입마다 Linear)
        self.input_projs = torch.nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.input_projs[node_type] = Linear(in_channels, hidden_channels)

        # 2. Hetero GAT 레이어들
        self.convs = torch.nn.ModuleList()

        # ── 1층: hidden -> hidden * heads ─────────────────────
        self.convs.append(
            HeteroConv(
                {
                    # homogeneous 관계: self-loop 허용
                    ("tx", "to", "tx"): GATConv(
                        hidden_channels,
                        hidden_channels,
                        heads=heads,
                        add_self_loops=True,
                    ),
                    ("addr", "to", "addr"): GATConv(
                        hidden_channels,
                        hidden_channels,
                        heads=heads,
                        add_self_loops=True,
                    ),
                    # heterogeneous 관계: self-loop 금지
                    ("addr", "to", "tx"): GATConv(
                        hidden_channels,
                        hidden_channels,
                        heads=heads,
                        add_self_loops=False,
                    ),
                    ("tx", "to", "addr"): GATConv(
                        hidden_channels,
                        hidden_channels,
                        heads=heads,
                        add_self_loops=False,
                    ),
                },
                aggr="sum",
            )
        )

        # ── 중간 레이어들: (hidden * heads) -> (hidden * heads) ─
        for _ in range(num_layers - 2):
            self.convs.append(
                HeteroConv(
                    {
                        ("tx", "to", "tx"): GATConv(
                            hidden_channels * heads,
                            hidden_channels,
                            heads=heads,
                            add_self_loops=True,
                        ),
                        ("addr", "to", "addr"): GATConv(
                            hidden_channels * heads,
                            hidden_channels,
                            heads=heads,
                            add_self_loops=True,
                        ),
                        ("addr", "to", "tx"): GATConv(
                            hidden_channels * heads,
                            hidden_channels,
                            heads=heads,
                            add_self_loops=False,
                        ),
                        ("tx", "to", "addr"): GATConv(
                            hidden_channels * heads,
                            hidden_channels,
                            heads=heads,
                            add_self_loops=False,
                        ),
                    },
                    aggr="sum",
                )
            )

        # ── 마지막 레이어: (hidden * heads) -> out_channels ────
        self.convs.append(
            HeteroConv(
                {
                    ("tx", "to", "tx"): GATConv(
                        hidden_channels * heads,
                        out_channels,
                        heads=1,
                        add_self_loops=True,
                    ),
                    ("addr", "to", "addr"): GATConv(
                        hidden_channels * heads,
                        out_channels,
                        heads=1,
                        add_self_loops=True,
                    ),
                    ("addr", "to", "tx"): GATConv(
                        hidden_channels * heads,
                        out_channels,
                        heads=1,
                        add_self_loops=False,
                    ),
                    ("tx", "to", "addr"): GATConv(
                        hidden_channels * heads,
                        out_channels,
                        heads=1,
                        add_self_loops=False,
                    ),
                },
                aggr="sum",
            )
        )

        # 3. LayerNorm (hidden * heads 차원에서 사용)
        self.norms = torch.nn.ModuleDict()
        hidden_out_dim = hidden_channels * heads
        for node_type in in_channels_dict.keys():
            self.norms[node_type] = LayerNorm(hidden_out_dim)

    def forward(self, x_dict, edge_index_dict):
        # Step 1: 타입별 입력 투영
        x_dict = {
            node_type: self.input_projs[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Step 2: GAT 레이어들
        for layer_idx, conv in enumerate(self.convs):
            x_res = x_dict  # residual 후보 (shape 맞는 경우에만 사용)

            x_dict = conv(x_dict, edge_index_dict)

            # 마지막 레이어 제외: Norm + Residual + 활성 + Dropout
            if layer_idx < self.num_layers - 1:
                new_x_dict = {}
                for node_type, x in x_dict.items():
                    # 중간 레이어 출력은 hidden * heads 차원
                    if x.shape[1] == self.hidden_channels * self.heads:
                        # LayerNorm
                        x = self.norms[node_type](x)
                        # residual (shape 맞는 경우에만)
                        if (
                            node_type in x_res
                            and x_res[node_type].shape == x.shape
                        ):
                            x = x + x_res[node_type]
                    # 비선형 + dropout
                    x = F.elu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    new_x_dict[node_type] = x
                x_dict = new_x_dict

        # node_type별 최종 출력 (예: x_dict["tx"]를 로짓으로 사용)
        return x_dict


class HeteroGCN(torch.nn.Module):
    """
    Heterogeneous Graph Convolutional Network
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension (number of classes)
        num_layers: Number of graph convolution layers (default: 2)
        dropout: Dropout rate (default: 0.5)
        aggr: Aggregation method for HeteroConv (default: "sum")
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        aggr="sum"
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(
            HeteroConv({
                ("tx", "to", "tx"):     GCNConv(in_channels, hidden_channels),
                ("addr", "to", "addr"): GCNConv(in_channels, hidden_channels),
                ("addr", "to", "tx"):   GCNConv(in_channels, hidden_channels),
                ("tx", "to", "addr"):   GCNConv(in_channels, hidden_channels),
            }, aggr=aggr)
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                HeteroConv({
                    ("tx", "to", "tx"):     GCNConv(hidden_channels, hidden_channels),
                    ("addr", "to", "addr"): GCNConv(hidden_channels, hidden_channels),
                    ("addr", "to", "tx"):   GCNConv(hidden_channels, hidden_channels),
                    ("tx", "to", "addr"):   GCNConv(hidden_channels, hidden_channels),
                }, aggr=aggr)
            )
        
        # Last layer
        self.convs.append(
            HeteroConv({
                ("tx", "to", "tx"):     GCNConv(hidden_channels, out_channels),
                ("addr", "to", "addr"): GCNConv(hidden_channels, out_channels),
                ("addr", "to", "tx"):   GCNConv(hidden_channels, out_channels),
                ("tx", "to", "addr"):   GCNConv(hidden_channels, out_channels),
            }, aggr=aggr)
        )

    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            if i < self.num_layers - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                         for k, v in x_dict.items()}
        
        return x_dict


def get_model(model_name, in_channels=None, in_channels_dict=None, hidden_channels=64, out_channels=2, **kwargs):
    """
    Factory function to get model by name
    
    Args:
        model_name: Name of the model ("sage", "gat", "gcn")
        in_channels: Input feature dimension (for SAGE and GCN, or if in_channels_dict not provided)
        in_channels_dict: Dictionary of input dimensions per node type (for GAT)
                         e.g., {"tx": 183, "addr": 56}
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension
        **kwargs: Additional model-specific arguments
    
    Returns:
        Model instance
    """
    models = {
        "sage": HeteroSAGE,
        "gat": HeteroGAT,
        "gcn": HeteroGCN,
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model_name_lower = model_name.lower()
    
    # GAT model requires in_channels_dict
    if model_name_lower == "gat":
        if in_channels_dict is None:
            if in_channels is None:
                raise ValueError("GAT model requires either 'in_channels_dict' or 'in_channels'")
            # Fallback: use same dimension for all node types (not ideal but works)
            in_channels_dict = {"tx": in_channels, "addr": in_channels}
        return models[model_name_lower](
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            **kwargs
        )
    else:
        # SAGE and GCN use in_channels
        if in_channels is None:
            raise ValueError(f"{model_name.upper()} model requires 'in_channels'")
        return models[model_name_lower](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            **kwargs
        )