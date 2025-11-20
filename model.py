import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Dropout
from torch_geometric.nn import SAGEConv, HeteroConv, GATConv, GCNConv, GraphConv


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

class GTNLiteLayer(nn.Module):
    def __init__(self, edge_types, in_channels, out_channels):
        super().__init__()
        self.edge_types = list(edge_types)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs = nn.ModuleDict()
        for (src, rel, dst) in self.edge_types:
            key = f"{src}__{rel}__{dst}"
            # SAGEConv는 bipartite message passing을 지원하므로 heterogeneous edge에 사용
            self.convs[key] = SAGEConv(in_channels, out_channels)

        # relation logits: [R]
        self.rel_logits = nn.Parameter(torch.zeros(len(self.edge_types)))

    def forward(self, x_dict, edge_index_dict):
        device = self.rel_logits.device
        rel_weight = F.softmax(self.rel_logits, dim=0)  # [R]

        # dst 타입별 출력 초기화
        new_x_dict = {
            ntype: x.new_zeros(x.size(0), self.out_channels, device=device)
            for ntype, x in x_dict.items()
        }

        for idx, (src, rel, dst) in enumerate(self.edge_types):
            if (src, rel, dst) not in edge_index_dict:
                continue

            key = f"{src}__{rel}__{dst}"
            conv = self.convs[key]
            edge_index = edge_index_dict[(src, rel, dst)]

            if src == dst:
                # 동종 관계 (tx->tx, addr->addr): 일반 모드
                h_dst = conv(x_dict[src], edge_index)
            else:
                # 이종 관계 (addr->tx, tx->addr): bipartite 모드
                # SAGEConv는 bipartite를 지원하므로 (src_x, dst_x) 튜플로 전달
                h_dst = conv((x_dict[src], x_dict[dst]), edge_index)

            new_x_dict[dst] = new_x_dict[dst] + rel_weight[idx] * h_dst

        return new_x_dict, rel_weight



class HeteroGTNLite(nn.Module):
    """
    GTN-lite 스타일의 Heterogeneous GCN:
    - edge-type별 SAGEConv (bipartite message passing 지원)
    - edge-type별 learnable weight (softmax)로 relation importance 학습
    - 메타패스 adjacency 곱(spspmm) 없음 → full-graph 제약 완화

    Args:
        in_channels: INPUT feature dimension (tx/addr 모두 동일하다고 가정)
        hidden_channels: Hidden dim
        out_channels: #classes
        num_layers: #layers (>=2 권장)
        dropout: dropout prob
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        aggr="sum",   # aggr는 여기선 사용 안 하지만 get_model 시그니처 맞추려고 유지
    ):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.dropout = dropout

        # 현재 그래프 스키마에 맞춘 edge_types (model.py 다른 모델과 동일) :contentReference[oaicite:2]{index=2}
        self.edge_types = [
            ("tx", "to", "tx"),
            ("addr", "to", "addr"),
            ("addr", "to", "tx"),
            ("tx", "to", "addr"),
        ]

        self.layers = nn.ModuleList()

        # 첫 레이어: in_channels -> hidden_channels
        if num_layers == 1:
            # 레이어 1개짜리면 바로 out_channels로
            self.layers.append(
                GTNLiteLayer(self.edge_types, in_channels, out_channels)
            )
        else:
            self.layers.append(
                GTNLiteLayer(self.edge_types, in_channels, hidden_channels)
            )
            # 중간 레이어들: hidden -> hidden
            for _ in range(num_layers - 2):
                self.layers.append(
                    GTNLiteLayer(self.edge_types, hidden_channels, hidden_channels)
                )
            # 마지막 레이어: hidden -> out_channels
            self.layers.append(
                GTNLiteLayer(self.edge_types, hidden_channels, out_channels)
            )

    def forward(self, x_dict, edge_index_dict, return_rel_weights=False):
        rel_weights_per_layer = []

        for layer_idx, layer in enumerate(self.layers):
            x_dict, rel_w = layer(x_dict, edge_index_dict)
            rel_weights_per_layer.append(rel_w)  # [R] (softmax된 relation weight)

            # 마지막 레이어가 아니면 활성 + dropout
            if layer_idx < self.num_layers - 1:
                x_dict = {ntype: F.relu(x) for ntype, x in x_dict.items()}
                x_dict = {
                    ntype: F.dropout(x, p=self.dropout, training=self.training)
                    for ntype, x in x_dict.items()
                }

        if return_rel_weights:
            return x_dict, rel_weights_per_layer
        else:
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
        "gtnlite": HeteroGTNLite,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    # GAT model requires in_channels_dict
    if model_name == "gat":
        if in_channels_dict is None:
            if in_channels is None:
                raise ValueError("GAT model requires either 'in_channels_dict' or 'in_channels'")
            # Fallback: use same dimension for all node types (not ideal but works)
            in_channels_dict = {"tx": in_channels, "addr": in_channels}
        return models[model_name](
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            **kwargs
        )
    else:
        # SAGE, GCN, and GTNLite use in_channels
        if in_channels is None:
            raise ValueError(f"{model_name.upper()} model requires 'in_channels'")
        return models[model_name](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            **kwargs
        )