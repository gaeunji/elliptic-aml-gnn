import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Dropout, init
from torch_geometric.nn import SAGEConv, HeteroConv, GATConv, GCNConv, GraphConv
from torch_scatter import scatter_mean, scatter_add


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
        use_neighbors: If False, only use self-loops (ignore graph structure) (default: True)
    """
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels,
        num_layers=2,
        dropout=0.5,
        aggr="sum",
        use_neighbors=True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_neighbors = use_neighbors
        
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
        # If use_neighbors=False, create self-loop only edge_index_dict
        if not self.use_neighbors:
            edge_index_dict = self._create_self_loop_edges(x_dict)
        
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation and dropout except for the last layer
            if i < self.num_layers - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                         for k, v in x_dict.items()}
        
        return x_dict
    
    def _create_self_loop_edges(self, x_dict):
        """
        Create edge_index_dict with self-loops only (no neighbor connections)
        
        Args:
            x_dict: Dictionary of node features {node_type: [N_type, F_type]}
        
        Returns:
            edge_index_dict: Dictionary with self-loop edges only
        """
        device = next(iter(x_dict.values())).device
        self_loop_edge_dict = {}
        
        for node_type, features in x_dict.items():
            num_nodes = features.size(0)
            # Create self-loop edge_index: [0, 1, 2, ..., N-1] -> [0, 1, 2, ..., N-1]
            self_loop_indices = torch.arange(num_nodes, device=device)
            self_loop_edge_dict[(node_type, "to", node_type)] = torch.stack([
                self_loop_indices, self_loop_indices
            ], dim=0)
        
        return self_loop_edge_dict

class HomoGAT(nn.Module):
    """
    Homogeneous GAT (Tx-only) version of HeteroGAT.

    - 단일 노드 타입 (예: Transaction 노드만 사용)
    - self-loop 추가 (add_self_loops=True)
    - 중간 레이어: hidden -> hidden * heads -> hidden * heads ...
    - LayerNorm + Residual + ELU + Dropout 구조는 HeteroGAT와 동일하게 유지

    Args:
        in_channels: 입력 피처 차원 (예: Tx feature dim = 184)
        hidden_channels: 공통 hidden dim
        out_channels: 클래스 개수 (예: 2)
        num_layers: GNN 레이어 수
        heads: attention heads 수
        dropout: dropout 비율
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        out_channels=2,
        num_layers=2,
        heads=2,
        dropout=0.5,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout

        # 1. 입력 투영: (in_channels -> hidden_channels)
        self.input_proj = Linear(in_channels, hidden_channels)

        # 2. GAT 레이어들
        self.convs = nn.ModuleList()

        # ── 1층: hidden -> hidden * heads ───────────────────────
        # HeteroGAT에서 ("tx","to","tx")에 쓰던 것과 동일한 설정
        self.convs.append(
            GATConv(
                hidden_channels,
                hidden_channels,
                heads=heads,
                add_self_loops=True,
            )
        )

        # ── 중간 레이어들: (hidden * heads) -> (hidden * heads) ─
        # HeteroGAT에서 hidden*heads 입력을 다시 hidden으로 줄이고 heads를 곱해서
        # hidden*heads를 유지하는 패턴과 동일
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    add_self_loops=True,
                )
            )

        # ── 마지막 레이어: (hidden * heads) -> out_channels ─────
        # heads=1로 설정해서 최종 출력 차원 = out_channels
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                out_channels,
                heads=1,
                add_self_loops=True,
            )
        )

        # 3. LayerNorm (hidden * heads 차원에서 사용)
        #   - 중간 레이어 출력이 hidden * heads일 때만 적용
        self.norm = LayerNorm(hidden_channels * heads)

    def forward(self, x, edge_index):
        """
        Args:
            x: Tensor [num_nodes, in_channels]
            edge_index: LongTensor [2, num_edges] (homogeneous tx->tx 관점)

        Returns:
            out: Tensor [num_nodes, out_channels] (Tx 노드 로짓)
        """
        # Step 1: 입력 투영
        x = self.input_proj(x)  # [N, hidden]

        # Step 2: GAT 레이어 반복
        for layer_idx, conv in enumerate(self.convs):
            x_res = x  # residual 후보

            x = conv(x, edge_index)  # GATConv 출력

            # 마지막 레이어 제외: Norm + Residual + 활성 + Dropout
            if layer_idx < self.num_layers - 1:
                # GATConv 출력: [N, hidden * heads] (중간 레이어들)
                if x.shape[1] == self.hidden_channels * self.heads:
                    # LayerNorm
                    x = self.norm(x)

                    # residual: shape이 맞을 때만 더해줌
                    if x_res.shape == x.shape:
                        x = x + x_res

                # 비선형 + dropout
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # 마지막 레이어 출력: [N, out_channels]
        return x



class HeteroGAT(torch.nn.Module):
    """
    Improved Heterogeneous GAT with:
      - Type-specific input projections
      - Projection-based residual connections (first layer alignment)
      - Pre-norm (LayerNorm before Conv)
      - Layer-wise dropout scheduling
      - 마지막 GNN 레이어는 Tx 타깃만 업데이트
      - Tx 전용 classifier head
    """
    def __init__(
        self,
        in_channels_dict,
        hidden_channels=64,
        out_channels=2,
        num_layers=2,
        heads=4,
        dropout=0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.base_dropout = dropout
        self.out_channels = out_channels

        # 1. 타입별 입력 투영
        self.input_projs = torch.nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.input_projs[node_type] = Linear(in_channels, hidden_channels)

        hidden_out_dim = hidden_channels * heads
        self.hidden_out_dim = hidden_out_dim

        # 2. Residual projection (첫 레이어 shape 맞추기)
        #    hidden_channels -> hidden_out_dim
        self.residual_projs = torch.nn.ModuleDict()
        for node_type in in_channels_dict.keys():
            self.residual_projs[node_type] = Linear(
                hidden_channels, hidden_out_dim
            )

        # 3. Hetero GAT 레이어들
        self.convs = torch.nn.ModuleList()
        
        # ── 1층: hidden -> hidden * heads ─────────────────────
        self.convs.append(
            HeteroConv(
                {
                    ("tx", "to", "tx"): GATConv(
                        hidden_channels,
                        hidden_channels,
                        heads=heads,
                        add_self_loops=True,
                        concat=True,
                    ),
                    ("addr", "to", "addr"): GATConv(
                        hidden_channels,
                        hidden_channels,
                        heads=heads,
                        add_self_loops=True,
                        concat=True,
                    ),
                    ("addr", "to", "tx"): GATConv(
                        hidden_channels,
                        hidden_channels,
                        heads=heads,
                        add_self_loops=False,
                        concat=True,
                    ),
                    ("tx", "to", "addr"): GATConv(
                        hidden_channels,
                        hidden_channels,
                        heads=heads,
                        add_self_loops=False,
                        concat=True,
                    ),
                },
                aggr="sum",
            )
        )

        # ── 중간 레이어들: (hidden * heads) -> (hidden * heads) ─
        #     (num_layers >= 3일 때만 존재)
        for _ in range(num_layers - 2):
            self.convs.append(
                HeteroConv(
                    {
                        ("tx", "to", "tx"): GATConv(
                            hidden_out_dim,
                            hidden_channels,
                            heads=heads,
                            add_self_loops=True,
                            concat=True,
                        ),
                        ("addr", "to", "addr"): GATConv(
                            hidden_out_dim,
                            hidden_channels,
                            heads=heads,
                            add_self_loops=True,
                            concat=True,
                        ),
                        ("addr", "to", "tx"): GATConv(
                            hidden_out_dim,
                            hidden_channels,
                            heads=heads,
                            add_self_loops=False,
                            concat=True,
                        ),
                        ("tx", "to", "addr"): GATConv(
                            hidden_out_dim,
                            hidden_channels,
                            heads=heads,
                            add_self_loops=False,
                            concat=True,
                        ),
                    },
                    aggr="sum",
                )
            )

        # ── 마지막 레이어: (hidden * heads) -> (hidden * heads) ─
        #     → Tx 타깃만 업데이트: ("tx","to","tx"), ("addr","to","tx")
        self.convs.append(
            HeteroConv(
                {
                    ("tx", "to", "tx"): GATConv(
                        hidden_out_dim,
                        hidden_channels,
                        heads=heads,
                        add_self_loops=True,
                        concat=True,
                    ),
                    ("addr", "to", "tx"): GATConv(
                        hidden_out_dim,
                        hidden_channels,
                        heads=heads,
                        add_self_loops=False,
                        concat=True,
                    ),
                },
                aggr="sum",
            )
        )

        # 4. Pre-norm LayerNorm (Conv 전에 적용)
        self.norms = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_norms = torch.nn.ModuleDict()
            for node_type in in_channels_dict.keys():
                layer_norms[node_type] = LayerNorm(
                    hidden_channels if layer_idx == 0 else hidden_out_dim
                )
            self.norms.append(layer_norms)

        # 5. 간소화된 'tx' classifier head
        self.tx_classifier = Linear(hidden_out_dim, out_channels)

    def _get_dropout_rate(self, layer_idx):
        """
        레이어별 dropout rate 스케줄
        깊은 레이어일수록 더 강한 정규화
        """
        progress = (layer_idx + 1) / self.num_layers
        return self.base_dropout * progress

    def forward(self, x_dict, edge_index_dict):
        """
        Returns:
            x_dict: node_type별 출력
                - "tx": logits (shape [N_tx, out_channels])
                - "addr": embedding (shape [N_addr, hidden * heads])
        """
        # Step 1: 타입별 입력 투영
        x_dict = {
            node_type: self.input_projs[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Step 2: GAT 레이어들
        for layer_idx, conv in enumerate(self.convs):
            is_last = (layer_idx == self.num_layers - 1)

            # Pre-norm: LayerNorm 먼저 적용
            x_normalized = {
                node_type: self.norms[layer_idx][node_type](x)
                for node_type, x in x_dict.items()
            }

            # Residual connection 저장 (norm 전)
            x_res = {k: v for k, v in x_dict.items()}

            if not is_last:
                # ── 중간 레이어: 기존 방식 그대로 ─────────────
                x_dict = conv(x_normalized, edge_index_dict)

                dropout_rate = self._get_dropout_rate(layer_idx)
                new_x_dict = {}

                for node_type, x in x_dict.items():
                    # 첫 번째 레이어: residual projection 적용
                    if layer_idx == 0:
                        x_res_proj = self.residual_projs[node_type](x_res[node_type])
                        x = x + x_res_proj
                    else:
                        # 이후 레이어: 직접 residual
                        if node_type in x_res and x_res[node_type].shape == x.shape:
                            x = x + x_res[node_type]

                    # 비선형 + dropout
                    x = F.elu(x)
                    x = F.dropout(x, p=dropout_rate, training=self.training)
                    new_x_dict[node_type] = x

                x_dict = new_x_dict

            else:
                # ── 마지막 레이어: Tx 타깃만 업데이트 ───────────
                x_last = conv(x_normalized, edge_index_dict)  # {"tx": ...}만 존재

                dropout_rate = self._get_dropout_rate(layer_idx)
                new_x_dict = {}

                for node_type, x_prev in x_dict.items():
                    if node_type == "tx":
                        x = x_last["tx"]
                        # 마지막 레이어에서도 residual 적용 (shape 동일)
                        if node_type in x_res and x_res[node_type].shape == x.shape:
                            x = x + x_res[node_type]
                        x = F.elu(x)
                        x = F.dropout(x, p=dropout_rate, training=self.training)
                        new_x_dict["tx"] = x
                    else:
                        # addr 등은 마지막 레이어에서 업데이트하지 않고 이전 값 유지
                        new_x_dict[node_type] = x_prev

                x_dict = new_x_dict

        # Step 3: tx embedding에 classifier 적용
        if "tx" not in x_dict:
            raise KeyError("x_dict does not contain 'tx' node type.")

        tx_emb = x_dict["tx"]  # [N_tx, hidden * heads]
        logits = self.tx_classifier(tx_emb)  # [N_tx, out_channels]

        x_dict["tx"] = logits
        return x_dict

# ============================================================
# 1) TAT 메타 경로 인코더 (Tx-Addr-Tx 기반 Tx 임베딩)
# ============================================================

class HeteroGAT_TATEncoder(nn.Module):
    """
    TAT 메타 경로 (Transaction-Address-Transaction) 기반 Tx 임베딩 인코더.

    - 사용하는 관계: ("tx","to","addr"), ("addr","to","tx")
    - 목적: Tx-Addr-Tx (및 반복) 패턴을 반영한 Tx 임베딩 h_tx_TAT 생성

    Args:
        in_channels_dict: {"tx": F_tx, "addr": F_addr}
        hidden_channels:  투영 후 hidden dim
        emb_dim:          최종 Tx 임베딩 차원 (메타패스들끼리 맞춰둘 D)
        num_layers:       GNN 레이어 수 (2면 T-A-T 정도)
        heads:            GAT attention heads 수
        dropout:          dropout 비율
    """
    def __init__(
        self,
        in_channels_dict,
        hidden_channels=32,
        emb_dim=64,
        num_layers=2,
        heads=2,
        dropout=0.5,
    ):
        super().__init__()
        assert "tx" in in_channels_dict and "addr" in in_channels_dict, \
            "in_channels_dict에는 'tx'와 'addr'가 모두 있어야 합니다."
        assert num_layers >= 1, "num_layers는 1 이상이어야 합니다."

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout

        # 1. 타입별 입력 투영
        self.input_projs = nn.ModuleDict()
        for ntype, in_ch in in_channels_dict.items():
            self.input_projs[ntype] = Linear(in_ch, hidden_channels)

        # 2. TAT 전용 Hetero GAT 레이어들
        self.convs = nn.ModuleList()

        # ── 1층: hidden -> hidden * heads ─────────────────────
        self.convs.append(
            HeteroConv(
                {
                    ("tx", "to", "addr"): GATConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        heads=heads,
                        add_self_loops=False,
                    ),
                    ("addr", "to", "tx"): GATConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        heads=heads,
                        add_self_loops=False,
                    ),
                },
                aggr="sum",
            )
        )

        # ── 중간 레이어들: (hidden * heads) -> (hidden * heads) ─
        for _ in range(num_layers - 1):
            self.convs.append(
                HeteroConv(
                    {
                        ("tx", "to", "addr"): GATConv(
                            in_channels=hidden_channels * heads,
                            out_channels=hidden_channels,
                            heads=heads,
                            add_self_loops=False,
                        ),
                        ("addr", "to", "tx"): GATConv(
                            in_channels=hidden_channels * heads,
                            out_channels=hidden_channels,
                            heads=heads,
                            add_self_loops=False,
                        ),
                    },
                    aggr="sum",
                )
            )

        # 3. LayerNorm (hidden * heads 구간)
        self.norms = nn.ModuleDict()
        hidden_out_dim = hidden_channels * heads
        for ntype in in_channels_dict.keys():
            self.norms[ntype] = LayerNorm(hidden_out_dim)

        # 4. 최종 Tx 임베딩 투영 (hidden*heads -> emb_dim)
        self.tx_out_proj = Linear(hidden_out_dim, emb_dim)

    def forward(self, x_dict, edge_index_dict):
        """
        x_dict: {"tx": [N_tx, F_tx], "addr": [N_addr, F_addr]}
        edge_index_dict: {
            ("tx","to","addr"): edge_index_tx_addr,
            ("addr","to","tx"): edge_index_addr_tx,
            ...
        }

        Returns:
            h_tx_TAT: [N_tx, emb_dim]  (TAT 메타 경로 기반 Tx 임베딩)
        """
        # Step 1. 타입별 입력 투영
        x_dict = {
            ntype: self.input_projs[ntype](x)
            for ntype, x in x_dict.items()
        }

        # Step 2. TAT 전용 GAT 레이어들
        for layer_idx, conv in enumerate(self.convs):
            x_res = x_dict

            x_dict = conv(x_dict, edge_index_dict)

            # 모든 레이어에서 Norm + Residual + 활성 + Dropout
            new_x_dict = {}
            for ntype, x in x_dict.items():
                if x.shape[1] == self.hidden_channels * self.heads:
                    x = self.norms[ntype](x)
                    if (
                        ntype in x_res
                        and x_res[ntype].shape == x.shape
                    ):
                        x = x + x_res[ntype]
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                new_x_dict[ntype] = x
            x_dict = new_x_dict

        # TAT 기반 Tx embedding
        h_tx_TAT = x_dict["tx"]                      # [N_tx, H*heads]
        h_tx_TAT = self.tx_out_proj(h_tx_TAT)        # [N_tx, emb_dim]
        return h_tx_TAT


# ============================================================
# 2) ATA 메타 경로 인코더 (Addr-Tx-Addr 기반 Tx 임베딩)
# ============================================================

class HeteroGAT_ATAEncoder(nn.Module):
    """
    ATA 메타 경로(Addr-Transaction-Addr)를 기반으로
    Tx 임베딩을 만들어 주는 인코더.

    - 사용하는 관계: ("addr","to","tx"), ("tx","to","addr")
    - GAT 레이어를 통해 Addr/Tx 임베딩 학습 → Addr 임베딩을 Tx 쪽으로 pooling

    Args:
        in_channels_dict: {"tx": F_tx, "addr": F_addr}
        hidden_channels:   입력 투영 후 hidden dim
        emb_dim:           최종 Tx 임베딩 차원 (fusion 전에 맞춰둘 D)
        num_layers:        GNN 레이어 수 (2면 A-T-A 구조 반영)
        heads:             GAT attention heads 수
        dropout:           드롭아웃 비율
        pool:              Addr -> Tx pooling 방식 ("mean" or "sum")
    """
    def __init__(
        self,
        in_channels_dict,
        hidden_channels=32,
        emb_dim=64,
        num_layers=2,
        heads=2,
        dropout=0.5,
        pool: str = "mean",
    ):
        super().__init__()
        assert "tx" in in_channels_dict and "addr" in in_channels_dict, \
            "in_channels_dict에는 'tx'와 'addr'가 모두 있어야 합니다."
        assert num_layers >= 1, "num_layers는 1 이상이어야 합니다."
        assert pool in ["mean", "sum"]

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.pool = pool

        # 1. 타입별 입력 투영
        self.input_projs = nn.ModuleDict()
        for ntype, in_ch in in_channels_dict.items():
            self.input_projs[ntype] = Linear(in_ch, hidden_channels)

        # 2. ATA 전용 Hetero GAT 레이어들
        self.convs = nn.ModuleList()

        # ── 1층: hidden -> hidden * heads ─────────────────────
        self.convs.append(
            HeteroConv(
                {
                    ("addr", "to", "tx"): GATConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        heads=heads,
                        add_self_loops=False,
                    ),
                    ("tx", "to", "addr"): GATConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        heads=heads,
                        add_self_loops=False,
                    ),
                },
                aggr="sum",
            )
        )

        # ── 중간 레이어들: (hidden * heads) -> (hidden * heads) ─
        for _ in range(num_layers - 1):
            self.convs.append(
                HeteroConv(
                    {
                        ("addr", "to", "tx"): GATConv(
                            in_channels=hidden_channels * heads,
                            out_channels=hidden_channels,
                            heads=heads,
                            add_self_loops=False,
                        ),
                        ("tx", "to", "addr"): GATConv(
                            in_channels=hidden_channels * heads,
                            out_channels=hidden_channels,
                            heads=heads,
                            add_self_loops=False,
                        ),
                    },
                    aggr="sum",
                )
            )

        # 3. LayerNorm (hidden * heads 구간)
        self.norms = nn.ModuleDict()
        hidden_out_dim = hidden_channels * heads
        for ntype in in_channels_dict.keys():
            self.norms[ntype] = LayerNorm(hidden_out_dim)

        # 4. 최종 Tx 임베딩 투영 (hidden*heads -> emb_dim)
        self.tx_out_proj = Linear(hidden_out_dim, emb_dim)

    def forward(self, x_dict, edge_index_dict):
        """
        Returns:
            h_tx_ATA: [N_tx, emb_dim]  (ATA 메타 경로 기반 Tx 임베딩)
        """
        # Step 1. 타입별 입력 투영
        x_dict = {
            ntype: self.input_projs[ntype](x)
            for ntype, x in x_dict.items()
        }

        # Step 2. Hetero GAT 레이어들 (ATA 관계만 사용)
        for layer_idx, conv in enumerate(self.convs):
            x_res = x_dict

            x_dict = conv(x_dict, edge_index_dict)

            new_x_dict = {}
            for ntype, x in x_dict.items():
                if x.shape[1] == self.hidden_channels * self.heads:
                    x = self.norms[ntype](x)
                    if (
                        ntype in x_res
                        and x_res[ntype].shape == x.shape
                    ):
                        x = x + x_res[ntype]
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                new_x_dict[ntype] = x
            x_dict = new_x_dict

        # 여기까지: x_dict["addr"], x_dict["tx"] 둘 다 [*, hidden*heads]

        # Step 3. Addr 임베딩을 Tx 기준으로 pooling
        if ("addr", "to", "tx") not in edge_index_dict:
            raise KeyError("edge_index_dict에 ('addr','to','tx') 엣지가 필요합니다.")

        edge_index_at = edge_index_dict[("addr", "to", "tx")]
        addr_idx, tx_idx = edge_index_at  # [E], [E]

        h_addr = x_dict["addr"]          # [N_addr, H*heads]
        N_tx = x_dict["tx"].size(0)

        if self.pool == "mean":
            h_tx_agg = scatter_mean(
                h_addr[addr_idx], tx_idx,
                dim=0, dim_size=N_tx
            )
        else:  # "sum"
            h_tx_agg = scatter_add(
                h_addr[addr_idx], tx_idx,
                dim=0, dim_size=N_tx
            )

        h_tx_ATA = self.tx_out_proj(h_tx_agg)  # [N_tx, emb_dim]
        return h_tx_ATA


# ============================================================
# 3) 메타 패스 Semantic Attention (HAN 스타일)
# ============================================================

class MetaPathSemanticAttention(nn.Module):
    """
    메타 경로별 Tx 임베딩 리스트를 받아,
    semantic-level attention으로 가중 평균하는 모듈.

    h_list: M개의 [N_tx, D] 텐서
    """
    def __init__(self, in_dim, hidden_dim, num_metapaths):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.context = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, h_list):
        # h_list: M개 [N, D]
        M = len(h_list)
        assert M == self.num_metapaths

        # 각 메타 경로별 대표 임베딩 (Tx 평균)
        H_bar = []
        for h in h_list:
            H_bar.append(h.mean(dim=0))     # [D]
        H_bar = torch.stack(H_bar, dim=0)   # [M, D]

        # semantic score 계산
        H_tilde = torch.tanh(self.proj(H_bar))       # [M, H]
        scores = (H_tilde * self.context).sum(dim=-1)  # [M]
        alpha = F.softmax(scores, dim=0)             # [M]

        # 가중 평균
        H = torch.stack(h_list, dim=0)      # [M, N, D]
        alpha_view = alpha.view(M, 1, 1)
        h_tx_fused = (alpha_view * H).sum(dim=0)  # [N, D]

        return h_tx_fused, alpha

class HeteroGAT_FullEncoder(nn.Module):
    """
    기존 HeteroGAT 구조를 그대로 활용하되,
    최종 출력은 Tx 임베딩 (h_tx_full)로 쓰기 위한 인코더 버전.

    - 사용하는 관계:
        ("tx","to","tx"), ("addr","to","addr"),
        ("tx","to","addr"), ("addr","to","tx")

    Args:
        in_channels_dict: {"tx": F_tx, "addr": F_addr}
        hidden_channels:  투영 후 hidden dim
        emb_dim:          최종 Tx 임베딩 차원 (TAT/ATA와 맞추는 D)
        num_layers:       GNN 레이어 수
        heads:            GAT heads
        dropout:          dropout 비율
    """
    def __init__(
        self,
        in_channels_dict,
        hidden_channels=32,
        emb_dim=64,
        num_layers=2,
        heads=2,
        dropout=0.5,
    ):
        super().__init__()
        assert "tx" in in_channels_dict and "addr" in in_channels_dict
        assert num_layers >= 1

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout

        # 1. 타입별 입력 투영
        self.input_projs = nn.ModuleDict()
        for ntype, in_ch in in_channels_dict.items():
            self.input_projs[ntype] = Linear(in_ch, hidden_channels)

        # 2. Hetero GAT 레이어들 (원본 구조)
        self.convs = nn.ModuleList()

        # ── 1층: hidden -> hidden * heads ─────────────────────
        self.convs.append(
            HeteroConv(
                {
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
                    ("tx", "to", "addr"): GATConv(
                        hidden_channels,
                        hidden_channels,
                        heads=heads,
                        add_self_loops=False,
                    ),
                    ("addr", "to", "tx"): GATConv(
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
        for _ in range(num_layers - 1):
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
                        ("tx", "to", "addr"): GATConv(
                            hidden_channels * heads,
                            hidden_channels,
                            heads=heads,
                            add_self_loops=False,
                        ),
                        ("addr", "to", "tx"): GATConv(
                            hidden_channels * heads,
                            hidden_channels,
                            heads=heads,
                            add_self_loops=False,
                        ),
                    },
                    aggr="sum",
                )
            )

        # 3. LayerNorm (hidden * heads)
        self.norms = nn.ModuleDict()
        hidden_out_dim = hidden_channels * heads
        for ntype in in_channels_dict.keys():
            self.norms[ntype] = LayerNorm(hidden_out_dim)

        # 4. 최종 Tx 임베딩 투영 (hidden*heads -> emb_dim)
        self.tx_out_proj = Linear(hidden_out_dim, emb_dim)

    def forward(self, x_dict, edge_index_dict):
        """
        Returns:
            h_tx_full: [N_tx, emb_dim]  (전체 이종 그래프 기반 Tx 임베딩)
        """
        # Step 1. 타입별 입력 투영
        x_dict = {
            ntype: self.input_projs[ntype](x)
            for ntype, x in x_dict.items()
        }

        # Step 2. GAT 레이어들
        for layer_idx, conv in enumerate(self.convs):
            x_res = x_dict

            x_dict = conv(x_dict, edge_index_dict)

            new_x_dict = {}
            for ntype, x in x_dict.items():
                if x.shape[1] == self.hidden_channels * self.heads:
                    x = self.norms[ntype](x)
                    if (
                        ntype in x_res
                        and x_res[ntype].shape == x.shape
                    ):
                        x = x + x_res[ntype]
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                new_x_dict[ntype] = x
            x_dict = new_x_dict

        h_tx_full = x_dict["tx"]             # [N_tx, H*heads]
        h_tx_full = self.tx_out_proj(h_tx_full)  # [N_tx, emb_dim]
        return h_tx_full

# ============================================================
# 4) 최종 Tx 분류 모델: TAT + ATA + Semantic Attention
# ============================================================

class MetaPathTxClassifier3Path(nn.Module):
    """
    3경로 fusion:
      - enc_TAT : TAT 메타 경로 (Tx-Addr-Tx)
      - enc_ATA : ATA 메타 경로 (Addr-Tx-Addr → Addr→Tx pooling)
      - enc_Full: 전체 이종 그래프 기반 Original HeteroGAT 인코더

    세 임베딩을 semantic attention으로 가중합한 뒤,
    최종 Tx 레이블을 분류하는 모델.
    """
    def __init__(
        self,
        in_channels_dict,
        hidden_channels=32,
        emb_dim=64,
        num_layers_tat=2,
        num_layers_ata=2,
        num_layers_full=2,
        heads=2,
        dropout=0.5,
        num_classes=2,
        semantic_hidden_dim=32,
    ):
        super().__init__()

        # 1) 메타 경로 인코더들
        self.enc_TAT = HeteroGAT_TATEncoder(
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            emb_dim=emb_dim,
            num_layers=num_layers_tat,
            heads=heads,
            dropout=dropout,
        )

        self.enc_ATA = HeteroGAT_ATAEncoder(
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            emb_dim=emb_dim,
            num_layers=num_layers_ata,
            heads=heads,
            dropout=dropout,
            pool="mean",
        )

        self.enc_Full = HeteroGAT_FullEncoder(
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            emb_dim=emb_dim,
            num_layers=num_layers_full,
            heads=heads,
            dropout=dropout,
        )

        # 2) 3경로 semantic attention
        self.semantic_attn = MetaPathSemanticAttention(
            in_dim=emb_dim,
            hidden_dim=semantic_hidden_dim,
            num_metapaths=3,
        )

        # 3) 최종 classifier
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x_dict, edge_index_dict):
        """
        Returns:
            logits_tx : [N_tx, num_classes]
            alpha     : [3] (TAT, ATA, Full 순서의 경로 가중치)
            h_tx_fused: [N_tx, emb_dim]
        """
        # 1) 각 경로별 Tx 임베딩
        h_tx_TAT  = self.enc_TAT(x_dict, edge_index_dict)   # [N_tx, D]
        h_tx_ATA  = self.enc_ATA(x_dict, edge_index_dict)   # [N_tx, D]
        h_tx_full = self.enc_Full(x_dict, edge_index_dict)  # [N_tx, D]

        # 2) 3경로 semantic attention fusion
        h_tx_fused, alpha = self.semantic_attn(
            [h_tx_TAT, h_tx_ATA, h_tx_full]
        )  # [N_tx, D], [3]

        # 3) 최종 분류
        logits = self.classifier(h_tx_fused)  # [N_tx, C]

        return logits, alpha, h_tx_fused





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
        # Use GCNConv for homogeneous edges, SAGEConv for heterogeneous edges (bipartite support)
        self.convs.append(
            HeteroConv({
                ("tx", "to", "tx"):     GCNConv(in_channels, hidden_channels, add_self_loops=True),
                ("addr", "to", "addr"): GCNConv(in_channels, hidden_channels, add_self_loops=True),
                ("addr", "to", "tx"):   SAGEConv(in_channels, hidden_channels),  # bipartite support
                ("tx", "to", "addr"):   SAGEConv(in_channels, hidden_channels),  # bipartite support
            }, aggr=aggr)
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                HeteroConv({
                    ("tx", "to", "tx"):     GCNConv(hidden_channels, hidden_channels, add_self_loops=True),
                    ("addr", "to", "addr"): GCNConv(hidden_channels, hidden_channels, add_self_loops=True),
                    ("addr", "to", "tx"):   SAGEConv(hidden_channels, hidden_channels),  # bipartite support
                    ("tx", "to", "addr"):   SAGEConv(hidden_channels, hidden_channels),  # bipartite support
                }, aggr=aggr)
            )
        
        # Last layer
        self.convs.append(
            HeteroConv({
                ("tx", "to", "tx"):     GCNConv(hidden_channels, out_channels, add_self_loops=True),
                ("addr", "to", "addr"): GCNConv(hidden_channels, out_channels, add_self_loops=True),
                ("addr", "to", "tx"):   SAGEConv(hidden_channels, out_channels),  # bipartite support
                ("tx", "to", "addr"):   SAGEConv(hidden_channels, out_channels),  # bipartite support
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


# ============================================================
# HGT Models - From hgt.py
# ============================================================
import math
from conv import GeneralConv


class HGTClassifier(nn.Module):
    """
    Classifier for HGT model output
    """
    def __init__(self, n_hid, n_out):
        super(HGTClassifier, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear = nn.Linear(n_hid, n_out)
    
    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)
    
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)


class HGTMatcher(nn.Module):
    """
    Matching between a pair of nodes to conduct link prediction.
    Use multi-head attention as matching model.
    """
    def __init__(self, n_hid):
        super(HGTMatcher, self).__init__()
        self.left_linear = nn.Linear(n_hid, n_hid)
        self.right_linear = nn.Linear(n_hid, n_hid)
        self.sqrt_hd = math.sqrt(n_hid)
        self.cache = None
    
    def forward(self, x, y, infer=False, pair=False):
        ty = self.right_linear(y)
        if infer:
            """
            During testing, we will consider millions or even billions of nodes as candidates (x).
            It's not possible to calculate them again for different query (y)
            Since the model is fixed, we propose to cache them, and directly use the results.
            """
            if self.cache is not None:
                tx = self.cache
            else:
                tx = self.left_linear(x)
                self.cache = tx
        else:
            tx = self.left_linear(x)
        if pair:
            res = (tx * ty).sum(dim=-1)
        else:
            res = torch.matmul(tx, ty.transpose(0, 1))
        return res / self.sqrt_hd
    
    def __repr__(self):
        return '{}(n_hid={})'.format(
            self.__class__.__name__, self.n_hid)


class HGTGNN(nn.Module):
    """
    HGT (Heterogeneous Graph Transformer) GNN model
    
    Args:
        in_dims: Input dimensions - can be a dict {type_id: dim} or a single int
        n_hid: Hidden dimension
        num_types: Number of node types
        num_relations: Number of relation types
        n_heads: Number of attention heads
        n_layers: Number of GNN layers
        dropout: Dropout rate (default: 0.2)
        conv_name: Convolution type ('hgt', 'hgt_care', 'dense_hgt') (default: 'hgt')
        prev_norm: Use normalization in previous layers (default: False)
        last_norm: Use normalization in last layer (default: False)
        use_RTE: Use Relative Temporal Encoding (default: True)
        care_temperature: Temperature for CARE scoring (default: 1.0)
        fusion_mode: Fusion mode for CARE ('log_add' or 'prob_mul') (default: 'log_add')
    """
    def __init__(self, in_dims, n_hid, num_types, num_relations, n_heads, n_layers, 
                 dropout=0.2, conv_name='hgt', prev_norm=False, last_norm=False, 
                 use_RTE=True, care_temperature=1.0, fusion_mode='log_add'):
        super(HGTGNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        # in_dims can be a dict {type_id: dim} or a single int (for backward compatibility)
        if isinstance(in_dims, dict):
            self.in_dims = in_dims
        else:
            # Backward compatibility: use same dimension for all types
            self.in_dims = {t: in_dims for t in range(num_types)}
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        for t in range(num_types):
            in_dim = self.in_dims.get(t, self.in_dims.get(list(self.in_dims.keys())[0] if self.in_dims else 0))
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, 
                                        dropout, use_norm=prev_norm, use_RTE=use_RTE, 
                                        care_temperature=care_temperature, fusion_mode=fusion_mode))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, 
                                     dropout, use_norm=last_norm, use_RTE=use_RTE, 
                                     care_temperature=care_temperature, fusion_mode=fusion_mode))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type, type_feature_dims=None):
        """
        Args:
            node_feature: [N, max_dim] padded features
            node_type: [N] node type indices
            edge_time: [E] edge time information
            edge_index: [2, E] edge indices
            edge_type: [E] edge type indices
            type_feature_dims: dict {type_id: actual_feature_dim} for heterogeneous graphs
        """
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            # Get actual feature dimension for this type
            if type_feature_dims is not None and t_id in type_feature_dims:
                actual_dim = type_feature_dims[t_id]
                # Use only the actual dimension (slice the padded features)
                type_features = node_feature[idx][:, :actual_dim]
            else:
                # Use all features (backward compatibility)
                type_features = node_feature[idx]
            res[idx] = torch.tanh(self.adapt_ws[t_id](type_features))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs


class HeteroHGT(torch.nn.Module):
    """
    HGT model wrapper that works with PyTorch Geometric heterogeneous graph format
    
    This wrapper converts PyTorch Geometric format (x_dict, edge_index_dict) to HGT format
    and provides a classifier head for node classification.
    
    Args:
        in_channels_dict: Dictionary of input dimensions per node type
                        e.g., {"tx": 183, "addr": 56}
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension (number of classes)
        num_types: Number of node types (default: 2 for tx and addr)
        num_relations: Number of relation types (default: 4 for tx->tx, addr->addr, tx->addr, addr->tx)
        n_heads: Number of attention heads (default: 4)
        num_layers: Number of GNN layers (default: 2)
        dropout: Dropout rate (default: 0.2)
        conv_name: Convolution type ('hgt', 'hgt_care', 'dense_hgt') (default: 'hgt')
        prev_norm: Use normalization in previous layers (default: False)
        last_norm: Use normalization in last layer (default: False)
        use_RTE: Use Relative Temporal Encoding (default: True)
        care_temperature: Temperature for CARE scoring (default: 1.0)
        fusion_mode: Fusion mode for CARE ('log_add' or 'prob_mul') (default: 'log_add')
        node_type_mapping: Mapping from node type name to type id (default: {"tx": 0, "addr": 1})
        relation_mapping: Mapping from relation tuple to relation id
    """
    def __init__(
        self,
        in_channels_dict,
        hidden_channels=64,
        out_channels=2,
        num_types=2,
        num_relations=4,
        n_heads=4,
        num_layers=2,
        dropout=0.2,
        conv_name='hgt',
        prev_norm=False,
        last_norm=False,
        use_RTE=True,
        care_temperature=1.0,
        fusion_mode='log_add',
        node_type_mapping=None,
        relation_mapping=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.num_relations = num_relations
        self.node_type_mapping = node_type_mapping or {"tx": 0, "addr": 1}
        self.relation_mapping = relation_mapping or {
            ("tx", "to", "tx"): 0,
            ("addr", "to", "addr"): 1,
            ("tx", "to", "addr"): 2,
            ("addr", "to", "tx"): 3,
        }
        
        # Create type_feature_dims dict
        type_feature_dims = {}
        for node_type, type_id in self.node_type_mapping.items():
            if node_type in in_channels_dict:
                type_feature_dims[type_id] = in_channels_dict[node_type]
        
        # Get max dimension for padding
        max_dim = max(in_channels_dict.values()) if in_channels_dict else 64
        
        # Create in_dims dict
        in_dims = {type_id: in_channels_dict.get(node_type, max_dim) 
                  for node_type, type_id in self.node_type_mapping.items()}
        
        # HGT GNN backbone
        self.gnn = HGTGNN(
            in_dims=in_dims,
            n_hid=hidden_channels,
            num_types=num_types,
            num_relations=num_relations,
            n_heads=n_heads,
            n_layers=num_layers,
            dropout=dropout,
            conv_name=conv_name,
            prev_norm=prev_norm,
            last_norm=last_norm,
            use_RTE=use_RTE,
            care_temperature=care_temperature,
            fusion_mode=fusion_mode,
        )
        
        # Classifier head
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.type_feature_dims = type_feature_dims
        self.max_dim = max_dim
    
    def convert_to_hgt_format(self, x_dict, edge_index_dict):
        """
        Convert PyTorch Geometric format to HGT format
        
        Args:
            x_dict: Dictionary of node features {node_type: [N_type, F_type]}
            edge_index_dict: Dictionary of edge indices {edge_type: [2, E]}
        
        Returns:
            node_feature: [N, max_dim] padded features
            node_type: [N] node type indices
            edge_time: [E] edge time (dummy if not available)
            edge_index: [2, E] edge indices
            edge_type: [E] edge type indices
        """
        device = next(iter(x_dict.values())).device
        
        # Collect all nodes
        node_features_list = []
        node_types_list = []
        node_offsets = {}  # {node_type: offset}
        
        current_offset = 0
        for node_type, features in x_dict.items():
            node_offsets[node_type] = current_offset
            num_nodes = features.size(0)
            feat_dim = features.size(1)
            
            # Pad features to max_dim
            if feat_dim < self.max_dim:
                padding = torch.zeros(num_nodes, self.max_dim - feat_dim, device=device)
                padded_features = torch.cat([features, padding], dim=1)
            else:
                padded_features = features[:, :self.max_dim]
            
            node_features_list.append(padded_features)
            type_id = self.node_type_mapping.get(node_type, 0)
            node_types_list.append(torch.full((num_nodes,), type_id, dtype=torch.long, device=device))
            current_offset += num_nodes
        
        node_feature = torch.cat(node_features_list, dim=0)  # [N, max_dim]
        node_type = torch.cat(node_types_list, dim=0)  # [N]
        
        # Collect all edges
        edge_index_list = []
        edge_type_list = []
        
        for edge_type_tuple, edge_index in edge_index_dict.items():
            if edge_type_tuple not in self.relation_mapping:
                continue
            
            rel_id = self.relation_mapping[edge_type_tuple]
            src_type, _, dst_type = edge_type_tuple
            
            # Adjust node indices with offsets
            src_offset = node_offsets.get(src_type, 0)
            dst_offset = node_offsets.get(dst_type, 0)
            
            adjusted_edge_index = edge_index.clone()
            adjusted_edge_index[0] += src_offset
            adjusted_edge_index[1] += dst_offset
            
            edge_index_list.append(adjusted_edge_index)
            edge_type_list.append(torch.full((edge_index.size(1),), rel_id, dtype=torch.long, device=device))
        
        if len(edge_index_list) > 0:
            edge_index = torch.cat(edge_index_list, dim=1)  # [2, E]
            edge_type = torch.cat(edge_type_list, dim=0)  # [E]
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type = torch.empty((0,), dtype=torch.long, device=device)
        
        # Dummy edge_time (can be replaced with actual time if available)
        edge_time = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
        
        return node_feature, node_type, edge_time, edge_index, edge_type
    
    def forward(self, x_dict, edge_index_dict, return_node_embeddings=False):
        """
        Forward pass
        
        Args:
            x_dict: Dictionary of node features {node_type: [N_type, F_type]}
            edge_index_dict: Dictionary of edge indices {edge_type: [2, E]}
            return_node_embeddings: If True, return node embeddings instead of logits
        
        Returns:
            If return_node_embeddings=False: Dictionary of logits {node_type: [N_type, out_channels]}
            If return_node_embeddings=True: Dictionary of embeddings {node_type: [N_type, hidden_channels]}
        """
        # Convert to HGT format
        node_feature, node_type, edge_time, edge_index, edge_type = self.convert_to_hgt_format(
            x_dict, edge_index_dict
        )
        
        # Forward through HGT GNN
        node_embeddings = self.gnn(
            node_feature, node_type, edge_time, edge_index, edge_type, 
            type_feature_dims=self.type_feature_dims
        )  # [N, hidden_channels]
        
        # Split embeddings back by node type
        result_dict = {}
        current_offset = 0
        for node_type_name, features in x_dict.items():
            num_nodes = features.size(0)
            type_embeddings = node_embeddings[current_offset:current_offset + num_nodes]
            
            if return_node_embeddings:
                result_dict[node_type_name] = type_embeddings
            else:
                # Apply classifier
                result_dict[node_type_name] = self.classifier(type_embeddings)
            
            current_offset += num_nodes
        
        return result_dict


# ============================================================
# CARE-GNN Models - Imported from care_gnn_model.py
# ============================================================
from care_gnn_model import OneLayerCARE


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
        "metapath": MetaPathTxClassifier3Path,
        "homogat": HomoGAT,
        "care": OneLayerCARE,
        "hgt": HeteroHGT,
    }
    
    # Add gtnlite if available
    try:
        from model import HeteroGTNLite
        models["gtnlite"] = HeteroGTNLite
    except (ImportError, NameError):
        pass  # HeteroGTNLite not defined, skip it
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    # HomoGAT uses in_channels (homogeneous graph)
    if model_name == "homogat":
        if in_channels is None:
            raise ValueError(f"{model_name.upper()} model requires 'in_channels'")
        return models[model_name](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            **kwargs
        )
    # HGT model requires in_channels_dict
    elif model_name == "hgt":
        if in_channels_dict is None:
            if in_channels is None:
                raise ValueError(f"{model_name.upper()} model requires either 'in_channels_dict' or 'in_channels'")
            # Fallback: use same dimension for all node types
            in_channels_dict = {"tx": in_channels, "addr": in_channels}
        
        return models[model_name](
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_types=kwargs.get("num_types", 2),
            num_relations=kwargs.get("num_relations", 4),
            n_heads=kwargs.get("n_heads", 4),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.2),
            conv_name=kwargs.get("conv_name", "hgt"),
            prev_norm=kwargs.get("prev_norm", False),
            last_norm=kwargs.get("last_norm", False),
            use_RTE=kwargs.get("use_RTE", True),
            care_temperature=kwargs.get("care_temperature", 1.0),
            fusion_mode=kwargs.get("fusion_mode", "log_add"),
        )
    # GAT and MetaPathTxClassifier models require in_channels_dict
    elif model_name == "gat" or model_name == "metapath":
        if in_channels_dict is None:
            if in_channels is None:
                raise ValueError(f"{model_name.upper()} model requires either 'in_channels_dict' or 'in_channels'")
            # Fallback: use same dimension for all node types (not ideal but works)
            in_channels_dict = {"tx": in_channels, "addr": in_channels}
        
        if model_name == "metapath":
            # MetaPathTxClassifier specific kwargs
            metapath_kwargs = {
                "in_channels_dict": in_channels_dict,
                "hidden_channels": hidden_channels,
                "emb_dim": kwargs.get("emb_dim", 64),
                "num_layers_tat": kwargs.get("num_layers_tat", 2),
                "num_layers_ata": kwargs.get("num_layers_ata", 2),
                "heads": kwargs.get("heads", 4),
                "dropout": kwargs.get("dropout", 0.5),
                "num_classes": out_channels,
                "semantic_hidden_dim": kwargs.get("semantic_hidden_dim", 32),
            }
            return models[model_name](**metapath_kwargs)
        else:
            # GAT model
            return models[model_name](
                in_channels_dict=in_channels_dict,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                **kwargs
            )
    # CARE-GNN model
    elif model_name == "care":
        if in_channels is None:
            if in_channels_dict is not None:
                # Use tx node dimension if available
                in_channels = in_channels_dict.get("tx", in_channels_dict.get(list(in_channels_dict.keys())[0]))
            else:
                raise ValueError(f"{model_name.upper()} model requires 'in_channels' or 'in_channels_dict'")
        
        return models[model_name](
            in_channels=in_channels,
            embed_dim=kwargs.get("embed_dim", hidden_channels),
            num_classes=out_channels,
            relation_names=kwargs.get("relation_names", None),
            lambda_1=kwargs.get("lambda_1", 1.0),
            inter=kwargs.get("inter", "GNN"),
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