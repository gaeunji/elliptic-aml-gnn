import os
import sys
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import HeteroData


# ─────────────────────────────────────
# ID 정규화: Tx / Addr / Edge에서 모두 사용
# ─────────────────────────────────────
def normalize_id(val):
    """
    Tx / Addr ID를 문자열로 통일하는 함수.
    - NaN -> None
    - 1.0 -> '1'
    - '123.0' -> '123'
    - 앞뒤 공백 제거
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "":
        return None
    # float처럼 생긴 경우 정수로 떨어지면 int 문자열로
    try:
        f = float(s)
        i = int(f)
        if f == i:
            return str(i)
    except Exception:
        pass
    # 끝이 ".0"인 경우 잘라줌
    if s.endswith(".0"):
        s = s[:-2]
    return s


# ─────────────────────────────────────
# defaultdict helper들 (pickle 호환)
# ─────────────────────────────────────
def _default_dict():
    return {}


def _default_list():
    return []


def _default_none():
    return None


def _edge_level5():
    return []  # 리스트로 변경하여 중복 허용


def _edge_level4():
    return defaultdict(_edge_level5)


def _edge_level3():
    return defaultdict(_edge_level4)


def _edge_level2():
    return defaultdict(_edge_level3)


def _edge_level1():
    return defaultdict(_edge_level2)


class BlockchainGraph():
    """
    블록체인 트랜잭션 데이터를 HGT / HeteroGNN에 사용할 수 있도록 전처리하는 클래스
    
    노드 타입:
    - 'Transaction'
    - 'Address'
    
    edge_list 구조:
      edge_list[target_type][source_type][relation_type][target_idx][source_idx] = time
    """

    def __init__(self):
        self.node_forward = defaultdict(_default_dict)   # type -> {id -> idx}
        self.node_feature = defaultdict(_default_list)   # type -> [feature_array, ...] (메모리 효율적)
        
        self.edge_list = defaultdict(_edge_level1)
        self.times = {}

    def add_node(self, node):
        """노드 추가 - feature만 저장하여 메모리 효율적"""
        nfl = self.node_forward[node['type']]
        node_id = node['id']
        if node_id not in nfl:
            ser = len(nfl)
            nfl[node_id] = ser
            
            # feature만 저장 (메모리 효율적)
            if 'features' in node:
                self.node_feature[node['type']].append(node['features'])
            else:
                # feature가 없으면 빈 배열 추가 (나중에 차원 확인 후 처리)
                self.node_feature[node['type']].append(None)
            
            return ser
        return nfl[node_id]

    def add_edge(self, source_node, target_node, time=None, relation_type=None, directed=False):
        """
        엣지 추가 (노드가 이미 존재해야 함)
        내부 저장: edge_list[target_type][source_type][relation_type][target_idx][source_idx] = [time, ...]
        중복 엣지를 허용하기 위해 리스트로 저장
        """
        source_type = source_node['type']
        target_type = target_node['type']
        source_id = source_node['id']
        target_id = target_node['id']

        if source_id not in self.node_forward[source_type]:
            return False
        if target_id not in self.node_forward[target_type]:
            return False

        s_idx = self.node_forward[source_type][source_id]
        t_idx = self.node_forward[target_type][target_id]

        # 중복 허용: 리스트에 추가
        self.edge_list[target_type][source_type][relation_type][t_idx][s_idx].append(time)

        if directed:
            # 역방향 relation도 별도 타입으로 추가
            rev_rel = 'rev_' + relation_type
            self.edge_list[source_type][target_type][rev_rel][s_idx][t_idx].append(time)

        self.times[time] = True
        return True

    def update_node(self, node):
        """노드 업데이트 - feature만 업데이트"""
        node_type = node['type']
        ser = self.add_node(node)  # 이미 존재하면 idx 반환, 없으면 추가
        
        # feature만 업데이트
        if 'features' in node:
            feature_list = self.node_feature[node_type]
            if ser < len(feature_list):
                feature_list[ser] = node['features']
            # 이미 add_node에서 추가되었으므로 여기서는 업데이트만

    def get_meta_graph(self):
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas.append((target_type, source_type, r_type))
        return metas

    def get_types(self):
        return list(self.node_feature.keys())


# ─────────────────────────────────────
# 시간 step 인코딩
# ─────────────────────────────────────
def encode_time_step(time_step, min_time, max_time):
    if max_time == min_time:
        return np.array([0.0, 0.0], dtype=np.float32)
    normalized = 2 * np.pi * (time_step - min_time) / (max_time - min_time)
    sin_enc = np.sin(normalized).astype(np.float32)
    cos_enc = np.cos(normalized).astype(np.float32)
    return np.array([sin_enc, cos_enc], dtype=np.float32)


# ─────────────────────────────────────
# 데이터 로드 + 그래프 / 라벨 생성 (multi-task friendly)
# ─────────────────────────────────────
def load_blockchain_data(data_dir):
    """
    EllipticPlusPlus 데이터셋을 로드하고 BlockchainGraph + multi-task labels 생성.

    Returns:
        graph: BlockchainGraph
        labels: {
          'Transaction': {'y': np.ndarray [N_tx] (0/1/-1)},
          'Address': {'y': np.ndarray [N_addr] (0/1/-1)}
        }
    """
    print("Loading blockchain dataset...")

    txs_dir = os.path.join(data_dir, 'Transactions Dataset')
    actors_dir = os.path.join(data_dir, 'Actors Dataset')

    if os.path.exists(txs_dir) and os.path.exists(actors_dir):
        print("Detected EllipticPlusPlus dataset structure (subdirectories)")
        txs_data_dir = txs_dir
        actors_data_dir = actors_dir
    else:
        print("Using flat directory structure")
        txs_data_dir = data_dir
        actors_data_dir = data_dir

    # CSV 로드
    txs_features = pd.read_csv(os.path.join(txs_data_dir, 'txs_features.csv'))
    txs_classes = pd.read_csv(os.path.join(txs_data_dir, 'txs_classes.csv'))
    wallets_features = pd.read_csv(os.path.join(actors_data_dir, 'wallets_features.csv'))
    wallets_classes = pd.read_csv(os.path.join(actors_data_dir, 'wallets_classes.csv'))

    addr_tx_edges = pd.read_csv(os.path.join(actors_data_dir, 'AddrTx_edgelist.csv'))
    tx_addr_edges = pd.read_csv(os.path.join(actors_data_dir, 'TxAddr_edgelist.csv'))

    tx_tx_edges = None
    addr_addr_edges = None

    tx_tx_path = os.path.join(txs_data_dir, 'txs_edgelist.csv')
    if os.path.exists(tx_tx_path):
        tx_tx_edges = pd.read_csv(tx_tx_path)

    addr_addr_path = os.path.join(actors_data_dir, 'AddrAddr_edgelist.csv')
    if os.path.exists(addr_addr_path):
        addr_addr_edges = pd.read_csv(addr_addr_path)

    graph = BlockchainGraph()

    # ─────────────────────────────
    # 1) Transaction 노드
    # ─────────────────────────────
    print("Loading Transaction labels...")
    tx_label_col = txs_classes.columns[0]
    tx_class_col = txs_classes.columns[1]

    # 라벨 있는 Tx ID 집합 (정규화) - itertuples() 사용
    # 인덱스 기반 접근으로 안전하게 처리
    tx_label_idx_for_set = txs_classes.columns.get_loc(tx_label_col)
    labeled_tx_ids = set()
    for row in txs_classes.itertuples(index=False, name=None):
        tx_id = normalize_id(row[tx_label_idx_for_set])
        if tx_id is not None:
            labeled_tx_ids.add(tx_id)
    print(f"  Found {len(labeled_tx_ids):,} Transaction nodes with labels")

    print("Adding Transaction nodes (ALL txs from txs_features.csv)...")
    tx_id_col = txs_features.columns[0]

    # Time step 컬럼
    time_col = None
    if 'Time step' in txs_features.columns:
        time_col = 'Time step'
    elif 'time_step' in txs_features.columns:
        time_col = 'time_step'

    tx_feat_cols = [col for col in txs_features.columns
                    if col != tx_id_col and col != time_col]

    # 한 번의 루프로 time step 범위 찾기와 노드 추가 통합
    added_tx = 0
    tx_time_steps = []
    
    # 벡터화된 연산으로 time step 추출 (더 빠름)
    if time_col:
        tx_time_steps = txs_features[time_col].astype(int).tolist()
    else:
        tx_time_steps = [0] * len(txs_features)
    
    tx_min_time = min(tx_time_steps) if tx_time_steps else 0
    tx_max_time = max(tx_time_steps) if tx_time_steps else 0
    print(f"  Transaction time step range: {tx_min_time} to {tx_max_time}")

    # itertuples() 사용 (iterrows()보다 훨씬 빠름)
    # pandas가 컬럼 이름을 자동으로 정규화 (공백/특수문자 -> _)
    for idx, row in enumerate(tqdm(txs_features.itertuples(index=False), total=len(txs_features))):
        # 컬럼 이름 접근 (pandas가 자동으로 정규화)
        raw_id = getattr(row, tx_id_col.replace(' ', '_').replace('-', '_'), None)
        if raw_id is None:
            # 정규화 실패 시 원본 이름으로 시도
            raw_id = getattr(row, tx_id_col, None)
        tx_id = normalize_id(raw_id)
        if tx_id is None:
            continue

        # Feature 추출 (벡터화)
        feat_values = [getattr(row, col.replace(' ', '_').replace('-', '_'), 
                              getattr(row, col, None)) for col in tx_feat_cols]
        feats = np.array(feat_values, dtype=np.float32)
        # Fill NaN values with 0
        feats = np.nan_to_num(feats, nan=0.0)

        time_step = tx_time_steps[idx] if time_col else 0

        node = {
            'id': tx_id,
            'type': 'Transaction',
            'features': feats,
            'time': time_step
        }
        graph.add_node(node)
        added_tx += 1

    print(f"  Added {added_tx:,} Transaction nodes")
    print(f"  Transaction feature dimension: {len(tx_feat_cols)}")

    # ─────────────────────────────
    # 2) Address 노드
    # ─────────────────────────────
    print("Adding Address nodes (only from wallets_features.csv)...")
    addr_id_col = wallets_features.columns[0]

    addr_time_col = None
    if 'Time step' in wallets_features.columns:
        addr_time_col = 'Time step'
    elif 'time_step' in wallets_features.columns:
        addr_time_col = 'time_step'

    addr_feat_cols = [col for col in wallets_features.columns
                      if col != addr_id_col and col != addr_time_col]

    # 한 번의 루프로 time step 범위 찾기와 노드 추가 통합
    added_addr = 0
    
    # 벡터화된 연산으로 time step 추출 (더 빠름)
    if addr_time_col:
        addr_time_steps = wallets_features[addr_time_col].astype(int).tolist()
    else:
        addr_time_steps = [0] * len(wallets_features)
    
    addr_min_time = min(addr_time_steps) if addr_time_steps else 0
    addr_max_time = max(addr_time_steps) if addr_time_steps else 0
    print(f"  Address time step range: {addr_min_time} to {addr_max_time}")

    # itertuples() 사용 (iterrows()보다 훨씬 빠름)
    # pandas가 컬럼 이름을 자동으로 정규화 (공백/특수문자 -> _)
    for idx, row in enumerate(tqdm(wallets_features.itertuples(index=False), total=len(wallets_features))):
        # 컬럼 이름 접근 (pandas가 자동으로 정규화)
        raw_id = getattr(row, addr_id_col.replace(' ', '_').replace('-', '_'), None)
        if raw_id is None:
            # 정규화 실패 시 원본 이름으로 시도
            raw_id = getattr(row, addr_id_col, None)
        addr_id = normalize_id(raw_id)
        if addr_id is None:
            continue

        # Feature 추출 (벡터화)
        feat_values = [getattr(row, col.replace(' ', '_').replace('-', '_'), 
                              getattr(row, col, None)) for col in addr_feat_cols]
        feats = np.array(feat_values, dtype=np.float32)
        # Fill NaN values with 0
        feats = np.nan_to_num(feats, nan=0.0)

        time_step = addr_time_steps[idx] if addr_time_col else 0

        node = {
            'id': addr_id,
            'type': 'Address',
            'features': feats,
            'time': time_step
        }
        graph.add_node(node)
        added_addr += 1

    print(f"  Added {added_addr:,} Address nodes")
    print(f"  Address feature dimension: {len(addr_feat_cols)}")

    # ─────────────────────────────
    # 3) Edges: Address -> Transaction
    # ─────────────────────────────
    print("Adding Address->Transaction edges...")
    if 'input_address' in addr_tx_edges.columns and 'txId' in addr_tx_edges.columns:
        source_col = 'input_address'
        target_col = 'txId'
    else:
        source_col = addr_tx_edges.columns[0]
        target_col = addr_tx_edges.columns[1]

    edge_count = 0
    skipped_tx = 0
    skipped_addr = 0
    
    # itertuples() 사용 (iterrows()보다 훨씬 빠름)
    # 인덱스 기반 접근으로 안전하게 처리
    source_idx = addr_tx_edges.columns.get_loc(source_col)
    target_idx = addr_tx_edges.columns.get_loc(target_col)
    
    for row in tqdm(addr_tx_edges.itertuples(index=False, name=None), total=len(addr_tx_edges)):
        s_id = normalize_id(row[source_idx])
        t_id = normalize_id(row[target_idx])
        if s_id is None or t_id is None:
            continue

        if s_id not in graph.node_forward['Address']:
            skipped_addr += 1
            continue
        if t_id not in graph.node_forward['Transaction']:
            skipped_tx += 1
            continue

        s_node = {'id': s_id, 'type': 'Address'}
        t_node = {'id': t_id, 'type': 'Transaction'}

        if graph.add_edge(s_node, t_node, time=0,
                          relation_type='addr_to_tx', directed=False):
            edge_count += 1

    print(f"  Added {edge_count:,} edges")
    print(f"  Skipped {skipped_tx:,} edges (transactions not in graph)")
    print(f"  Skipped {skipped_addr:,} edges (addresses not in wallets_features.csv)")

    # ─────────────────────────────
    # 4) Edges: Transaction -> Address
    # ─────────────────────────────
    print("Adding Transaction->Address edges...")
    if 'txId' in tx_addr_edges.columns and 'output_address' in tx_addr_edges.columns:
        source_col = 'txId'
        target_col = 'output_address'
    else:
        source_col = tx_addr_edges.columns[0]
        target_col = tx_addr_edges.columns[1]

    edge_count = 0
    skipped_tx = 0
    skipped_addr = 0
    
    # itertuples() 사용 (iterrows()보다 훨씬 빠름)
    # 인덱스 기반 접근으로 안전하게 처리
    source_idx = tx_addr_edges.columns.get_loc(source_col)
    target_idx = tx_addr_edges.columns.get_loc(target_col)
    
    for row in tqdm(tx_addr_edges.itertuples(index=False, name=None), total=len(tx_addr_edges)):
        s_id = normalize_id(row[source_idx])
        t_id = normalize_id(row[target_idx])
        if s_id is None or t_id is None:
            continue

        if s_id not in graph.node_forward['Transaction']:
            skipped_tx += 1
            continue
        if t_id not in graph.node_forward['Address']:
            skipped_addr += 1
            continue

        s_node = {'id': s_id, 'type': 'Transaction'}
        t_node = {'id': t_id, 'type': 'Address'}

        if graph.add_edge(s_node, t_node, time=0,
                          relation_type='tx_to_addr', directed=False):
            edge_count += 1

    print(f"  Added {edge_count:,} edges")
    print(f"  Skipped {skipped_tx:,} edges (transactions not in graph)")
    print(f"  Skipped {skipped_addr:,} edges (addresses not in wallets_features.csv)")

    # ─────────────────────────────
    # 5) Edges: Transaction -> Transaction (선택)
    # ─────────────────────────────
    if tx_tx_edges is not None:
        print("Adding Transaction->Transaction edges...")
        if 'txId1' in tx_tx_edges.columns and 'txId2' in tx_tx_edges.columns:
            source_col = 'txId1'
            target_col = 'txId2'
        else:
            source_col = tx_tx_edges.columns[0]
            target_col = tx_tx_edges.columns[1]

        edge_count = 0
        skipped_edge = 0
        
        # itertuples() 사용 (iterrows()보다 훨씬 빠름)
        # 인덱스 기반 접근으로 안전하게 처리
        source_idx = tx_tx_edges.columns.get_loc(source_col)
        target_idx = tx_tx_edges.columns.get_loc(target_col)
        
        for row in tqdm(tx_tx_edges.itertuples(index=False, name=None), total=len(tx_tx_edges)):
            s_id = normalize_id(row[source_idx])
            t_id = normalize_id(row[target_idx])
            if s_id is None or t_id is None:
                continue

            if s_id not in graph.node_forward['Transaction'] or t_id not in graph.node_forward['Transaction']:
                skipped_edge += 1
                continue

            s_node = {'id': s_id, 'type': 'Transaction'}
            t_node = {'id': t_id, 'type': 'Transaction'}

            if graph.add_edge(s_node, t_node, time=0,
                              relation_type='tx_to_tx', directed=False):
                edge_count += 1

        print(f"  Added {edge_count:,} edges, skipped {skipped_edge:,} edges (transactions not in graph)")

    # ─────────────────────────────
    # 6) Edges: Address -> Address (선택)
    # ─────────────────────────────
    if addr_addr_edges is not None:
        print("Adding Address->Address edges...")
        if 'input_address' in addr_addr_edges.columns and 'output_address' in addr_addr_edges.columns:
            source_col = 'input_address'
            target_col = 'output_address'
        else:
            source_col = addr_addr_edges.columns[0]
            target_col = addr_addr_edges.columns[1]

        edge_count = 0
        skipped = 0
        
        # itertuples() 사용 (iterrows()보다 훨씬 빠름)
        # 인덱스 기반 접근으로 안전하게 처리
        source_idx = addr_addr_edges.columns.get_loc(source_col)
        target_idx = addr_addr_edges.columns.get_loc(target_col)
        
        for row in tqdm(addr_addr_edges.itertuples(index=False, name=None), total=len(addr_addr_edges)):
            s_id = normalize_id(row[source_idx])
            t_id = normalize_id(row[target_idx])
            if s_id is None or t_id is None:
                continue

            if s_id not in graph.node_forward['Address'] or t_id not in graph.node_forward['Address']:
                skipped += 1
                continue

            s_node = {'id': s_id, 'type': 'Address'}
            t_node = {'id': t_id, 'type': 'Address'}

            if graph.add_edge(s_node, t_node, time=0,
                              relation_type='addr_to_addr', directed=False):
                edge_count += 1

        print(f"  Added {edge_count:,} edges")
        print(f"  Skipped {skipped:,} edges (addresses not in wallets_features.csv)")

    # ─────────────────────────────
    # 7) node_feature 행렬로 변환
    # ─────────────────────────────
    print("Converting node features to matrix...")
    for node_type in ['Transaction', 'Address']:
        if node_type in graph.node_feature:
            feature_list = graph.node_feature[node_type]
            if len(feature_list) == 0:
                continue
            
            # feature 리스트를 numpy 배열로 변환
            node_features = []
            feature_dim = None
            
            for feat in feature_list:
                if feat is not None:
                    if feature_dim is None:
                        feature_dim = len(feat)
                    node_features.append(feat)
                else:
                    # feature가 None인 경우 (이론적으로는 발생하지 않아야 함)
                    if feature_dim is None:
                        feature_dim = 0
                    node_features.append(np.zeros(feature_dim, dtype=np.float32))
            
            if len(node_features) > 0:
                feats = np.stack(node_features)
                # Fill any remaining NaN values with 0 (safety check)
                feats = np.nan_to_num(feats, nan=0.0)
                graph.node_feature[node_type] = feats
                print(f"  Converted {len(node_features)} {node_type} nodes to feature matrix (shape: {feats.shape})")
            else:
                # feature_list를 빈 배열로 유지
                graph.node_feature[node_type] = np.array([], dtype=np.float32).reshape(0, 0)

    # ─────────────────────────────
    # 8) 레이블 처리 (multi-task용 dense vector)
    # ─────────────────────────────
    print("Processing labels for multi-task...")

    labels = {}

    # Transaction labels: raw 1/2/3 → mapped 1(illicit)/0(licit)/-1(unlabeled)
    num_tx_nodes = len(graph.node_forward['Transaction'])
    tx_y = np.full((num_tx_nodes,), -1, dtype=np.int64)
    raw_tx_label_counts = defaultdict(int)
    mapped_tx_counts = {0: 0, 1: 0, -1: 0}

    # itertuples() 사용 (iterrows()보다 훨씬 빠름)
    # 인덱스 기반 접근으로 안전하게 처리
    tx_label_idx = txs_classes.columns.get_loc(tx_label_col)
    tx_class_idx = txs_classes.columns.get_loc(tx_class_col)
    
    for row in txs_classes.itertuples(index=False, name=None):
        tx_id = normalize_id(row[tx_label_idx])
        if tx_id is None:
            continue
        if tx_id not in graph.node_forward['Transaction']:
            continue
        node_idx = graph.node_forward['Transaction'][tx_id]
        raw_label = int(row[tx_class_idx])
        raw_tx_label_counts[raw_label] += 1

        if raw_label == 1:
            mapped = 1  # illicit
        elif raw_label == 2:
            mapped = 0  # licit (정상)
        else:  # 3 or 기타
            mapped = -1  # unlabeled

        tx_y[node_idx] = mapped
        mapped_tx_counts[mapped] += 1

    print("  Tx raw label counts:", dict(raw_tx_label_counts))
    print("  Tx mapped label counts (0,1,-1):", mapped_tx_counts)
    labels['Transaction'] = {'y': tx_y}

    # Address labels: raw 1/2/3 → mapped 1(illicit)/0(legal)/-1(unlabeled)
    num_addr_nodes = len(graph.node_forward['Address'])
    addr_y = np.full((num_addr_nodes,), -1, dtype=np.int64)
    addr_label_col = wallets_classes.columns[0]
    addr_class_col = wallets_classes.columns[1]

    raw_addr_label_counts = defaultdict(int)
    mapped_addr_counts = {0: 0, 1: 0, -1: 0}

    # itertuples() 사용 (iterrows()보다 훨씬 빠름)
    # 인덱스 기반 접근으로 안전하게 처리
    addr_label_idx = wallets_classes.columns.get_loc(addr_label_col)
    addr_class_idx = wallets_classes.columns.get_loc(addr_class_col)
    
    for row in wallets_classes.itertuples(index=False, name=None):
        addr_id = normalize_id(row[addr_label_idx])
        if addr_id is None:
            continue
        if addr_id not in graph.node_forward['Address']:
            continue
        node_idx = graph.node_forward['Address'][addr_id]
        raw_label = int(row[addr_class_idx])
        raw_addr_label_counts[raw_label] += 1

        if raw_label == 1:      # illicit
            mapped = 1
        elif raw_label == 2:    # legal
            mapped = 0
        else:                   # 3 = unknown
            mapped = -1

        addr_y[node_idx] = mapped
        mapped_addr_counts[mapped] += 1

    print("  Addr raw label counts:", dict(raw_addr_label_counts))
    print("  Addr mapped label counts (0,1,-1):", mapped_addr_counts)
    labels['Address'] = {'y': addr_y}

    # ─────────────────────────────
    # 9) Graph 통계
    # ─────────────────────────────
    print("\nGraph Statistics:")
    print(f"- Transaction nodes: {len(graph.node_forward['Transaction'])}")
    print(f"- Address nodes: {len(graph.node_forward['Address'])}")
    print(f"- Edge types: {len(graph.get_meta_graph())}")
    print(f"- Meta graph: {graph.get_meta_graph()}")

    return graph, labels


# ─────────────────────────────────────
# BlockchainGraph + labels → HeteroData 변환
# ─────────────────────────────────────
def convert_to_heterodata(graph, labels, device='cpu', random_state=42):
    """
    BlockchainGraph + labels를 PyG HeteroData로 변환.
    노드 타입은 'Transaction'->'tx', 'Address'->'addr'로 매핑.
    train/val/test mask를 stratified split으로 생성.
    """
    print("\nConverting to HeteroData format...")
    data = HeteroData()

    # 노드 피처
    if 'Transaction' in graph.node_feature:
        x_tx = torch.from_numpy(graph.node_feature['Transaction']).float()
        data['tx'].x = x_tx
        print(f"  Transaction features: {x_tx.shape}")
    if 'Address' in graph.node_feature:
        x_addr = torch.from_numpy(graph.node_feature['Address']).float()
        data['addr'].x = x_addr
        print(f"  Address features: {x_addr.shape}")

    # 라벨
    if 'Transaction' in labels:
        tx_y = torch.from_numpy(labels['Transaction']['y']).long()
        data['tx'].y = tx_y
        
        # Transaction train/val/test mask 생성 (stratified split)
        print("  Creating Transaction train/val/test masks (stratified split)...")
        tx_y_np = tx_y.numpy()
        
        # 라벨이 있는 노드만 사용 (0, 1만, -1 제외)
        labeled_mask = (tx_y_np == 0) | (tx_y_np == 1)
        labeled_indices = np.where(labeled_mask)[0]
        labeled_labels = tx_y_np[labeled_indices]
        
        if len(labeled_indices) > 0:
            # Train 60%, Val 20%, Test 20%
            train_indices, temp_indices, train_labels_split, temp_labels = train_test_split(
                labeled_indices,
                labeled_labels,
                test_size=0.3,
                stratify=labeled_labels,
                random_state=random_state
            )
            
            val_indices, test_indices, val_labels_split, test_labels_split = train_test_split(
                temp_indices,
                temp_labels,
                test_size=0.5,
                stratify=temp_labels,
                random_state=random_state
            )
            
            train_mask = torch.zeros(tx_y.size(0), dtype=torch.bool)
            val_mask = torch.zeros(tx_y.size(0), dtype=torch.bool)
            test_mask = torch.zeros(tx_y.size(0), dtype=torch.bool)
            
            train_mask[train_indices] = True
            val_mask[val_indices] = True
            test_mask[test_indices] = True
            
            data['tx'].train_mask = train_mask
            data['tx'].val_mask = val_mask
            data['tx'].test_mask = test_mask
            
            print(f"    Train: {train_mask.sum().item():,} nodes")
            print(f"    Val:   {val_mask.sum().item():,} nodes")
            print(f"    Test:  {test_mask.sum().item():,} nodes")
        else:
            print("    Warning: No labeled Transaction nodes found for mask creation")
    
    if 'Address' in labels:
        addr_y = torch.from_numpy(labels['Address']['y']).long()
        data['addr'].y = addr_y
        
        # Address train/val/test mask 생성 (stratified split)
        print("  Creating Address train/val/test masks (stratified split)...")
        addr_y_np = addr_y.numpy()
        
        # 라벨이 있는 노드만 사용 (0, 1만, -1 제외)
        labeled_mask = (addr_y_np == 0) | (addr_y_np == 1)
        labeled_indices = np.where(labeled_mask)[0]
        labeled_labels = addr_y_np[labeled_indices]
        
        if len(labeled_indices) > 0:
            # Train 60%, Val 20%, Test 20%
            train_indices, temp_indices, train_labels_split, temp_labels = train_test_split(
                labeled_indices,
                labeled_labels,
                test_size=0.3,
                stratify=labeled_labels,
                random_state=random_state
            )
            
            val_indices, test_indices, val_labels_split, test_labels_split = train_test_split(
                temp_indices,
                temp_labels,
                test_size=0.5,
                stratify=temp_labels,
                random_state=random_state
            )
            
            train_mask = torch.zeros(addr_y.size(0), dtype=torch.bool)
            val_mask = torch.zeros(addr_y.size(0), dtype=torch.bool)
            test_mask = torch.zeros(addr_y.size(0), dtype=torch.bool)
            
            train_mask[train_indices] = True
            val_mask[val_indices] = True
            test_mask[test_indices] = True
            
            data['addr'].train_mask = train_mask
            data['addr'].val_mask = val_mask
            data['addr'].test_mask = test_mask
            
            print(f"    Train: {train_mask.sum().item():,} nodes")
            print(f"    Val:   {val_mask.sum().item():,} nodes")
            print(f"    Test:  {test_mask.sum().item():,} nodes")
        else:
            print("    Warning: No labeled Address nodes found for mask creation")

    # 엣지
    total_edges = 0
    edge_stats = defaultdict(int)

    for target_type in graph.edge_list:
        for source_type in graph.edge_list[target_type]:
            for rel_type, tgt_dict in graph.edge_list[target_type][source_type].items():
                # src / dst 타입 매핑
                if source_type == 'Transaction':
                    src_ty = 'tx'
                elif source_type == 'Address':
                    src_ty = 'addr'
                else:
                    continue

                if target_type == 'Transaction':
                    dst_ty = 'tx'
                elif target_type == 'Address':
                    dst_ty = 'addr'
                else:
                    continue

                src_indices = []
                dst_indices = []
                times = []

                for t_idx, srcs in tgt_dict.items():
                    for s_idx, t_vals in srcs.items():
                        # t_vals는 리스트 (중복 허용)
                        if isinstance(t_vals, list):
                            for t_val in t_vals:
                                src_indices.append(s_idx)
                                dst_indices.append(t_idx)
                                times.append(0 if t_val is None else t_val)
                        else:
                            # 기존 호환성 (단일 값인 경우)
                            src_indices.append(s_idx)
                            dst_indices.append(t_idx)
                            times.append(0 if t_vals is None else t_vals)

                if len(src_indices) == 0:
                    continue

                edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
                data[(src_ty, rel_type, dst_ty)].edge_index = edge_index
                data[(src_ty, rel_type, dst_ty)].edge_time = torch.tensor(times, dtype=torch.long)

                num_e = edge_index.size(1)
                total_edges += num_e
                edge_stats[(src_ty, rel_type, dst_ty)] += num_e

    for k, v in edge_stats.items():
        print(f"  Edge type {k}: {v:,} edges")
    print(f"  Total edges: {total_edges:,}")

    data = data.to(device)
    return data


# ─────────────────────────────────────
# 저장 / 로드 함수 (선택)
# ─────────────────────────────────────
def save_pt(data, output_path):
    print(f"\nSaving HeteroData to {output_path}...")
    torch.save(data, output_path)
    print("Save complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build EllipticPlusPlus hetero graph (.pt)")
    parser.add_argument("--data_dir", type=str, default="EllipticPlusPlus",
                        help="Path to EllipticPlusPlus dataset root")
    parser.add_argument("--output", type=str, default="elliptic_hetero_static.pt",
                        help="Output .pt path")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    graph, labels = load_blockchain_data(data_dir)
    hetero_data = convert_to_heterodata(graph, labels, device='cpu')
    save_pt(hetero_data, os.path.abspath(args.output))
