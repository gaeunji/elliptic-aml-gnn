# Anti-Money Laundering Detection

Heterogeneous 그래프 신경망을 사용한 비트코인 거래 이상 탐지 프로젝트입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

```bash
python main.py --model sage --epochs 200
```

### 주요 옵션

- `--model`: 모델 선택 (sage, gat, gcn)
- `--data_path`: 데이터 파일 경로
- `--epochs`: 학습 에포크 수
- `--hidden_channels`: 은닉층 차원
- `--lr`: 학습률

## 모델

- **SAGE**: GraphSAGE 기반 이종 그래프 모델
- **GAT**: Graph Attention Network 기반 모델
- **GCN**: Graph Convolutional Network 기반 모델

