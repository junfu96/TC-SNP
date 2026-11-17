# -*- coding: utf-8 -*-
"""
TC-SNP implementation for CSI300 with a one-trading-day prediction horizon.

Input : the preceding 60 trading days of six daily features for each stock.
Target: next trading day's raw close-to-close return:
        y[i,t+1] = close[i,t+1] / close[i,t] - 1.

The train/validation/test split is determined from the dataset's calendar years:
- training: all years before the penultimate year;
- validation: the penultimate year;
- testing: the final year.
For a 2018-2023 CSI300 file, this gives 2018-2021 / 2022 / 2023.

This script follows the TC-SNP architecture and settings described in the
manuscript. The correlation lookback (250 trading days) is retained from the
uploaded CSI300 script because the manuscript defines T_g but does not give
its numerical value. The spike-generation function f(.) is implemented with
 tanh; confirm this choice against the exact LSTM-SNP implementation used in
your experiments if it differs.
"""

import os
import copy
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Experiment settings (edit DATA_PATH for your local machine)

DATA_PATH = "/home/liyuante/dataset/hs300_2018_2023_new_1.csv"
OUTPUT_DIR = "./csi300_tcsnp_nextday_results"

FEATURE_COLS = ["open", "close", "high", "low", "volume", "turnover"]
HISTORICAL_WINDOW = 60          # T = 60 trading days
HISTORICAL_STATE_LENGTH = 10    # L = 10 preceding internal states
CORRELATION_LOOKBACK = 250      # T_g; retained from the uploaded CSI300 code
CORRELATION_THRESHOLD = 0.8

TEMPORAL_HIDDEN_DIMS = (32, 10)  # two stacked layers in each direction
GAT_HEADS = 4
GAT_HEAD_DIM_1 = 32
GAT_HEAD_DIM_2 = 4
LATENT_STATE_COUNT = 8
FEATURE_DIM = 32
MHCA_HEADS = 4
PREDICTION_HIDDEN_DIM = 32

BATCH_SIZE = 32
LEARNING_RATE = 0.0002
NUM_RUNS = 15
NUM_EPOCHS = 20
RANDOM_SEED = 2026
TOP_K = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility and preprocessing

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cross_sectional_standardize(values: np.ndarray) -> np.ndarray:
    """3-sigma clipping followed by daily cross-sectional z-score."""
    x = values.astype(np.float64, copy=True)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)

    mean = np.nanmean(x)
    std = np.nanstd(x, ddof=1)
    if not np.isfinite(std) or std < 1e-12:
        x[~finite] = mean if np.isfinite(mean) else 0.0
        return np.zeros_like(x, dtype=np.float32)

    x[~finite] = mean
    x = np.clip(x, mean - 3.0 * std, mean + 3.0 * std)
    mean2 = x.mean()
    std2 = x.std(ddof=1)
    if not np.isfinite(std2) or std2 < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mean2) / std2).astype(np.float32)


@dataclass
class MarketPanel:
    dates: List[pd.Timestamp]
    codes: List[str]
    features: np.ndarray       # [D, N, 6], daily cross-sectionally standardized
    closes: np.ndarray         # [D, N], raw close prices
    returns: np.ndarray        # [D, N], raw close-to-close returns


def load_market_panel(filename: str) -> MarketPanel:
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"CSV file not found: {filename}\n"
            "Please change DATA_PATH or pass --data with the correct path."
        )

    df = pd.read_csv(filename)
    required = ["dt", "kdcode", "close"] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"The CSV is missing required columns: {missing}")

    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce").dt.normalize()
    df["kdcode"] = df["kdcode"].astype(str)
    for col in set(FEATURE_COLS + ["close"]):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["dt", "kdcode", "close"])
    df = df.sort_values(["dt", "kdcode"])
    df = df.drop_duplicates(subset=["dt", "kdcode"], keep="last")

    dates = sorted(df["dt"].unique().tolist())
    if len(dates) <= HISTORICAL_WINDOW:
        raise ValueError(
            f"Only {len(dates)} trading dates were found; at least "
            f"{HISTORICAL_WINDOW + 1} are required."
        )

    # Keep a fixed stock universe with an observation on every trading date.
    # This preserves a consistent node order across daily stock graphs.
    date_count = df.groupby("kdcode")["dt"].nunique()
    codes = sorted(date_count[date_count == len(dates)].index.tolist())
    if len(codes) < TOP_K:
        raise ValueError(
            f"Only {len(codes)} stocks have complete daily records. "
            "At least 10 are required for the Top-10 portfolio. Check the CSV."
        )

    df = df[df["kdcode"].isin(codes)].copy()
    dates = sorted(df["dt"].unique().tolist())

    close_panel = (
        df.pivot(index="dt", columns="kdcode", values="close")
        .reindex(index=dates, columns=codes)
    )
    if close_panel.isna().any().any():
        raise ValueError(
            "The selected stock universe still contains missing close prices. "
            "Please inspect duplicate or missing daily records."
        )
    closes = close_panel.to_numpy(dtype=np.float64)

    # Standardize each feature across stocks separately on each trading day.
    feature_panel = np.zeros((len(dates), len(codes), len(FEATURE_COLS)), dtype=np.float32)
    grouped = {d: g.set_index("kdcode").reindex(codes) for d, g in df.groupby("dt")}
    for t, date in enumerate(dates):
        day = grouped[date]
        for j, col in enumerate(FEATURE_COLS):
            vals = day[col].to_numpy(dtype=np.float64)
            if not np.isfinite(vals).any():
                vals = np.zeros(len(codes), dtype=np.float64)
            feature_panel[t, :, j] = cross_sectional_standardize(vals)

    returns = np.full_like(closes, np.nan, dtype=np.float64)
    returns[1:] = closes[1:] / np.maximum(closes[:-1], 1e-12) - 1.0

    print(f"Trading dates: {len(dates)}")
    print(f"Complete stocks: {len(codes)}")
    print(f"Feature shape: {feature_panel.shape}")
    print(f"Raw close-to-close return shape: {returns.shape}")

    return MarketPanel(
        dates=dates,
        codes=codes,
        features=feature_panel,
        closes=closes,
        returns=returns,
    )



# Dynamic stock graph: |Pearson correlation| >= threshold, with self-loops
def build_graph_edges(
    returns: np.ndarray,
    origin_index: int,
    lookback: int = CORRELATION_LOOKBACK,
    threshold: float = CORRELATION_THRESHOLD,
) -> np.ndarray:
    start = max(1, origin_index - lookback + 1)
    window = returns[start:origin_index + 1]
    num_stocks = returns.shape[1]

    if window.shape[0] < 2:
        corr = np.eye(num_stocks, dtype=np.float64)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(window, rowvar=False)
        corr = np.asarray(corr, dtype=np.float64)
        if corr.ndim == 0:
            corr = np.eye(num_stocks, dtype=np.float64)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = 0.5 * (corr + corr.T)

    adjacency = np.abs(corr) >= threshold
    np.fill_diagonal(adjacency, True)
    src, dst = np.where(adjacency)
    return np.stack([src, dst], axis=0).astype(np.int64)


# Build one sample per prediction date. The label is the raw next-day return.
class StockDateDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


def build_samples(panel: MarketPanel) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    samples: List[Dict[str, Any]] = []
    num_dates = len(panel.dates)

    # target_index is t+1; the input window contains dates t-59,...,t.
    for target_index in range(HISTORICAL_WINDOW, num_dates):
        origin_index = target_index - 1
        target_date = panel.dates[target_index]
        origin_date = panel.dates[origin_index]

        x = panel.features[target_index - HISTORICAL_WINDOW:target_index]
        x = np.transpose(x, (1, 0, 2)).astype(np.float32)  # [N, 60, 6]
        y = panel.returns[target_index].astype(np.float32)  # close[t+1]/close[t]-1

        if not np.isfinite(y).all():
            continue

        edge_index = build_graph_edges(panel.returns, origin_index)
        samples.append({
            "x": x,
            "y": y,
            "edge_index": edge_index,
            "origin_date": origin_date.strftime("%Y-%m-%d"),
            "target_date": target_date.strftime("%Y-%m-%d"),
            "target_year": int(target_date.year),
        })

    if not samples:
        raise ValueError("No valid one-step-ahead samples could be generated.")

    final_year = max(s["target_year"] for s in samples)
    validation_year = final_year - 1
    split_counts = {
        "train": sum(s["target_year"] < validation_year for s in samples),
        "validation": sum(s["target_year"] == validation_year for s in samples),
        "test": sum(s["target_year"] == final_year for s in samples),
        "validation_year": validation_year,
        "test_year": final_year,
    }

    train_samples = [s for s in samples if s["target_year"] < validation_year]
    val_samples = [s for s in samples if s["target_year"] == validation_year]
    test_samples = [s for s in samples if s["target_year"] == final_year]

    if not train_samples or not val_samples or not test_samples:
        raise ValueError(
            "The automatic chronological split produced an empty subset. "
            f"Counts: {split_counts}. Ensure the CSV spans at least three calendar years."
        )

    print(
        "Chronological split by target date: "
        f"train < {validation_year}, validation = {validation_year}, "
        f"test = {final_year}"
    )
    print(
        f"Samples: train={len(train_samples)}, "
        f"validation={len(val_samples)}, test={len(test_samples)}"
    )

    return train_samples, val_samples, test_samples, split_counts


def collate_stock_dates(batch: List[Dict[str, Any]]):
    x = torch.from_numpy(np.stack([item["x"] for item in batch])).float()  # [B,N,T,F]
    y = torch.from_numpy(np.stack([item["y"] for item in batch])).float()  # [B,N]
    num_stocks = x.shape[1]

    edge_parts = []
    for batch_index, item in enumerate(batch):
        edge = torch.from_numpy(item["edge_index"]).long()
        edge_parts.append(edge + batch_index * num_stocks)
    edge_index = torch.cat(edge_parts, dim=1)

    target_dates = [item["target_date"] for item in batch]
    origin_dates = [item["origin_date"] for item in batch]
    return x, y, edge_index, target_dates, origin_dates


# Att-LSTM-SNP and bidirectional temporal branch
class AttLSTMSNPCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, history_length: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.history_length = history_length
        self.attention_dim = hidden_dim
        self.value_dim = hidden_dim

        self.W_q = nn.Linear(hidden_dim, self.attention_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, self.attention_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, self.value_dim, bias=False)

        self.W_rm = nn.Linear(self.value_dim, hidden_dim, bias=False)
        self.W_rx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))

        self.W_c = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_c = nn.Parameter(torch.zeros(hidden_dim))

        self.W_o = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))

        self.W_a = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_a = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        x_t: torch.Tensor,
        u_prev: torch.Tensor,
        history_states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_t.shape[0]
        recent = history_states[-self.history_length:]
        valid_count = len(recent)

        if valid_count:
            hist = torch.stack(recent, dim=1)  # [B, L_valid, H]
            if valid_count < self.history_length:
                pad = torch.zeros(
                    batch_size,
                    self.history_length - valid_count,
                    self.hidden_dim,
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
                hist = torch.cat([pad, hist], dim=1)
            mask = torch.zeros(
                batch_size, self.history_length, device=x_t.device, dtype=torch.bool
            )
            mask[:, -valid_count:] = True
        else:
            hist = torch.zeros(
                batch_size,
                self.history_length,
                self.hidden_dim,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            mask = torch.zeros(
                batch_size, self.history_length, device=x_t.device, dtype=torch.bool
            )

        q = self.W_q(u_prev).unsqueeze(1)  # [B,1,d_a]
        k = self.W_k(hist)                 # [B,L,d_a]
        v = self.W_v(hist)                 # [B,L,d_v]
        scores = torch.sum(k * q, dim=-1) / (self.attention_dim ** 0.5)
        scores = scores.masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        context = torch.bmm(alpha.unsqueeze(1), v).squeeze(1)

        reset_gate = torch.sigmoid(
            self.W_rm(context) + self.W_rx(x_t) + self.U_r(u_prev) + self.b_r
        )
        consumption_gate = torch.sigmoid(
            self.W_c(x_t) + self.U_c(u_prev) + self.b_c
        )
        output_gate = torch.sigmoid(
            self.W_o(x_t) + self.U_o(u_prev) + self.b_o
        )

        spike = torch.tanh(self.W_a(x_t) + self.U_a(u_prev) + self.b_a)

        u_t = reset_gate * u_prev - consumption_gate * spike
        h_t = output_gate * spike
        return u_t, h_t


class BiAttLSTMSNP(nn.Module):
    def __init__(
        self,
        input_dim: int = len(FEATURE_COLS),
        hidden_dims: Tuple[int, int] = TEMPORAL_HIDDEN_DIMS,
        history_length: int = HISTORICAL_STATE_LENGTH,
    ):
        super().__init__()
        h1, h2 = hidden_dims

        # Independent forward and backward parameter sets.
        self.forward_cell_1 = AttLSTMSNPCell(input_dim, h1, history_length)
        self.forward_cell_2 = AttLSTMSNPCell(h1, h2, history_length)
        self.backward_cell_1 = AttLSTMSNPCell(input_dim, h1, history_length)
        self.backward_cell_2 = AttLSTMSNPCell(h1, h2, history_length)

    @staticmethod
    def _run_direction(
        x: torch.Tensor,
        cells: Tuple[AttLSTMSNPCell, AttLSTMSNPCell],
        reverse: bool,
    ) -> torch.Tensor:
        if reverse:
            x = torch.flip(x, dims=[1])

        for cell in cells:
            batch_size, seq_len, _ = x.shape
            u = torch.zeros(batch_size, cell.hidden_dim, device=x.device, dtype=x.dtype)
            history: List[torch.Tensor] = []
            outputs = []
            for t in range(seq_len):
                u, h = cell(x[:, t, :], u, history)
                history.append(u)
                if len(history) > cell.history_length:
                    history.pop(0)
                outputs.append(h)
            x = torch.stack(outputs, dim=1)

        return x[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, 60, 6]
        forward_last = self._run_direction(
            x, (self.forward_cell_1, self.forward_cell_2), reverse=False
        )
        backward_first_aligned = self._run_direction(
            x, (self.backward_cell_1, self.backward_cell_2), reverse=True
        )
        return torch.cat([forward_last, backward_first_aligned], dim=-1)  # [B*N,20]



# GAT-SNP
def segment_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax edge scores over incoming edges of each destination node."""
    num_heads = scores.shape[1]
    index = dst.unsqueeze(1).expand(-1, num_heads)
    max_per_node = torch.full(
        (num_nodes, num_heads),
        -torch.inf,
        device=scores.device,
        dtype=scores.dtype,
    )
    max_per_node.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    exp_scores = torch.exp(scores - max_per_node[dst])
    denom = torch.zeros(
        (num_nodes, num_heads), device=scores.device, dtype=scores.dtype
    )
    denom.index_add_(0, dst, exp_scores)
    return exp_scores / denom[dst].clamp_min(1e-12)


class GATSNPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim_per_head: int,
        num_heads: int,
        attention_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim_per_head = output_dim_per_head
        self.num_heads = num_heads
        self.attention_dim = attention_dim

        self.W_self = nn.Parameter(torch.empty(num_heads, attention_dim, input_dim))
        self.W_neigh = nn.Parameter(torch.empty(num_heads, attention_dim, input_dim))
        self.W_agg = nn.Parameter(torch.empty(num_heads, output_dim_per_head, input_dim))
        self.att_vector = nn.Parameter(torch.empty(num_heads, attention_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads, output_dim_per_head))

        for p in (self.W_self, self.W_neigh, self.W_agg):
            nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.att_vector)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [total_nodes, input_dim]; edge_index[0]=source j, edge_index[1]=target i
        x_nonlinear = F.gelu(x)
        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        self_proj = torch.einsum("nd,kad->nka", x_nonlinear, self.W_self)
        neigh_proj = torch.einsum("nd,kad->nka", x_nonlinear, self.W_neigh)
        pair_features = self_proj[dst] + neigh_proj[src]
        scores = F.leaky_relu(
            torch.einsum("eka,ka->ek", pair_features, self.att_vector),
            negative_slope=0.2,
        )
        alpha = segment_softmax(scores, dst, num_nodes)

        transformed = torch.einsum("nd,kod->nko", x_nonlinear, self.W_agg)
        messages = transformed[src] * alpha.unsqueeze(-1)
        aggregated = torch.zeros(
            num_nodes,
            self.num_heads,
            self.output_dim_per_head,
            device=x.device,
            dtype=x.dtype,
        )
        aggregated.index_add_(0, dst, messages)
        aggregated = aggregated + self.bias.unsqueeze(0)
        return aggregated.reshape(num_nodes, self.num_heads * self.output_dim_per_head)


class GATSNP(nn.Module):
    def __init__(self, input_dim: int = HISTORICAL_WINDOW * len(FEATURE_COLS)):
        super().__init__()
        self.layer1 = GATSNPLayer(
            input_dim=input_dim,
            output_dim_per_head=GAT_HEAD_DIM_1,
            num_heads=GAT_HEADS,
            attention_dim=GAT_HEAD_DIM_1,
        )
        self.layer2 = GATSNPLayer(
            input_dim=GAT_HEADS * GAT_HEAD_DIM_1,
            output_dim_per_head=GAT_HEAD_DIM_2,
            num_heads=GAT_HEADS,
            attention_dim=GAT_HEAD_DIM_2,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x, edge_index)  # [B*N, 128]
        x = self.layer2(x, edge_index)  # [B*N, 16]
        return x


# TC-SNP
class TCSNP(nn.Module):
    def __init__(self):
        super().__init__()
        temporal_dim = 2 * TEMPORAL_HIDDEN_DIMS[-1]  # 20
        graph_dim = GAT_HEADS * GAT_HEAD_DIM_2      # 16

        self.temporal_branch = BiAttLSTMSNP()
        self.graph_branch = GATSNP()
        self.temporal_projection = nn.Linear(temporal_dim, FEATURE_DIM)
        self.graph_projection = nn.Linear(graph_dim, FEATURE_DIM)

        self.latent_temporal = nn.Parameter(
            torch.randn(1, LATENT_STATE_COUNT, FEATURE_DIM) * 0.02
        )
        self.latent_cross_sectional = nn.Parameter(
            torch.randn(1, LATENT_STATE_COUNT, FEATURE_DIM) * 0.02
        )
        self.latent_attention_temporal = nn.MultiheadAttention(
            FEATURE_DIM, MHCA_HEADS, batch_first=True
        )
        self.latent_attention_cross = nn.MultiheadAttention(
            FEATURE_DIM, MHCA_HEADS, batch_first=True
        )
        self.mhca = nn.MultiheadAttention(FEATURE_DIM, MHCA_HEADS, batch_first=True)
        self.layer_norm = nn.LayerNorm(FEATURE_DIM)

        self.prediction_head = nn.Sequential(
            nn.Linear(FEATURE_DIM, PREDICTION_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(PREDICTION_HIDDEN_DIM, 1),
        )

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x_seq: [B,N,T,F]; graph is a disjoint union of B daily stock graphs.
        batch_size, num_stocks, time_steps, num_features = x_seq.shape
        flat_seq = x_seq.reshape(batch_size * num_stocks, time_steps, num_features)

        temporal = self.temporal_branch(flat_seq).reshape(
            batch_size, num_stocks, 2 * TEMPORAL_HIDDEN_DIMS[-1]
        )
        graph_input = x_seq.reshape(batch_size * num_stocks, time_steps * num_features)
        graph = self.graph_branch(graph_input, edge_index).reshape(
            batch_size, num_stocks, GAT_HEADS * GAT_HEAD_DIM_2
        )

        A1 = self.temporal_projection(temporal)  # [B,N,32]
        A2 = self.graph_projection(graph)        # [B,N,32]

        R1 = self.latent_temporal.expand(batch_size, -1, -1)
        R2 = self.latent_cross_sectional.expand(batch_size, -1, -1)
        Z1, _ = self.latent_attention_temporal(R1, A1, A1, need_weights=False)
        Z2, _ = self.latent_attention_cross(R2, A2, A2, need_weights=False)

        context = torch.cat([A2, Z1, Z2], dim=1)  # [B,N+16,32]
        cross_output, _ = self.mhca(A1, context, context, need_weights=False)
        fused = self.layer_norm(A1 + A2 + cross_output)

        # One unrestricted real-valued prediction per stock; no ReLU/sigmoid.
        return self.prediction_head(fused).squeeze(-1)  # [B,N]


# Training, validation, prediction averaging, and Top-10 portfolio output
def evaluate_mse(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    squared_error_sum = 0.0
    element_count = 0
    with torch.no_grad():
        for x, y, edge_index, _, _ in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            edge_index = edge_index.to(DEVICE)
            pred = model(x, edge_index)
            squared_error_sum += F.mse_loss(pred, y, reduction="sum").item()
            element_count += y.numel()
    return squared_error_sum / max(element_count, 1)


def predict(model: nn.Module, loader: DataLoader):
    model.eval()
    predictions = []
    actuals = []
    target_dates: List[str] = []
    origin_dates: List[str] = []
    with torch.no_grad():
        for x, y, edge_index, dates, origins in loader:
            x = x.to(DEVICE)
            edge_index = edge_index.to(DEVICE)
            pred = model(x, edge_index)
            predictions.append(pred.cpu().numpy())
            actuals.append(y.numpy())
            target_dates.extend(dates)
            origin_dates.extend(origins)
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(actuals, axis=0),
        target_dates,
        origin_dates,
    )


def portfolio_summary(daily_returns: np.ndarray) -> Dict[str, float]:
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    if len(daily_returns) == 0:
        return {"AR": np.nan, "AL": np.nan, "MD": np.nan, "AS": np.nan}

    wealth = np.cumprod(1.0 + daily_returns)
    annualized_return = float(wealth[-1] ** (252.0 / len(daily_returns)) - 1.0)
    annualized_volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252.0)) if len(daily_returns) > 1 else 0.0
    wealth_with_initial = np.concatenate([[1.0], wealth])
    running_peak = np.maximum.accumulate(wealth_with_initial)[1:]
    drawdown = wealth / np.maximum(running_peak, 1e-12) - 1.0
    max_drawdown = float(np.min(drawdown))
    return_to_volatility = (
        float(annualized_return / annualized_volatility)
        if annualized_volatility > 1e-12 else np.nan
    )
    return {
        "AR": annualized_return,
        "AL": annualized_volatility,
        "MD": max_drawdown,
        "AS": return_to_volatility,
    }


def run_experiment(data_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    panel = load_market_panel(data_path)
    train_samples, val_samples, test_samples, split_counts = build_samples(panel)

    train_loader = DataLoader(
        StockDateDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_stock_dates,
        drop_last=False,
    )
    val_loader = DataLoader(
        StockDateDataset(val_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_stock_dates,
        drop_last=False,
    )
    test_loader = DataLoader(
        StockDateDataset(test_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_stock_dates,
        drop_last=False,
    )

    all_run_predictions = []
    actual_returns = None
    test_target_dates = None
    test_origin_dates = None
    run_summaries = []

    for run_id in range(NUM_RUNS):
        seed = RANDOM_SEED + run_id
        set_seed(seed)
        model = TCSNP().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        best_val_mse = float("inf")
        best_epoch = 0
        best_state = None

        print(f"\n===== Run {run_id + 1}/{NUM_RUNS} | seed={seed} =====")
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for x, y, edge_index, _, _ in train_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                edge_index = edge_index.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                pred = model(x, edge_index)
                loss = F.mse_loss(pred, y)
                loss.backward()
                optimizer.step()

                train_loss_sum += F.mse_loss(pred.detach(), y, reduction="sum").item()
                train_count += y.numel()

            train_mse = train_loss_sum / max(train_count, 1)
            val_mse = evaluate_mse(model, val_loader)
            print(
                f"Run {run_id + 1:02d} | Epoch {epoch + 1:02d}/{NUM_EPOCHS} "
                f"| train MSE={train_mse:.8f} | val MSE={val_mse:.8f}"
            )

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("No valid model checkpoint was selected on the validation set.")

        model.load_state_dict(best_state)
        run_pred, run_actual, dates, origins = predict(model, test_loader)
        all_run_predictions.append(run_pred)
        if actual_returns is None:
            actual_returns = run_actual
            test_target_dates = dates
            test_origin_dates = origins
        else:
            if not np.allclose(actual_returns, run_actual, atol=1e-7, rtol=1e-6):
                raise RuntimeError("Actual test labels changed unexpectedly between runs.")

        run_mse = float(np.mean((run_pred - run_actual) ** 2))
        run_mae = float(np.mean(np.abs(run_pred - run_actual)))
        run_summaries.append({
            "run": run_id + 1,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_mse": best_val_mse,
            "test_mse": run_mse,
            "test_mae": run_mae,
        })
        print(
            f"Run {run_id + 1} selected epoch {best_epoch}; "
            f"test MSE={run_mse:.8f}, test MAE={run_mae:.8f}"
        )

    # Average the predictions from all 15 independent runs, as stated in the paper.
    mean_predictions = np.mean(np.stack(all_run_predictions, axis=0), axis=0)
    test_mse = float(np.mean((mean_predictions - actual_returns) ** 2))
    test_mae = float(np.mean(np.abs(mean_predictions - actual_returns)))

    prediction_rows = []
    for date_idx, target_date in enumerate(test_target_dates):
        origin_date = test_origin_dates[date_idx]
        for stock_idx, code in enumerate(panel.codes):
            prediction_rows.append({
                "signal_date": origin_date,
                "target_date": target_date,
                "kdcode": code,
                "actual_next_day_return": float(actual_returns[date_idx, stock_idx]),
                "predicted_next_day_return": float(mean_predictions[date_idx, stock_idx]),
            })

    prediction_df = pd.DataFrame(prediction_rows)
    prediction_path = os.path.join(output_dir, "csi300_test_nextday_predictions_mean.csv")
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    # Daily Top-10 equal-weight portfolio using the averaged next-day predictions.
    daily_portfolio_returns = []
    portfolio_rows = []
    k = min(TOP_K, len(panel.codes))
    for date_idx, target_date in enumerate(test_target_dates):
        selected_idx = np.argsort(mean_predictions[date_idx])[-k:][::-1]
        realized = float(np.mean(actual_returns[date_idx, selected_idx]))
        daily_portfolio_returns.append(realized)
        portfolio_rows.append({
            "signal_date": test_origin_dates[date_idx],
            "target_date": target_date,
            "selected_stocks": ";".join(panel.codes[i] for i in selected_idx),
            "portfolio_return": realized,
        })

    portfolio_df = pd.DataFrame(portfolio_rows)
    portfolio_path = os.path.join(output_dir, "csi300_top10_equal_weight_daily_returns.csv")
    portfolio_df.to_csv(portfolio_path, index=False, encoding="utf-8-sig")

    run_summary_df = pd.DataFrame(run_summaries)
    run_summary_path = os.path.join(output_dir, "csi300_run_summary.csv")
    run_summary_df.to_csv(run_summary_path, index=False, encoding="utf-8-sig")

    portfolio_metrics = portfolio_summary(np.asarray(daily_portfolio_returns))
    summary = {
        "prediction_horizon_trading_days": 1,
        "historical_input_trading_days": HISTORICAL_WINDOW,
        "num_stocks": len(panel.codes),
        "num_test_dates": len(test_target_dates),
        "num_runs": NUM_RUNS,
        "test_mse_of_averaged_predictions": test_mse,
        "test_mae_of_averaged_predictions": test_mae,
        **portfolio_metrics,
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, "csi300_nextday_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n===== Final results (15-run averaged predictions) =====")
    print(f"Test MSE: {test_mse:.8f}")
    print(f"Test MAE: {test_mae:.8f}")
    print("Top-10 portfolio metrics:")
    for key, value in portfolio_metrics.items():
        print(f"  {key}: {value:.6f}" if np.isfinite(value) else f"  {key}: NaN")
    print("\nSaved files:")
    print(f"  Predictions: {prediction_path}")
    print(f"  Portfolio returns: {portfolio_path}")
    print(f"  Run summary: {run_summary_path}")
    print(f"  Final summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TC-SNP CSI300 next-day return prediction")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Path to CSI300 CSV")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()
    run_experiment(args.data, args.output)
