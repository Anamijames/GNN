import os
PUBLIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'public'))
os.makedirs(PUBLIC_DIR, exist_ok=True)
import argparse
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
HAS_PYG = True
try:
    from torch_geometric.data import Data
    # Prefer the newer loader location but fall back for older PyG releases
    try:
        from torch_geometric.loader import DataLoader as PyGDataLoader
    except ImportError:
        from torch_geometric.data import DataLoader as PyGDataLoader
    from torch_geometric.nn import GCNConv
    try:
        from torch_geometric.nn import GATConv
    except ImportError:
        GATConv = None
except ImportError:
    # Don't hard-fail at import time. Allow --help and --generate-sample to work without PyG.
    HAS_PYG = False
    Data = None
    PyGDataLoader = None
    GCNConv = None
    GATConv = None
    print("Warning: torch_geometric not found. Install it to run training/evaluation. See requirements.txt and PyG docs.")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import shutil
PUBLIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'public'))

# -------------------------
# Configuration / constants
# -------------------------
CLEANED_CSV_OUT = "cleaned_it_sector_data.csv"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# -----------------------------------------------------
# 1. Data loading and preprocessing (robust)
# -----------------------------------------------------
def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns and strip to improve robustness to input variations."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # keep original case for "Company" mapping later; but here lower-case mapping for lookups
    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the first matching column name from candidates (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise KeyError(f"None of columns {candidates} found in dataframe. Available: {list(df.columns)}")


def validate_no_nans(dfs: List[pd.DataFrame]) -> None:
    """Raise if any DataFrame in list has NaN/Inf; used as final guard."""
    for i, df in enumerate(dfs):
        if not np.isfinite(df.values).all():
            # Provide some diagnostic info
            nan_locs = np.argwhere(~np.isfinite(df.values))
            sample = nan_locs[:5]
            raise ValueError(f"Validation failed: DataFrame #{i} contains non-finite values. "
                             f"Sample indices: {sample}")


def load_it_sector_data(excel_path: str, save_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str], pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """
    Load IT sector data from Excel and build cleaned feature + target tensors.

    Returns:
        features: np.ndarray [num_dates, num_companies, num_features]
        targets (returns): np.ndarray [num_dates, num_companies]
        dates: list of pd.Timestamp
        companies: list of str
        close_scaled: pd.DataFrame (per-company scaled close)
        vol20: pd.DataFrame (per-company rolling vol)
        sectors_map: Optional[dict] mapping company->sector
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Could not find {excel_path}")

    # Support both Excel and CSV inputs (CSV is convenient for systems without openpyxl)
    if str(excel_path).lower().endswith('.csv'):
        df = pd.read_csv(excel_path)
    else:
        try:
            df = pd.read_excel(excel_path)
        except (ImportError, ValueError, OSError):
            # Fallback: try reading as CSV if excel reader backend missing or unreadable
            df = pd.read_csv(excel_path)
    if df.shape[0] == 0:
        raise ValueError("Excel file appears empty.")

    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["Date", "Company", "Close", "Open", "High", "Low", "Volume"]
    alt_map = {
        "Date": ["date", "trade_date"],
        "Company": ["company", "symbol", "ticker", "code"],
        "Close": ["close", "close price", "close_price"],
        "Open": ["open", "open price", "open_price"],
        "High": ["high"],
        "Low": ["low"],
        "Volume": ["volume", "vol"],
    }
    col_map = {}
    for col in required_cols:
        if col in df.columns:
            col_map[col] = col
        else:
            for alt in alt_map[col]:
                if alt in df.columns:
                    col_map[col] = alt
                    break
            else:
                raise ValueError(f"Required column '{col}' not found in Excel file.")

    df = df.rename(columns={col_map[k]: k for k in required_cols})
    df = df.dropna(subset=["Date", "Company", "Close", "Open", "High", "Low", "Volume"])
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df[df["Volume"] > 0]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).drop_duplicates(subset=["Date", "Company"], keep="last")
    df = df.sort_values(["Date", "Company"]).reset_index(drop=True)

    for c in ["Close", "Open", "High", "Low", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    companies = sorted(df["Company"].unique().tolist())
    if len(companies) == 0:
        raise ValueError("No companies found after filtering. Check Company column in Excel.")

    close = df.pivot_table(index="Date", columns="Company", values="Close")
    volume = df.pivot_table(index="Date", columns="Company", values="Volume")
    dates_index = pd.DatetimeIndex(sorted(df["Date"].unique()))
    close = close.reindex(dates_index)
    volume = volume.reindex(dates_index)

    close = close.ffill().bfill().fillna(0.0)
    volume = volume.ffill().bfill().fillna(1.0)

    for col in close.columns:
        low, high = np.nanpercentile(close[col].values, [1, 99])
        if low < high:
            close[col] = close[col].clip(lower=low, upper=high)
    for col in volume.columns:
        low, high = np.nanpercentile(volume[col].values, [1, 99])
        if low < high:
            volume[col] = volume[col].clip(lower=low, upper=high)

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ma5 = close.rolling(window=5, min_periods=1).mean().ffill().bfill()
    vol20 = returns.rolling(window=20, min_periods=1).std().replace([np.inf, -np.inf], np.nan).fillna(0.0).ffill().bfill()

    close = close.ffill().bfill().fillna(0.0)
    volume = volume.ffill().bfill().fillna(1.0)
    ma5 = ma5.ffill().bfill().fillna(0.0)
    vol20 = vol20.fillna(0.0)
    returns = returns.fillna(0.0)

    validate_no_nans([close, volume, ma5, vol20, returns])

    cleaned_stack = pd.concat({"Close": close, "Volume": volume, "MA5": ma5, "VOL20": vol20, "Returns": returns}, axis=1)
    cleaned_stack.columns = ["_".join(map(str, tup)).strip() for tup in cleaned_stack.columns.values]
    cleaned_out_path = os.path.join(save_dir or PUBLIC_DIR, CLEANED_CSV_OUT)
    cleaned_stack.to_csv(cleaned_out_path)
    print(f"Saved cleaned data snapshot to {cleaned_out_path}")

    scaler_close = StandardScaler()
    scaler_vol = StandardScaler()
    scaler_ma5 = StandardScaler()
    scaler_vol20 = StandardScaler()

    close_scaled = pd.DataFrame(scaler_close.fit_transform(close.values), index=close.index, columns=close.columns)
    volume_scaled = pd.DataFrame(scaler_vol.fit_transform(volume.values), index=volume.index, columns=volume.columns)
    ma5_scaled = pd.DataFrame(scaler_ma5.fit_transform(ma5.values), index=ma5.index, columns=ma5.columns)
    vol20_scaled = pd.DataFrame(scaler_vol20.fit_transform(vol20.values), index=vol20.index, columns=vol20.columns)

    dates = list(close.index)
    num_dates = len(dates)
    num_companies = len(companies)
    base_num_features = 5

    # read raw df for sectors/macro columns; reuse same extension logic
    if str(excel_path).lower().endswith('.csv'):
        raw_df = pd.read_csv(excel_path)
    else:
        try:
            raw_df = pd.read_excel(excel_path)
        except (ImportError, ValueError, OSError):
            raw_df = pd.read_csv(excel_path)
    sectors_map = None
    sector_onehot = None
    if 'Sector' in raw_df.columns:
        sectors_map = raw_df.dropna(subset=['Company', 'Sector']).drop_duplicates('Company').set_index('Company')['Sector'].to_dict()
        unique_sectors = sorted(set(sectors_map.values()))
        sector_onehot = np.zeros((num_companies, len(unique_sectors)), dtype=np.float32)
        for j, c in enumerate(companies):
            s = sectors_map.get(c)
            if s is not None:
                sector_onehot[j, unique_sectors.index(s)] = 1.0

    macro_cols = [c for c in raw_df.columns if str(c).startswith('Macro_')]
    macro_matrix = None
    if len(macro_cols) > 0:
        macro_df = raw_df[['Date'] + macro_cols].drop_duplicates('Date').set_index('Date').reindex(dates_index)
        macro_df = macro_df.ffill().bfill().fillna(0.0)
        macro_matrix = macro_df.values.astype(np.float32)

    num_features = base_num_features + (sector_onehot.shape[1] if sector_onehot is not None else 0) + (macro_matrix.shape[1] if macro_matrix is not None else 0)
    features = np.zeros((num_dates, num_companies, num_features), dtype=np.float32)
    for j, c in enumerate(companies):
        features[:, j, 0] = close_scaled[c].values
        features[:, j, 1] = volume_scaled[c].values
        features[:, j, 2] = returns[c].values
        features[:, j, 3] = ma5_scaled[c].values
        features[:, j, 4] = vol20_scaled[c].values
        if sector_onehot is not None:
            features[:, j, base_num_features: base_num_features + sector_onehot.shape[1]] = sector_onehot[j]
        if macro_matrix is not None:
            mf_start = base_num_features + (sector_onehot.shape[1] if sector_onehot is not None else 0)
            features[:, j, mf_start: mf_start + macro_matrix.shape[1]] = macro_matrix

    targets = returns.values.astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite values found in features after scaling.")
    if not np.isfinite(targets).all():
        raise ValueError("Non-finite values found in targets after scaling.")

    return features, targets, dates, companies, close_scaled, vol20, sectors_map, scaler_close, scaler_vol20


def _generate_sample_excel(path: str, num_companies: int = 5, num_days: int = 120) -> None:
    """Generate a small synthetic Sector.xlsx file for quick testing.

    The generated file contains columns: Date, Company, Open, High, Low, Close, Volume
    with simple random-walk prices per company.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    start = pd.Timestamp('2020-01-01')
    dates = pd.date_range(start, periods=num_days, freq='B')  # business days
    companies = [f"COMP{i+1}" for i in range(num_companies)]

    rows = []
    for comp in companies:
        price = 100.0 + rng.normal(0, 1)
        for d in dates:
            # random walk
            ret = rng.normal(0, 0.01)
            open_p = price
            close_p = max(0.1, open_p * (1 + ret))
            high_p = max(open_p, close_p) * (1 + rng.uniform(0, 0.01))
            low_p = min(open_p, close_p) * (1 - rng.uniform(0, 0.01))
            volume = int(max(1, rng.integers(1000, 10000)))
            rows.append({'Date': d, 'Company': comp, 'Open': open_p, 'High': high_p, 'Low': low_p, 'Close': close_p, 'Volume': volume})
            price = close_p

    df = pd.DataFrame(rows)
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wrote_path = None
    # Always write a CSV fallback; prefer Excel if available
    csv_path = path
    if str(path).lower().endswith('.xlsx'):
        csv_path = os.path.splitext(path)[0] + '.csv'
    try:
        # Try Excel first
        df.to_excel(path, index=False)
        wrote_path = path
    except (ImportError, ValueError, OSError) as e:
        # If Excel backend not available or cannot write, write CSV instead
        try:
            df.to_csv(csv_path, index=False)
            wrote_path = csv_path
        except (OSError, ValueError) as e2:
            raise RuntimeError(f"Failed to write sample data to Excel or CSV: {e2}") from e2
    # Also ensure a CSV copy exists for convenience
    try:
        df.to_csv(os.path.splitext(path)[0] + '.csv', index=False)
    except (OSError, ValueError):
        pass
    return wrote_path


# -----------------------------------------------------
# 2. Graph construction (correlation-based)
# -----------------------------------------------------
def build_correlation_graph(returns: np.ndarray, threshold: float = 0.2, companies: Optional[List[str]] = None, sectors: Optional[dict] = None, same_sector_bonus: bool = True) -> torch.Tensor:
    """Build an undirected graph of companies based on return correlations.

    If `sectors` mapping is provided (company -> sector), edges between companies in the same
    sector will be included regardless of correlation (or treated with a relaxed threshold).
    """
    # returns: [num_dates, num_companies]
    if returns.ndim != 2:
        raise ValueError("returns should be 2D: [num_dates, num_companies]")

    # Correlation across companies
    # Use np.corrcoef with rows=companies by transposing
    corr = np.corrcoef(returns.T)
    num_nodes = corr.shape[0]

    edge_index_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            add_edge = False
            if np.abs(corr[i, j]) >= threshold:
                add_edge = True
            elif same_sector_bonus and sectors is not None and companies is not None:
                # if both companies present in sector map and share sector, include edge
                ci = companies[i]
                cj = companies[j]
                si = sectors.get(ci)
                sj = sectors.get(cj)
                if si is not None and sj is not None and si == sj:
                    add_edge = True
            if add_edge:
                edge_index_list.append([i, j])

    # Fallback to fully-connected graph if threshold too strict
    if len(edge_index_list) == 0:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index_list.append([i, j])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    return edge_index


# -----------------------------------------------------
# 3. Dataset for time-windowed graph snapshots
# -----------------------------------------------------
class TimeSeriesGraphDataset(torch.utils.data.Dataset):
    """Dataset that converts time series into windowed graph snapshots for GNNs."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        edge_index: torch.Tensor,
        window_size: int,
        start_t: int,
        end_t: int,
    ) -> None:
        super().__init__()
        self.features = features
        self.targets = targets
        self.edge_index = edge_index
        self.window_size = window_size
        self.start_t = start_t
        self.end_t = end_t

        # Basic checks
        num_dates, num_companies, num_features = features.shape
        if targets.shape[0] != num_dates or targets.shape[1] != num_companies:
            raise ValueError("Features and targets dimension mismatch.")

        if self.start_t < self.window_size:
            raise ValueError("start_t must be >= window_size to have a full context window")

    def __len__(self) -> int:
        return max(0, self.end_t - self.start_t)

    def __getitem__(self, idx: int):
        t = self.start_t + idx
        # Use previous `window_size` days to predict returns at day t
        window_feats = self.features[t - self.window_size : t]  # [W, N, F]
        window, num_nodes, num_feat = window_feats.shape

        # Permute to [num_nodes, window, num_feat]
        x_seq = window_feats.transpose(1, 0, 2).astype(np.float32)
        y = self.targets[t]  # [num_nodes]

        data = Data(
            x=torch.from_numpy(x_seq),  # [num_nodes, window, num_feat]
            edge_index=self.edge_index,
            y=torch.from_numpy(y),
        )
        return data


# -----------------------------------------------------
# 4. GNN model definition (GCN for regression)
# -----------------------------------------------------
class GNNTimeSeriesModel(nn.Module):
    """Temporal encoder (LSTM/Transformer) + GNN (GCN/GAT) hybrid for multi-node time-series regression."""

    def __init__(
        self,
        window_size: int,
        num_features: int,
        temporal_model: str = 'lstm',  # 'lstm' or 'transformer'
        gnn_type: str = 'gcn',  # 'gcn' or 'gat'
        hidden_lstm: int = 64,
        hidden_gnn: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.temporal_model = temporal_model.lower()

        # Temporal encoder: either LSTM or Transformer
        if self.temporal_model == 'lstm':
            self.lstm = nn.LSTM(
                input_size=num_features,
                hidden_size=hidden_lstm,
                num_layers=1,
                batch_first=False,  # we will feed [W, N, F]
            )
            self.temporal_out_dim = hidden_lstm
        elif self.temporal_model == 'transformer':
            # Simple transformer encoder: project features to hidden_lstm dims
            self.input_proj = nn.Linear(num_features, hidden_lstm)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_lstm, nhead=max(1, min(8, hidden_lstm // 8)))
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.temporal_out_dim = hidden_lstm
        else:
            raise ValueError("Unsupported temporal_model; choose 'lstm' or 'transformer'")

        # Graph convolution layers operating on temporal embeddings
        if gnn_type.lower() == 'gat' and GATConv is not None:
            self.conv1 = GATConv(self.temporal_out_dim, hidden_gnn, heads=1)
            self.conv2 = GATConv(hidden_gnn, hidden_gnn, heads=1)
        else:
            self.conv1 = GCNConv(self.temporal_out_dim, hidden_gnn)
            self.conv2 = GCNConv(hidden_gnn, hidden_gnn)

        self.lin = nn.Linear(hidden_gnn, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes_total_in_batch, window, num_features]
        edge_index: edge indices for the batched graph (provided by PyG DataLoader)
        """
        num_nodes_total, window, num_feat = x.shape
        assert window == self.window_size and num_feat == self.num_features

        # Temporal encoder
        # LSTM expects [seq_len, batch, input_size]
        x_seq = x.permute(1, 0, 2)  # [window, num_nodes_total, num_features]
        if self.temporal_model == 'lstm':
            _, (h_n, _) = self.lstm(x_seq)
            # Last layer hidden state: [num_nodes_total, hidden_lstm]
            h_last = h_n[-1]
        else:
            # transformer expects [seq_len, batch, d_model]
            proj = self.input_proj(x_seq)  # [window, num_nodes_total, d_model]
            trans_out = self.transformer(proj)
            # pool over sequence dimension (mean)
            h_last = trans_out.mean(dim=0)

        # Graph conv layers
        x_g = self.conv1(h_last, edge_index)
        x_g = torch.relu(x_g)
        x_g = self.dropout(x_g)

        x_g = self.conv2(x_g, edge_index)
        x_g = torch.relu(x_g)
        x_g = self.dropout(x_g)

        out = self.lin(x_g).squeeze(-1)  # [num_nodes_total]
        return out


# -----------------------------------------------------
# 5. Training and evaluation utilities
# -----------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    output_dir: Optional[str] = None,
    task: str = 'return',
) -> float:
    model.train()
    # Choose loss based on task
    if task == 'trend':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    total_loss = 0.0
    all_y_true = []
    all_y_pred = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        # For classification task (trend), ensure labels are floats
        if task == 'trend':
            y_true_batch = batch.y.float()
            loss = criterion(out, y_true_batch)
        else:
            loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        all_y_true.append(batch.y.cpu().numpy())
        all_y_pred.append(out.detach().cpu().numpy())

    # Draw and save a single plot at the end of the epoch (non-interactive)
    if len(all_y_true) > 0:
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        if task == 'trend':
            # show predicted probability histogram vs true labels
            probs = 1.0 / (1.0 + np.exp(-y_pred))
            ax.hist(probs, bins=20, alpha=0.6, label='pred_prob')
            ax.hist(y_true, bins=2, alpha=0.6, label='true_label')
            ax.set_xlabel('Predicted probability / True label')
            ax.set_ylabel('Count')
            ax.legend()
            ax.set_title('Predicted Probabilities (Train)')
        else:
            ax.scatter(y_true, y_pred, alpha=0.3, s=10)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Predicted vs Actual (Train)")
            lims = [min(float(y_true.min()), float(y_pred.min())), max(float(y_true.max()), float(y_pred.max()))]
            ax.plot(lims, lims, "r--", linewidth=1)
        out_dir = output_dir or PUBLIC_DIR
        per_epoch_name = os.path.join(out_dir, f"realtime_train_epoch{epoch:03d}.png")
        latest_name = os.path.join(out_dir, "realtime_train_latest.png")
        fig.savefig(per_epoch_name, dpi=200)
        # update latest copy with a safe file operation
        try:
            shutil.copyfile(per_epoch_name, latest_name)
        except (OSError, IOError) as e:
            # fallback: write the latest directly
            print(f"Warning: failed to copy training figure: {e}")
            fig.savefig(latest_name, dpi=200)
        plt.close(fig)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss

def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    epoch: int = 0,
    stage: str = "val",
    output_dir: Optional[str] = None,
    task: str = 'return',
):
    model.eval()
    if task == 'trend':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    total_loss = 0.0
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            if task == 'trend':
                y_true_batch = batch.y.float()
                loss = criterion(out, y_true_batch)
            else:
                loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            all_y_true.append(batch.y.cpu().numpy())
            all_y_pred.append(out.detach().cpu().numpy())

    # Save a single evaluation scatter plot
    if len(all_y_true) > 0:
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        if task == 'trend':
            probs = 1.0 / (1.0 + np.exp(-y_pred))
            ax.hist(probs, bins=20, alpha=0.6, label='pred_prob')
            ax.hist(y_true, bins=2, alpha=0.6, label='true_label')
            ax.set_xlabel('Predicted probability / True label')
            ax.set_ylabel('Count')
            ax.legend()
            ax.set_title(f'Predicted Probabilities ({stage})')
        else:
            ax.scatter(y_true, y_pred, alpha=0.3, s=10)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"Predicted vs Actual ({stage})")
            lims = [min(float(y_true.min()), float(y_pred.min())), max(float(y_true.max()), float(y_pred.max()))]
            ax.plot(lims, lims, "r--", linewidth=1)
        out_dir = output_dir or PUBLIC_DIR
        per_epoch_name = os.path.join(out_dir, f"realtime_eval_{stage}_epoch{epoch:03d}.png")
        latest_name = os.path.join(out_dir, f"realtime_eval_{stage}_latest.png")
        fig.savefig(per_epoch_name, dpi=200)
        try:
            shutil.copyfile(per_epoch_name, latest_name)
        except (OSError, IOError) as e:
            print(f"Warning: failed to copy eval figure: {e}")
            fig.savefig(latest_name, dpi=200)
        plt.close(fig)

    if len(all_y_true) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), np.array([]), np.array([])

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    # Guard against NaN/Inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        mse = float("nan")
        mae = float("nan")
        directional_accuracy = float("nan")
        avg_loss = total_loss / max(len(loader.dataset), 1)
        return avg_loss, mse, mae, directional_accuracy, y_true, y_pred

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    avg_loss = total_loss / max(len(loader.dataset), 1)
    if task == 'trend':
        # probabilities
        probs = 1.0 / (1.0 + np.exp(-y_pred_clean))
        preds_label = (probs >= 0.5).astype(np.float32)
        class_acc = float((preds_label == y_true_clean).mean())
        return avg_loss, float('nan'), float('nan'), float('nan'), class_acc, y_true_clean, preds_label
    else:
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = float(np.sqrt(mse))
        directional_accuracy = float((np.sign(y_true_clean) == np.sign(y_pred_clean)).mean())
        return avg_loss, mse, mae, rmse, directional_accuracy, y_true_clean, y_pred_clean


# -----------------------------------------------------
# 6. Main experiment pipeline
# -----------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Train/evaluate a GNN time-series model on IT sector data")
    parser.add_argument("--excel", type=str, default=r"C:\Users\Diya James\OneDrive\Documents\Grp\Sector.xlsx", help="Path to Sector.xlsx")
    parser.add_argument("--public-dir", type=str, default=None, help="Directory to save output figures and cleaned CSV")
    parser.add_argument("--window-size", type=int, default=20, help="Window size (days) for temporal encoder")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--temporal", type=str, choices=["lstm", "transformer"], default="lstm", help="Temporal encoder to use (lstm or transformer)")
    parser.add_argument("--gnn", type=str, choices=["gcn", "gat"], default="gcn", help="GNN conv type to use (gcn or gat)")
    parser.add_argument("--dynamic-graph", action="store_true", help="Enable dynamic graph updates during training")
    parser.add_argument("--graph-window", type=int, default=60, help="Number of recent days to use when recomputing correlation graph")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting (useful on headless servers)")
    parser.add_argument("--generate-sample", action="store_true", help="If excel not found, generate a small synthetic Sector.xlsx and run")
    parser.add_argument("--realtime", action="store_true", help="Run in realtime/deployment loop instead of training")
    parser.add_argument("--task", type=str, choices=["return", "price", "trend", "volatility"], default="return", help="Prediction task: return, price, trend (classification), or volatility")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping based on validation loss")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--lr-scheduler", action="store_true", help="Enable ReduceLROnPlateau learning rate scheduler")
    parser.add_argument("--lr-patience", type=int, default=5, help="LR scheduler patience (epochs)")
    args = parser.parse_args(argv)

    # Determine output directory (do not rebind module PUBLIC_DIR)
    out_dir = os.path.abspath(args.public_dir) if args.public_dir is not None else PUBLIC_DIR
    os.makedirs(out_dir, exist_ok=True)

    excel_path = args.excel
    if not os.path.exists(excel_path):
        if args.generate_sample:
            print(f"Excel file not found at {excel_path}; generating synthetic sample as requested...")
            try:
                written = _generate_sample_excel(excel_path)
                if written is not None:
                    excel_path = written
                print(f"Generated sample data at {excel_path}")
            except (OSError, RuntimeError) as e:
                raise RuntimeError(f"Failed to generate sample excel: {e}") from e
        else:
            raise FileNotFoundError(f"Could not find {excel_path} in the current directory. Use --generate-sample to create a test file.")

    print("Loading and preprocessing data...")
    # --- BEFORE REFINING: Raw Excel Feature Distributions ---
    # read raw input (support CSV fallback)
    if str(excel_path).lower().endswith('.csv'):
        raw_df = pd.read_csv(excel_path)
    else:
        try:
            raw_df = pd.read_excel(excel_path)
        except (ImportError, ValueError, OSError) as e:
            print(f"Warning: pd.read_excel failed ({e}), falling back to pd.read_csv")
            raw_df = pd.read_csv(excel_path)
    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    raw_features = ["Close", "Open", "High", "Low", "Volume"]
    plt.figure(figsize=(15, 5))
    for i, feat in enumerate(raw_features):
        if feat in raw_df.columns:
            plt.subplot(1, len(raw_features), i+1)
            plt.hist(pd.to_numeric(raw_df[feat], errors='coerce').dropna(), bins=30, alpha=0.7)
            plt.title(f"Raw {feat}")
    plt.suptitle("Raw Feature Distributions (Excel)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "feature_distributions_before_refining.png"), dpi=200)
    print(f"Saved raw feature distribution plot to {os.path.join(out_dir, 'feature_distributions_before_refining.png')}")

    # --- AFTER REFINING: Cleaned Feature Distributions ---
    features, returns, dates, companies, close_scaled_df, vol20_df, sectors_map, scaler_close, scaler_vol20 = load_it_sector_data(excel_path, save_dir=out_dir)
    # Build targets based on requested task
    task = args.task
    if task == 'return':
        targets = returns
        target_name = 'Returns'
    elif task == 'price':
        # predict scaled close price
        targets = close_scaled_df.values.astype(np.float32)
        target_name = 'Scaled Close Price'
    elif task == 'volatility':
        # predict VOL20 (rolling volatility)
        targets = vol20_df.values.astype(np.float32)
        target_name = 'VOL20'
    elif task == 'trend':
        # binary trend classification: 1 if return>0 else 0
        targets = (returns > 0).astype(np.float32)
        target_name = 'Trend(1=up)'
    else:
        targets = returns
        target_name = 'Returns'
    num_dates, num_companies, num_features = features.shape
    print(f"Num dates: {num_dates}, Num companies (nodes): {num_companies}, Num features: {num_features}")
    # show distributions for the first company's base features
    base_names = ["Normalized Close", "Normalized Volume", "Raw Return", "Normalized MA5", "Normalized VOL20"]
    plt.figure(figsize=(15, 5))
    for i in range(min(5, num_features)):
        plt.subplot(1, min(5, num_features), i+1)
        plt.hist(features[:, 0, i], bins=30, alpha=0.7)
        plt.title(base_names[i])
    plt.suptitle("Cleaned Feature Distributions (First Company)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "feature_distributions_after_refining.png"), dpi=200)
    print(f"Saved cleaned feature distribution plot to {os.path.join(out_dir, 'feature_distributions_after_refining.png')}")

    # --- BEFORE MODEL: Target Returns Distribution ---
    plt.figure(figsize=(6, 4))
    plt.hist(targets.flatten(), bins=50, alpha=0.7)
    plt.title("Target Returns Distribution (All Companies)")
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "target_returns_distribution_before_model.png"), dpi=200)
    print(f"Saved target returns distribution plot to {os.path.join(out_dir, 'target_returns_distribution_before_model.png')}")

    # training window settings
    window_size = args.window_size
    if num_dates <= window_size + 1:
        raise ValueError("Not enough dates to create time windows. Reduce window_size or use more data.")

    first_t = window_size
    last_t = num_dates - 1
    total_samples = last_t - first_t + 1

    train_samples = int(total_samples * 0.7)
    val_samples = int(total_samples * 0.15)
    test_samples = total_samples - train_samples - val_samples

    train_start_t = first_t
    train_end_t = train_start_t + train_samples
    val_start_t = train_end_t
    val_end_t = val_start_t + val_samples
    test_start_t = val_end_t
    test_end_t = last_t + 1

    print(f"Total usable samples: {total_samples}")
    print(f"Train: {train_samples}, Val: {val_samples}, Test: {test_samples}")

    # Use training period to compute correlations (avoid look-ahead)
    train_returns = targets[train_start_t:train_end_t]
    edge_index = build_correlation_graph(train_returns, threshold=0.2, companies=companies, sectors=sectors_map)
    print("Edge index shape:", edge_index.shape)

    # Create datasets
    train_dataset = TimeSeriesGraphDataset(
        features=features,
        targets=targets,
        edge_index=edge_index,
        window_size=window_size,
        start_t=train_start_t,
        end_t=train_end_t,
    )

    val_dataset = TimeSeriesGraphDataset(
        features=features,
        targets=targets,
        edge_index=edge_index,
        window_size=window_size,
        start_t=val_start_t,
        end_t=val_end_t,
    )

    test_dataset = TimeSeriesGraphDataset(
        features=features,
        targets=targets,
        edge_index=edge_index,
        window_size=window_size,
        start_t=test_start_t,
        end_t=test_end_t,
    )

    # Require PyTorch Geometric for DataLoader and graph ops
    if not HAS_PYG or PyGDataLoader is None:
        raise RuntimeError("torch_geometric is required to run training. Install it following instructions in requirements.txt and PyG docs.")
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = GNNTimeSeriesModel(
        window_size=window_size,
        num_features=num_features,
        temporal_model=args.temporal,
        gnn_type=args.gnn,
        hidden_lstm=64,
        hidden_gnn=64,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Optionally use a LR scheduler (ReduceLROnPlateau) to reduce LR on plateau of val loss
    scheduler = None
    if args.lr_scheduler:
        # Note: some torch versions do not accept `verbose` argument
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, factor=0.5)

    num_epochs = args.epochs
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_since_improve = 0
    # store epoch metrics
    metrics_rows = []

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        # Optionally update graph structure dynamically during training
        if args.dynamic_graph and (epoch % args.graph_update_every == 1 or epoch == 1):
            # gradually include more training data when rebuilding the graph to avoid look-ahead
            frac = float(epoch) / max(1.0, float(num_epochs))
            include_samples = max(1, int(train_samples * frac))
            end_idx = train_start_t + include_samples
            end_idx = min(end_idx, train_end_t)
            print(f"[Graph Update] Recomputing edges using training returns up to index {end_idx}")
            try:
                train_returns_partial = targets[train_start_t:end_idx]
                new_edge_index = build_correlation_graph(train_returns_partial, threshold=0.2, companies=companies, sectors=sectors_map)
                # update datasets' edge_index in place
                train_dataset.edge_index = new_edge_index
                val_dataset.edge_index = new_edge_index
                test_dataset.edge_index = new_edge_index
                # recreate loaders so they pick up any internal changes
                train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
                test_loader = PyGDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                print(f"[Graph Update] New edge_index shape: {new_edge_index.shape}")
            except ValueError as e:
                print(f"[Graph Update] Failed to update graph (ValueError): {e}")
            except RuntimeError as e:
                print(f"[Graph Update] Failed to update graph (RuntimeError): {e}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=epoch, output_dir=out_dir, task=task)
        val_loss, val_mse, val_mae, val_rmse, val_dir_acc, val_y_true, val_y_pred = evaluate(model, val_loader, device, epoch=epoch, stage='val', output_dir=out_dir, task=task)

        # Save per-epoch validation predictions to CSV (helps track predictions over epochs)
        try:
            if val_y_true.size > 0 and val_y_pred.size > 0:
                n_val_steps = val_end_t - val_start_t
                dates_for_val = [dates[t] for t in range(val_start_t, val_end_t)]
                date_expanded = np.repeat(dates_for_val, num_companies)
                company_expanded = np.tile(companies, n_val_steps)
                val_df = pd.DataFrame({
                    'Date': date_expanded,
                    'Company': company_expanded,
                    'ScaledTrue': val_y_true,
                    'ScaledPred': val_y_pred,
                })
                # If price task and scaler available, add raw values
                if task == 'price' and hasattr(scaler_close, 'mean_') and hasattr(scaler_close, 'scale_'):
                    scales = scaler_close.scale_
                    means = scaler_close.mean_
                    company_idx = np.tile(np.arange(num_companies), n_val_steps)
                    val_df['RawTrue'] = val_y_true * scales[company_idx] + means[company_idx]
                    val_df['RawPred'] = val_y_pred * scales[company_idx] + means[company_idx]
                per_epoch_val_csv = os.path.join(out_dir, f'val_predictions_epoch{epoch:03d}.csv')
                try:
                    val_df.to_csv(per_epoch_val_csv, index=False)
                    latest_val_csv = os.path.join(out_dir, 'val_predictions_latest.csv')
                    try:
                        shutil.copyfile(per_epoch_val_csv, latest_val_csv)
                    except (OSError, IOError):
                        val_df.to_csv(latest_val_csv, index=False)
                except (OSError, ValueError) as e:
                    print(f"Warning: failed to save per-epoch validation predictions: {e}")
        except (RuntimeError, ValueError, OSError) as e:
            # Non-fatal: continue training
            print(f"Warning: could not save per-epoch validation predictions: {e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # save a checkpoint of the best model so far
            try:
                torch.save(best_state_dict, os.path.join(out_dir, "best_model.pt"))
            except (OSError, RuntimeError) as e:
                print(f"Warning: failed to save best_model.pt: {e}")
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        # LR scheduler step (ReduceLROnPlateau expects the metric)
        if scheduler is not None:
            try:
                scheduler.step(val_loss)
            except (ValueError, RuntimeError, TypeError) as e:
                # scheduler.step may raise on invalid values; log and continue
                print(f"Warning: scheduler.step failed: {e}")

        # Early stopping check
        if args.early_stopping and epochs_since_improve >= args.patience:
            print(f"Early stopping triggered (no improvement for {epochs_since_improve} epochs). Restoring best model.")
            break

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}, Val MSE: {val_mse:.6f}, Val RMSE: {val_rmse:.6f}, Val MAE: {val_mae:.6f}, "
            f"Val DirAcc: {val_dir_acc:.4f}"
        )
        # Produce a simple mean Actual vs Pred bar plot every 20 epochs (and on final epoch)
        if (epoch % 10 == 0) or (epoch == num_epochs):
            if val_y_true.size > 0 and val_y_pred.size > 0:
                mean_actual = float(np.mean(val_y_true))
                mean_pred = float(np.mean(val_y_pred))
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.bar(["Actual", "Predicted"], [mean_actual, mean_pred], color=["steelblue", "orange"]) 
                ax.set_ylabel(target_name)
                ax.set_title(f"Mean Actual vs Predicted (Val) - Epoch {epoch}")
                plt.tight_layout()
                per_epoch_bar = os.path.join(out_dir, f"mean_actual_vs_pred_val_epoch{epoch:03d}.png")
                latest_bar = os.path.join(out_dir, "mean_actual_vs_pred_val_latest.png")
                fig.savefig(per_epoch_bar, dpi=200)
                try:
                    shutil.copyfile(per_epoch_bar, latest_bar)
                except (OSError, IOError) as e:
                    print(f"Warning: failed to copy mean bar image: {e}")
                    fig.savefig(latest_bar, dpi=200)
                plt.close(fig)

        # record metrics for this epoch
        metrics_rows.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'val_mse': float(val_mse),
            'val_rmse': float(val_rmse),
            'val_mae': float(val_mae),
            'val_dir_acc': float(val_dir_acc),
        })

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # persist training metrics to CSV
    try:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv_path = os.path.join(out_dir, 'training_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved training metrics to {metrics_csv_path}")
    except (OSError, ValueError) as e:
        print(f"Warning: failed to save training metrics CSV: {e}")

    print("Evaluating on test set...")
    test_loss, test_mse, test_mae, test_rmse, test_dir_acc, y_true, y_pred = evaluate(model, test_loader, device, output_dir=out_dir, task=task)
    print(
        f"Test Loss: {test_loss:.6f}, Test MSE: {test_mse:.6f}, Test RMSE: {test_rmse:.6f}, "
        f"Test MAE: {test_mae:.6f}, Test DirAcc: {test_dir_acc:.4f}"
    )

    # --- AFTER MODEL: Single Bar Plot Actual vs Predicted Returns (Test) ---
    if y_true.size > 0 and y_pred.size > 0:
        mean_actual = float(np.mean(y_true))
        mean_pred = float(np.mean(y_pred))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(["Actual", "Predicted"], [mean_actual, mean_pred], color=["steelblue", "orange"])
        ax.set_ylabel("Mean Return")
        ax.set_title("Mean Actual vs Predicted Return (Test)")
        plt.tight_layout()
        per_epoch_bar = os.path.join(out_dir, f"mean_actual_vs_pred_test_epoch{num_epochs:03d}.png")
        latest_bar = os.path.join(out_dir, "mean_actual_vs_pred_test_latest.png")
        fig.savefig(per_epoch_bar, dpi=200)
        try:
            shutil.copyfile(per_epoch_bar, latest_bar)
        except (OSError, IOError) as e:
            # fallback: write another file
            try:
                fig.savefig(latest_bar, dpi=200)
            except (OSError, IOError) as e2:
                print(f"Warning: failed to write latest bar image: {e2}")
        if not args.no_plot:
            plt.show()
    # Update all other plt.savefig calls to use PUBLIC_DIR
    # Example for other images:
    # plt.savefig(os.path.join(PUBLIC_DIR, "realtime_train_pred_vs_actual.png"), dpi=200)
    # plt.savefig(os.path.join(PUBLIC_DIR, "feature_distributions_before_refining.png"), dpi=200)
    # plt.savefig(os.path.join(PUBLIC_DIR, "feature_distributions_after_refining.png"), dpi=200)
    # plt.savefig(os.path.join(PUBLIC_DIR, "target_returns_distribution_before_model.png"), dpi=200)
    else:
        print("No predictions available to plot (empty test outputs).")
    # If task is price, inverse-transform scaled predictions back to raw price units and save CSV with metrics
    if task == 'price' and y_true.size > 0 and y_pred.size > 0:
        try:
            # test time indices
            test_times = list(range(test_start_t, test_end_t))
            n_test_steps = len(test_times)
            company_idx = np.tile(np.arange(num_companies), n_test_steps)
            # scaler_close stores per-company mean/scale (fitted on close.values with shape [num_dates, num_companies])
            if hasattr(scaler_close, 'mean_') and hasattr(scaler_close, 'scale_'):
                means = scaler_close.mean_
                scales = scaler_close.scale_
                raw_true = y_true * scales[company_idx] + means[company_idx]
                raw_pred = y_pred * scales[company_idx] + means[company_idx]
                raw_mse = mean_squared_error(raw_true, raw_pred)
                raw_mae = mean_absolute_error(raw_true, raw_pred)
                raw_rmse = float(np.sqrt(raw_mse))
                print(f"Test (raw price units) MSE: {raw_mse:.6f}, RMSE: {raw_rmse:.6f}, MAE: {raw_mae:.6f}")
                # Build DataFrame mapping dates and companies
                dates_for_test = [dates[t] for t in test_times]
                date_expanded = np.repeat(dates_for_test, num_companies)
                company_expanded = np.tile(companies, n_test_steps)
                preds_df = pd.DataFrame({
                    'Date': date_expanded,
                    'Company': company_expanded,
                    'ScaledTrue': y_true,
                    'ScaledPred': y_pred,
                    'RawTrue': raw_true,
                    'RawPred': raw_pred,
                })
                preds_csv = os.path.join(out_dir, f'predictions_price_raw_epoch{num_epochs:03d}.csv')
                try:
                    preds_df.to_csv(preds_csv, index=False)
                except (OSError, ValueError) as e:
                    print(f"Warning: failed to save predictions CSV: {e}")
                latest_csv = os.path.join(out_dir, 'predictions_price_raw_latest.csv')
                try:
                    shutil.copyfile(preds_csv, latest_csv)
                except (OSError, IOError):
                    try:
                        preds_df.to_csv(latest_csv, index=False)
                    except (OSError, ValueError) as e2:
                        print(f"Warning: failed to write latest predictions CSV: {e2}")
                print(f"Saved inverse-scaled price predictions to {preds_csv} and latest copy to {latest_csv}")
                # append raw metrics to training metrics CSV if present
                try:
                    metrics_path = os.path.join(out_dir, 'training_metrics.csv')
                    if os.path.exists(metrics_path):
                        metrics_df = pd.read_csv(metrics_path)
                        new_row_idx = metrics_df.index.max() + 1 if len(metrics_df) > 0 else 0
                        metrics_df.loc[new_row_idx, 'test_raw_mse'] = raw_mse
                        metrics_df.to_csv(metrics_path, index=False)
                except (OSError, pd.errors.EmptyDataError, ValueError) as e:
                    print(f"Warning: failed to append raw metrics to training_metrics.csv: {e}")
        except (AttributeError, ValueError, OSError) as e:
            print(f"Failed to inverse-transform price predictions: {e}")
    # If realtime/deployment flag set, generate a single live prediction CSV for the most recent date
    if args.realtime:
        try:
            latest_t = num_dates - 1
            if latest_t >= window_size:
                # prepare input window for the last available date
                window_feats = features[latest_t - window_size + 1 : latest_t + 1]  # [W, N, F]
                x_seq = window_feats.transpose(1, 0, 2).astype(np.float32)
                x_t = torch.from_numpy(x_seq)
                model.to(device)
                model.eval()
                with torch.no_grad():
                    out = model(x_t.to(device), edge_index.to(device))
                preds = out.cpu().numpy()
                if task == 'price' and hasattr(scaler_close, 'mean_') and hasattr(scaler_close, 'scale_'):
                    # inverse transform per-company
                    scales = scaler_close.scale_
                    means = scaler_close.mean_
                    company_idx = np.arange(len(companies))
                    raw_preds = preds * scales[company_idx] + means[company_idx]
                    preds_df = pd.DataFrame({'Company': companies, 'PredictedPriceScaled': preds, 'PredictedPrice': raw_preds})
                else:
                    preds_df = pd.DataFrame({'Company': companies, 'PredictedReturn': preds})
                live_path = os.path.join(out_dir, 'live_predictions.csv')
                try:
                    preds_df.to_csv(live_path, index=False)
                    print(f"Saved live predictions to {live_path}")
                except (OSError, ValueError) as e:
                    print(f"Warning: failed to save live_predictions.csv: {e}")
        except (RuntimeError, ValueError, OSError) as e:
            print(f"Realtime prediction error: {e}")


def run_realtime(interval_sec=10):
    """Run the GNN pipeline in a loop, saving new graph images at intervals."""
    import time
    while True:
        print("[Realtime] Running GNN pipeline...")
        try:
            main()
        except Exception as e:
            print(f"[Realtime] Error: {e}")
        print(f"[Realtime] Sleeping for {interval_sec} seconds...")
        time.sleep(interval_sec)

if __name__ == "__main__":
    # For normal run, just call main()
    main()
