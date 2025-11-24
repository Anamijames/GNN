## Project: GNN Time-Series (IT Sector)

This repository contains an end-to-end experiment pipeline (`main.py`) that trains and evaluates a hybrid temporal + graph neural network model on per-company time-series data (example: IT sector stock prices).

This README describes how to run the code, what each file does, what outputs are produced, and a detailed explanation of the main components of the code.

## Repository structure (important files)

- `main.py` — Primary entrypoint. Loads data, preprocesses, builds correlation graphs, constructs datasets, defines model (LSTM/Transformer + GCN/GAT), trains, evaluates, and saves artifacts.
- `requirements.txt` — Base Python packages used during development.
- `Sector.xlsx`, `Sector_test.csv`, `Sector_test.xlsx` — Example / real data files (your dataset). The code accepts CSV or Excel.
- `public_*` directories — Output directories created by `main.py` when runs are executed (plots, cleaned CSV, metrics, model checkpoints, predictions).
- `venv/` — Project virtual environment (not committed normally, but present in this workspace for convenience).

## Quick setup & prerequisites

1. Use the virtual environment provided (recommended) or create/activate your own:

PowerShell (if venv exists):
```powershell
& "C:/Users/Diya James/OneDrive/Documents/Grp/venv/Scripts/Activate.ps1"
```

Or create+activate a venv:
```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
```

2. Install core dependencies:
```powershell
python -m pip install -r requirements.txt
```

3. PyTorch Geometric (PyG) is required for training and graph layers. Install the PyG wheels that match your installed `torch` version and CUDA. Example (CPU, torch 2.9.1):
```powershell
python -m pip install --no-cache-dir -U "torch-scatter" -f https://data.pyg.org/whl/torch-2.9.1+cpu.html
python -m pip install --no-cache-dir -U "torch-sparse" -f https://data.pyg.org/whl/torch-2.9.1+cpu.html
python -m pip install --no-cache-dir -U "torch-cluster" -f https://data.pyg.org/whl/torch-2.9.1+cpu.html
python -m pip install --no-cache-dir -U "torch-spline-conv" -f https://data.pyg.org/whl/torch-2.9.1+cpu.html
python -m pip install --no-cache-dir -U "torch-geometric" -f https://data.pyg.org/whl/torch-2.9.1+cpu.html
```

If you want, I can auto-detect your `torch.__version__` and provide the exact wheel index commands.

## How to run

1) Show CLI help (see all options):
```powershell
python .\main.py --help
```

2) Quick sample run (generate small synthetic dataset and run 1 epoch):
```powershell
python .\main.py --generate-sample --excel .\Sector_test.xlsx --public-dir .\public_quick --epochs 1 --batch-size 2 --no-plot
```

3) Full example — price prediction (with inverse-scaling of predictions to raw price):
```powershell
python .\main.py --excel .\Sector.xlsx --public-dir .\public_price --task price --temporal lstm --gnn gcn --epochs 10 --batch-size 16
```

4) Trend classification (binary up/down):
```powershell
python .\main.py --excel .\Sector.xlsx --public-dir .\public_trend --task trend --temporal transformer --gnn gat --epochs 20
```

5) Enable dynamic graph updates, LR scheduler and early stopping:
```powershell
python .\main.py --excel .\Sector.xlsx --public-dir .\public_dyn --dynamic-graph --graph-update-every 5 --lr-scheduler --lr-patience 5 --early-stopping --patience 10 --epochs 100
```

## What `main.py` does (detailed walkthrough)

1) Argument parsing and setup
- `--excel` path to your dataset (Excel or CSV). The script falls back to CSV when Excel reader/writer is not available.
- `--public-dir` output directory for artifacts.
- `--task` selects target: `return` (default), `price` (scaled close), `volatility` (VOL20), or `trend` (binary classification).
- `--temporal` choose `lstm` or `transformer` for per-node temporal encoding.
- `--gnn` choose `gcn` or `gat` for spatial message passing.
- `--dynamic-graph` recomputes correlation graph periodically using recent training returns.
- `--early-stopping`, `--patience`, `--lr-scheduler`, `--lr-patience` control training callbacks.

2) Data loading and preprocessing (`load_it_sector_data`)
- Loads Date/Company/Open/High/Low/Close/Volume columns (tolerant to common name variants).
- Pivots to create per-date × per-company tables of Close and Volume.
- Fills missing data (ffill/bfill), clips extreme percentiles (1st/99th) per company.
- Computes derived features: returns (pct_change), MA5 (5-day moving average of close), VOL20 (rolling std of returns, window=20).
- Fits per-company StandardScaler on Close, Volume, MA5, VOL20 and stores them (used for inverse-scaling price predictions).
- Returns `features` array shaped [num_dates, num_companies, num_features] and `targets` (depends on `--task`). Also returns `dates`, `companies`, `close_scaled_df`, `vol20_df`, `sectors_map`, and scalers.

3) Graph construction (`build_correlation_graph`)
- Computes correlation matrix across companies using returns in the training period (avoid look-ahead).
- Adds edges where |corr| >= threshold (default 0.2).
- If sector mapping is available, optionally connects companies of the same sector even if correlation below threshold.

4) Dataset (`TimeSeriesGraphDataset`)
- Converts sliding windows of length `--window-size` into PyG `Data` objects.
- Each sample is a snapshot: x is [num_nodes, window, num_features], edge_index is the graph built from training returns, and y is target vector for that date across nodes.

5) Model (`GNNTimeSeriesModel`)
- Temporal encoder (per-node) options:
	- LSTM: processes sequences of shape [seq_len, batch, input_size] where batch is nodes across the batch.
	- Transformer: input-projection + TransformerEncoder; sequence dimension pooled by mean.
- Spatial GNN layers:
	- GCNConv (default) or GATConv (if PyG has it).
- After GNN layers, a linear head returns one scalar per node (regression) or a logit (classification for `trend`).

6) Training loop
- Uses MSE for regression tasks and BCEWithLogitsLoss for trend classification.
- Per-epoch: trains, evaluates on validation set, saves best model state (checkpoint `best_model.pt`), saves per-epoch scatter/hist PNGs and copies a latest PNG.
- Optional ReduceLROnPlateau scheduler steps on validation loss when `--lr-scheduler` is set.
- Optional early stopping stops training if validation loss hasn't improved for `--patience` epochs and restores the best parameters.

7) Evaluation and postprocessing
- After training, the code runs evaluation on the test set and computes: MSE, MAE, RMSE, and directional accuracy (sign accuracy).
- If `--task price`, predictions are inverse-transformed back to raw price units using the per-company `scaler_close` fitted during preprocessing. The script saves:
	- `predictions_price_raw_epoch{epoch:03d}.csv` and `predictions_price_raw_latest.csv` with columns: Date, Company, ScaledTrue, ScaledPred, RawTrue, RawPred.
- If `--realtime` is used, the script produces `live_predictions.csv` for the most recent date (and includes raw price if `--task price`).

## Output artifacts and explanation of graphs

- `cleaned_it_sector_data.csv` — pivoted multi-index CSV: columns like `Close_COMP1, Volume_COMP1, MA5_COMP1, VOL20_COMP1, Returns_COMP1, ...` across all companies and dates. Use this to inspect preprocessing.

- Feature distribution PNGs (`feature_distributions_before_refining.png`, `feature_distributions_after_refining.png`) — histograms that help inspect skew/outliers before/after clipping and scaling.

- `target_returns_distribution_before_model.png` — distribution of the target variable (useful to detect heavy tails or class imbalance for `trend`).

- Per-epoch scatter plots (`realtime_train_epochNNN.png`, `realtime_eval_val_epochNNN.png`):
	- For regression: scatter of actual vs predicted values across nodes/dates for that epoch.
	- For `trend`: histograms of predicted probabilities vs true labels.

- Mean Actual vs Predicted bar plots (`mean_actual_vs_pred_val_epochNNN.png`, `..._test_epochNNN.png`): show the mean of actual and predicted targets across the test/val sets. Useful sanity check to detect systematic bias (over/under prediction).

- `training_metrics.csv`: per-epoch metrics with columns: epoch, train_loss, val_loss, val_mse, val_rmse, val_mae, val_dir_acc. If `--task price` raw metrics are appended when available.

- `best_model.pt` — pickled best model state (state_dict). Loadable with `model.load_state_dict(torch.load('best_model.pt'))` after creating the same model architecture.

- `predictions_price_raw_epochNNN.csv` — CSV mapping each test sample (date × company) to scaled & raw predictions and ground truth. Use this for downstream evaluation or dashboarding.

## Example: Interpreting Model Graphs and Metrics

- Scatter Actual vs Predicted: points along the dashed diagonal mean good fit. Systematic deviation indicates bias; wide spread indicates high variance.
- Mean Actual vs Predicted bar: if Predicted > Actual consistently, model may be overestimating. Use this to spot global bias across companies.
- Training/Validation loss curves (from `training_metrics.csv`) reveal overfitting (val loss rising while train loss decreases) or underfitting (both high).

## Extending and debugging

- To change the temporal encoder size or hidden dims, edit the `GNNTimeSeriesModel` initialization in `main.py`.
- To change graph thresholding, modify the call to `build_correlation_graph(..., threshold=...)` in the main pipeline.
- If you see NaNs: inspect `cleaned_it_sector_data.csv` and ensure there are no all-zero or extremely sparse columns for any company.

## Reproducibility and tips

- Set a random seed at the top (already present in `main.py`), but for full determinism you may need to control additional torch/cuda flags depending on your environment.
- For GPU acceleration, ensure PyTorch with CUDA is installed and then install matching PyG CUDA wheels (replace `+cpu` in the wheel index with the appropriate `+cuXXX` tag).

## Want me to do more?

I can:
- Create a small PowerShell helper `scripts/install_pyg.ps1` that detects `torch.__version__` and prints/executesthe exact pip commands to install matching PyG wheels.
- Tidy up lint warnings and narrow exception catches in `main.py`.
- Add unit tests or a small Jupyter notebook to visualize per-company predictions over time.

If you'd like one of those, tell me which and I'll add it next.

GNN Time-Series (IT Sector) Experiment Pipeline

Overview

This repository contains a single entrypoint `main.py` that trains and evaluates a hybrid temporal + graph neural network model on per-company time series data (e.g., IT sector stock prices).

Key features
- Per-node temporal encoder (LSTM or Transformer) followed by a graph convolution (GCN or GAT).
- Correlation-based graph construction (optionally include same-sector edges).
- Supports multiple tasks: predict returns, scaled close price, rolling volatility (VOL20), or binary trend classification.
- Dynamic graph recomputation during training.
- Per-epoch artifacts (scatter plots), periodic mean Actual vs Predicted bar plots, training metrics CSV, and best-model checkpointing.
- Optional early stopping and ReduceLROnPlateau scheduler.
- Inverse-scaling for `--task price`: predictions saved back in raw price units.

Quick start (PowerShell)

# Show help
python .\main.py --help

# Generate a small sample Excel and run a quick 1-epoch regression (returns)
python .\main.py --generate-sample --excel .\Sector_sample.xlsx --public-dir .\public_test --epochs 1 --batch-size 4

# Run price prediction with LSTM + GCN and save inverse-scaled predictions (raw price)
python .\main.py --excel .\Sector.xlsx --public-dir .\public_price --task price --temporal lstm --gnn gcn --epochs 10

# Run trend classification with Transformer + GAT (if PyG + GAT available)
python .\main.py --excel .\Sector.xlsx --public-dir .\public_trend --task trend --temporal transformer --gnn gat --epochs 20

# Enable dynamic graph updates and ReduceLROnPlateau scheduler
python .\main.py --excel .\Sector.xlsx --public-dir .\public_dyn --dynamic-graph --graph-update-every 5 --lr-scheduler --epochs 50

Important output files (in --public-dir)
- cleaned_it_sector_data.csv: cleaned pivoted data saved after preprocessing.
- feature_distributions_before_refining.png / feature_distributions_after_refining.png: histograms of each feature.
- target_returns_distribution_before_model.png: target distribution histogram.
- realtime_train_epochNNN.png, realtime_eval_val_epochNNN.png: per-epoch scatter/hist plots.
- mean_actual_vs_pred_val_epochNNN.png / mean_actual_vs_pred_test_epochNNN.png: periodic bar plots comparing mean actual vs predicted.
- training_metrics.csv: CSV log containing per-epoch train/val losses and metrics.
- best_model.pt: checkpoint of the best validation model (state_dict saved as a pickle).
- predictions_price_raw_epochNNN.csv, predictions_price_raw_latest.csv: (when --task price) CSVs containing scaled and inverse-transformed raw price predictions for the test set.
- live_predictions.csv: single-shot predictions for the most recent date when using --realtime.

Notes and tips
- The script expects an Excel file with columns: Date, Company, Open, High, Low, Close, Volume. It tolerates common variants (close_price, trade_date, symbol, etc.).
- If PyTorch Geometric (PyG) is not installed, the script will print a warning and the help/generation features will still work, but training will fail unless PyG is installed.
- StandardScaler is fitted per-company for Close (used for the `--task price` inverse-scaling). The inverse transformation is applied per-company using the scaler mean/scale attributes.
- Early stopping can be enabled with `--early-stopping --patience N`. LR scheduling (ReduceLROnPlateau) can be enabled with `--lr-scheduler --lr-patience M`.

License / Acknowledgements
- This code is provided as-is for experimentation and educational purposes.

If you want, I can:
- Add an example PowerShell script that runs a full experiment and archives the public directory.
- Add unit tests or small validation notebooks to visualize predictions per-company.

 