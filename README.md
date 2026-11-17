# TC-SNP
# TC-SNP: A Temporal and Cross-Sectional Modeling Framework for Stock Market Prediction

This repository contains the implementation of **TC-SNP** for one-trading-day-ahead stock return prediction on four stock universes: CSI300, CSI500, S&P 500, and NASDAQ-100.

The implementation uses a 60-trading-day historical window to predict each stock's next-trading-day close-to-close return. It combines temporal representation learning, graph-based cross-sectional modeling, and attention-based feature fusion.

> **Data note:** The CSV files are not included in this repository. Users must obtain or prepare the stock-level data separately and place it in `data/raw/`. The index links below are reference pages; the model requires daily records for individual constituent stocks, not a single index-level time series.

## 1. Repository Structure

```text
TC-SNP/
├── README.md
├── scripts/
│   ├── csi300_tcsnp_nextday.py
│   ├── csi500_tcsnp_nextday.py
│   ├── sp500_tcsnp_nextday.py
│   └── nasdaq100_tcsnp_nextday.py
├── data/
│   ├── raw/
│   │   ├── hs300_2018_2023_new_1.csv
│   │   ├── csi500_2018_2023_new_1.csv
│   │   ├── sp500_2018_2023_new_1.csv
│   │   └── nd100_2018_2023_new_1.csv
│   └── README.md
├── results/
│   ├── csi300_tcsnp_nextday_results/
│   ├── csi500_tcsnp_nextday_results/
│   ├── sp500_tcsnp_nextday_results/
│   └── nasdaq100_tcsnp_nextday_results/
└── .gitignore
```

The `results/` directories are created automatically when the scripts run.

## 2. Datasets and Data Sources

The following pages identify the four index markets used in the study:

| Dataset | Reference page |
|---|---|
| CSI300 | (https://www.kaggle.com/datasets/liqiang2022/csi-300-index-fully-data-20172022) |
| CSI500 | (https://www.kaggle.com/datasets/kyrie35goat/csi500-stock-price-volume-data-2010-2023) |
| S&P 500 | (https://www.kaggle.com/datasets/camnugent/sandp500/data) |
| NASDAQ-100 | (https://www.kaggle.com/datasets/novandraanugrah/nasdaq-100-nas100-historical-price-data) |

### 2.1 Required Data Format

Each CSV must contain one row per stock per trading day. The code constructs a cross-sectional stock panel and a stock graph for each prediction date. Therefore, a CSV containing only the daily closing values of an index is not sufficient.

The required columns are:

| Column | Description |
|---|---|
| `dt` | Trading date |
| `kdcode` | Stock identifier or ticker |
| `open` | Daily opening price |
| `high` | Daily highest price |
| `low` | Daily lowest price |
| `close` | Daily closing price |
| `volume` | Daily trading volume |
| `turnover` | Daily turnover; required for CSI300 and CSI500 |

Feature configuration:

| Dataset | Input Features |
|---|---|
| CSI300 | open, close, high, low, volume, turnover |
| CSI500 | close, open, high, low, turnover, volume |
| S&P 500 | close, open, high, low, volume |
| NASDAQ-100 | close, open, high, low, volume |

All price and volume columns should be numeric. Dates should be valid trading dates, and stock identifiers should be consistent across rows.

### 2.2 Stock-Universe Handling

The current implementation retains stocks that have an observation on every trading date in the loaded file. It also removes duplicate `(dt, kdcode)` rows by keeping the last occurrence.

At least 10 complete stocks are required because the portfolio evaluation selects the Top 10 stocks.


### 2.3 Data Access and Redistribution

The raw CSV files are intentionally excluded from this repository. Before redistributing downloaded data, check the terms and licensing conditions of the original data provider.

## 3. Prediction Task and Data Split

For stock `i`, the model uses the preceding 60 trading days to predict the next trading day's close-to-close return:

`r[i,t+1] = close[i,t+1] / close[i,t] - 1`

The training, validation, and test periods are determined automatically from the calendar years in each CSV:

- **Training:** target dates earlier than the penultimate calendar year.
- **Validation:** target dates in the penultimate calendar year.
- **Testing:** target dates in the final calendar year.

For a dataset whose final year is 2023, the split is:

| Split | Target Years |
|---|---|
| Training | 2018–2021 |
| Validation | 2022 |
| Testing | 2023 |

The split is dynamic. For the NASDAQ-100 file described for this project (2016-11-15 to 2025-10-01), the validation year is 2024 and the test year is 2025 (through 2025-10-01); training samples use target years before 2024.

The first 60 trading days are used to form historical input windows and do not produce prediction samples.

## 4. Main Experimental Settings

The principal settings are defined near the top of each Python script.

| Setting | Value |
|---|---:|
| Historical input window | 60 trading days |
| Historical-state attention length | 10 |
| Correlation lookback | 250 trading days |
| Graph construction threshold | 0.8 |
| Temporal hidden dimensions | 32 and 10 |
| GAT attention heads | 4 |
| Number of latent states | 8 |
| Multi-head cross-attention heads | 4 |
| Prediction hidden dimension | 32 |
| Batch size | 32 |
| Learning rate | 0.0002 |
| Training epochs per run | 20 |
| Independent runs | 15 |
| Base random seed | 2026 |
| Portfolio selection | Top 10, equal-weighted |

The random seed is increased by one for each independent run. The checkpoint with the lowest validation MSE is selected within each run. Predictions from the 15 runs are averaged before the final test metrics and portfolio returns are calculated.

## 5. Environment and Installation

The scripts use Python, NumPy, pandas, and PyTorch. A PyTorch build compatible with your CPU or CUDA environment should be installed.

## 6. Run the Experiments

Run commands from the repository root. The `--data` argument specifies the input CSV, and `--output` specifies where results will be written.

```bash
python scripts/csi300_tcsnp_nextday.py \
  --data data/raw/hs300_2018_2023_new_1.csv \
  --output results/csi300_tcsnp_nextday_results

python scripts/csi500_tcsnp_nextday.py \
  --data data/raw/csi500_2018_2023_new_1.csv \
  --output results/csi500_tcsnp_nextday_results

python scripts/sp500_tcsnp_nextday.py \
  --data data/raw/sp500_2018_2023_new_1.csv \
  --output results/sp500_tcsnp_nextday_results

python scripts/nasdaq100_tcsnp_nextday.py \
  --data data/raw/nd100_2018_2023_new_1.csv \
  --output results/nasdaq100_tcsnp_nextday_results
```

If you keep the Python scripts in the repository root rather than in `scripts/`, remove the `scripts/` prefix from each command.


## 7. Output Files

Each script creates its output directory automatically and writes four CSV files.

| Output File | Contents |
|---|---|
| `<dataset>_test_nextday_predictions_mean.csv` | Test-date predictions averaged over 15 runs, including signal date, target date, stock code, actual return, and predicted return |
| `<dataset>_top10_equal_weight_daily_returns.csv` | Daily Top-10 stock selections and equal-weighted realized portfolio returns |
| `<dataset>_run_summary.csv` | Per-run seed, best epoch, best validation MSE, test MSE, and test MAE |
| `<dataset>_nextday_summary.csv` | Overall test MSE and MAE based on averaged predictions, plus portfolio performance metrics |

Here, `<dataset>` is `csi300`, `csi500`, `sp500`, or `nasdaq100`.

For example:

```text
results/
└── csi500_tcsnp_nextday_results/
    ├── csi500_test_nextday_predictions_mean.csv
    ├── csi500_top10_equal_weight_daily_returns.csv
    ├── csi500_run_summary.csv
    └── csi500_nextday_summary.csv
```

The portfolio summary reports:

- **AR:** annualized return.
- **AL:** annualized volatility.
- **MD:** maximum drawdown.
- **AS:** annualized return-to-volatility ratio, calculated as AR divided by AL.

## 8. Reproducibility and Implementation Notes

- The prediction horizon is one trading day.
- The input window contains the 60 trading days ending on the signal date.
- The target is the raw next-day close-to-close return, not a cross-sectional rank label.
- The model output is a real-valued return prediction and is not restricted to the interval [0,1].
- Graph edges are constructed when the absolute Pearson correlation of historical stock returns reaches the configured threshold; self-loops are also included.
- The validation set is used for checkpoint selection; the test set is reserved for final evaluation.
- The reported final prediction is the average of predictions from 15 independent runs.

For strict reproduction, use the same CSV files, constituent universe, date coverage, preprocessing, and parameter settings as those used in the reported experiments.

## 9. Citation

If you use this code in academic work, please cite the associated TC-SNP paper. Add the final bibliographic information here when the paper is published.

## 10. Contact

For questions about the implementation or reproducibility, please open a GitHub issue in this repository.
