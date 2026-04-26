# Forecasting Bitcoin Volatility with Regime-Aware Models

This repository contains the Group 13 project for DSE4211. The project studies whether modelling Bitcoin volatility regimes improves one-day-ahead realized variance forecasts compared with regime-unaware econometric and machine learning baselines.

The core workflow is:

1. Collect and clean Bitcoin, blockchain, macro, sentiment, and Google Trends data.
2. Construct daily realized variance from higher-frequency BTC returns.
3. Label low- and high-volatility regimes.
4. Train baseline forecasting models.
5. Train regime-aware models that feed predicted regime probabilities into an SVR volatility forecaster.
6. Compare the forecasts against benchmarks.

## Repository Layout

```text
DSE4211-GRP13/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- *.ipynb                         # Data collection, cleaning, merging, and regime labelling
|   |-- df_clean.csv                    # Cleaned merged dataset
|   |-- df_with_2regimes.csv            # Main modelling dataset with two-regime labels
|   |-- daily_realized.csv              # Daily realized variance data
|   |-- btc_usd_daily.csv               # BTC-USD daily market data
|   |-- blockchain_data.csv             # Blockchain covariates
|   |-- macro_indicators_data_2016_2026.csv  # Macro-economic covariates
|   |-- google_trends_bitcoin_daily.csv      # Investor attention covariate
|   `-- fear_greed_daily.csv                 # Investor sentiment covariate
|-- benchmark garch/
|   |-- garch.ipynb                     # GARCH benchmark 
|   |-- baseline_egarch.ipynb           # EGARCH benchmark
|   |-- garch.csv                       # GARCH test forecasts
|-- SVR/
|   |-- svr-l_update.ipynb              # Regime-unaware linear SVR benchmark
|   `-- results/svr_test_results.csv
|-- 2_step_hmm_model/
|   |-- hmm_svrl_revised.ipynb          # HMM regime classifier + SVR forecaster
|   |-- df_hmm.csv                          
|   `-- results/
|-- lstm-regimeaware/
|   |-- lstm_regime4.ipynb              # LSTM regime classifier + GARCH forecaster (this was subsequently abandoned)
|   |-- lstm_regime6new.ipynb           # LSTM regime classifier + SVR forecaster
|   |-- model_final/                    # LSTM SHAP and feature importance outputs
|   `-- results/
|-- rf-regimeaware/
|   |-- rf-svr(withoutcov)_update.ipynb # Random Forest regime classifier + SVR forecaster
|   |-- *.png                           # RF feature importance / SHAP plots
|   `-- results/
|-- error_analysis/
|   |-- error_analysis.ipynb            # Neutral/oracle benchmark and error propagation analysis
|   |-- neutral_results.csv
|   `-- oracle_results.csv
|-- test_analysis.ipynb                 # Comparative post-analysis of all models (DM test, plots)
```

## Main Components

### Data Preparation

Start in `data/` if you want to understand or regenerate the dataset.

Recommended reading order:

1. `data/data.ipynb`
2. `data/realized_data.ipynb`
3. `data/blockchain_data.ipynb`
4. `data/yfinance_data.ipynb`
5. `data/gtrends_data.ipynb`
6. `data/fearandgreed_data.ipynb`
7. `data/data_merging.ipynb`
8. `data/regime_classification_labelling.ipynb`
9. `data/covariate_test.ipynb`

The key modelling file is `data/df_with_2regimes.csv`.

### Baseline Models

- `benchmark garch/garch.ipynb`: GARCH benchmark.
- `benchmark garch/baseline_egarch.ipynb`: EGARCH benchmark.
- `SVR/svr-l_update.ipynb`: single-regime linear SVR benchmark.

### Regime-Aware Models

- `2_step_hmm_model/hmm_svrl_revised.ipynb`: two-step HMM + SVR model.
- `lstm-regimeaware/lstm_regime6new.ipynb`: LSTM-based regime probability model + SVR forecasting.
- `rf-regimeaware/rf-svr(withoutcov)_update.ipynb`: Random Forest regime classifier with SVR forecasting.

Each regime-aware model produces test-set forecasts and regime probabilities under its `results/` folder.

### Error Analysis

Use `error_analysis/error_analysis.ipynb` to compare the regime-aware models against:

- a neutral benchmark where regime probabilities are fixed at 0.5;
- an oracle benchmark where the true next-day regime is supplied.

This notebook is useful for diagnosing whether performance gains or losses come from the volatility forecaster itself or from noisy regime classification.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Therulow/DSE4211-GRP13.git
cd DSE4211-GRP13
```

If you already have the repo locally, just move into the project directory:

```bash
cd /path/to/DSE4211-GRP13
```

### 2. Create a virtual environment

Using Python's built-in `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project uses common data science packages such as `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tensorflow`, and `shap`.

### 4. Install Jupyter if needed

If Jupyter is not already available in your environment:

```bash
pip install notebook
```

Then launch:

```bash
jupyter notebook
```

## How to Navigate the Project

For a quick project overview, start with:

1. `data/df_with_2regimes.csv` to inspect the final modelling dataset.
2. `SVR/svr-l_update.ipynb` to understand the regime-unaware ML baseline.
3. `rf-regimeaware/rf-svr(withoutcov)_update.ipynb` to see the best-performing regime-aware workflow.
4. `2_step_hmm_model/hmm_svrl_revised.ipynb` and `lstm-regimeaware/lstm_regime6new.ipynb` for alternative regime classifiers.
5. `error_analysis/error_analysis.ipynb` for neutral/oracle comparison and final diagnostic analysis.

For reproducing the full pipeline, use this order:

1. Run the notebooks in `data/` to rebuild the merged dataset and regime labels.
2. Run the baseline notebooks in `benchmark garch/` and `SVR/`.
3. Run the regime-aware notebooks in `2_step_hmm_model/`, `lstm-regimeaware/`, and `rf-regimeaware/`. 
4. Run `test_analysis.ipynb` to consolidate benchmark comparisons.
5. Run `error_analysis/error_analysis.ipynb` to examine quality of regime signals. 

Most notebooks assume they are run from the repository root or from their own folder with relative paths unchanged. If a file path error appears, restart Jupyter from the repository root.

## Key Outputs

Important generated outputs include:

- `benchmark garch/garch.csv`
- `SVR/results/svr_test_results.csv`
- `2_step_hmm_model/results/hmm_svrl_revised_test_results.csv`
- `2_step_hmm_model/results/hmm_svrl_revised_summary.csv`
- `lstm-regimeaware/results/lstm-svr3.csv`
- `rf-regimeaware/results/rf-svr(withoutcov)_results.csv`
- `error_analysis/neutral_results.csv`
- `error_analysis/oracle_results.csv`

The root-level PNG files and selected PNG files inside model folders are reporting plots, including predicted-versus-actual variance, regime probability diagnostics, correlation plots, and feature importance summaries.

## Notes

- The repository is notebook-first; there are no central command-line training scripts.
- Some paths contain spaces or parentheses, so quote them in shell commands, for example `cd "benchmark garch"`.
- Re-running data collection notebooks may depend on external data sources and network availability.
- Existing CSV result files are included so the analysis can be inspected without fully re-running every model.

## Team

Group 13, DSE4211, National University of Singapore

- Low Jun Chen Jason
- Abigail Tan Huey Ern
- Liang Ying Ying
- Tang Wei Hao
- Royston How
- Elsie Woo
