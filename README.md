# 📈 Quant Trading System

A **research‑grade quantitative trading framework** for **multi‑asset stock prediction, portfolio construction, and backtesting** using **machine learning and deep learning models**.
This repository provides an **end‑to‑end pipeline** — from **raw market data ingestion** to **strategy evaluation with risk‑adjusted metrics**.

---

## 🚀 Key Highlights

* 🔄 **End‑to‑End Quant Pipeline**: Data → Feature Engineering → Prediction → Portfolio Construction → Backtesting
* 🌍 **Multi‑Market Support**:

  * US Dow Jones 30 (DOW30)
  * US NASDAQ‑100 (NASQ100)
  * China Shanghai Stock Exchange 50 (SSE50)
 
* 📊 **Rich Feature Engineering**:

  * OHLCV features
  * Technical indicators (MACD, RSI, Bollinger Bands, SMA, CCI, DX)
  * Temporal & covariance‑based features
* 🧠 **Model‑Agnostic Design** – plug‑and‑play with ML / DL / RL models
* 📉 **Professional Backtesting Engine** with:

  * Top‑K stock selection
  * Custom holding periods
  * Risk‑adjusted performance metrics

---

## 🏗️ Project Architecture

```
Quant_Trading_System/
│
├── config.py                  # Global configuration (dates, features, model params)
├── config_tickers.py          # Market‑wise ticker lists
├── download_stock_data.py     # Data ingestion entry point
├── YahooFinance.py            # Yahoo Finance data pipeline & preprocessing
├── stock_data_handle.py       # Dataset builder for ML/DL models
│
├── data_dir/                  # Auto‑generated datasets
│   ├── DOW30/
│   ├── NASQ100/
│   └── SSE50/
│
├── trained_models/            # Saved trained models
├── tensorboard_log/           # Training logs
├── results/                   # Backtesting & evaluation outputs
│
├── backtest.ipynb             # Strategy evaluation notebook
├── Code_Challenge.ipynb       # Experimental / exploratory analysis
└── README.md                  # Project documentation
```

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Quant_Trading_System.git
cd Quant_Trading_System
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Libraries Used**

* `pandas`, `numpy`, `scikit‑learn`
* `yfinance`
* `torch`
* `stockstats`
* `tqdm`

---

## 📥 Data Pipeline

### Supported Data Source

**Yahoo Finance API** (via `yfinance`)

### Download & Preprocess Market Data

```bash
python download_stock_data.py
```

This will:

* Download OHLCV data
* Clean missing/inconsistent tickers
* Add technical indicators
* Normalize & standardize features
* Split data into **Train / Validation / Test** sets

📁 Output directory:

```
data_dir/
 ├── DOW30/
 ├── NASQ100/
 └── SSE50/
```

---

## 🧠 Feature Engineering

### Technical Indicators

Configured in `config.py`:

* MACD
* Bollinger Upper / Lower Bands
* RSI (30)
* CCI (30)
* DX (30)
* SMA (30, 60)

### Temporal Features

* Open
* High
* Low
* Close
* Volume

### Advanced Features

* Rolling covariance matrix (252‑day lookback)
* Short‑term return labels (future price movement)

---

## 📊 Dataset Construction

The `Stock_Data` class builds **model‑ready tensors**:

```python
from stock_data_handle import Stock_Data

stock_data = Stock_Data(
    dataset_name="DOW",
    full_stock_path="DOW30",
    window_size=60,
    prediction_len=5
)
```

### Output Shapes

* **Input**: `(Days, Stocks, Features)`
* **Labels**: `(Prediction Horizon, Days, Stocks)`
* **Price Matrix**: Used for portfolio valuation

---

# 🧠 Architecture & Trading Strategy Design

This document provides a **deep technical explanation** of the internal architecture, data flow, and trading strategy logic used in the **Quant Trading System**.
It is intended for **researchers, reviewers, and advanced users** who want to understand *how and why* the system works.

---

## 1. System Architecture Overview

The system follows a **modular, pipeline-based architecture**, ensuring scalability, reproducibility, and research extensibility.

```
┌────────────┐
│ Yahoo API  │
└─────┬──────┘
      │
      ▼
┌────────────────────┐
│ Data Preprocessing │
│ (Cleaning, TI,     │
│  Scaling)          │
└─────┬──────────────┘
      │
      ▼
┌────────────────────┐
│ Feature Engineering│
│ - Technical        │
│ - Temporal         │
│ - Covariance       │
└─────┬──────────────┘
      │
      ▼
┌────────────────────┐
│ Prediction Models  │
│ (ML / DL / RL)     │
└─────┬──────────────┘
      │
      ▼
┌────────────────────┐
│ Portfolio Engine   │
│ (Ranking + Top-K)  │
└─────┬──────────────┘
      │
      ▼
┌────────────────────┐
│ Backtesting Engine │
│ (Risk Metrics)     │
└────────────────────┘
```

---

## 2. Data Flow Explanation

### Step 1: Market Data Ingestion

* Source: **Yahoo Finance**
* Frequency: **Daily bars**
* Assets:

  * DOW30
  * NASQ100
  * SSE50

Handled by:

* `YahooFinance.py`
* `download_stock_data.py`

---

### Step 2: Data Cleaning & Alignment

Key challenges addressed:

* Missing trading days
* Delisted / illiquid stocks
* Cross-asset date alignment

Solution:

* Keep only **intersection of tickers** available across all dates
* Enforce consistent `(date, ticker)` ordering

---

### Step 3: Feature Engineering

#### 3.1 Technical Indicators

Computed using `stockstats`:

* MACD
* Bollinger Bands (UB/LB)
* RSI (30)
* CCI (30)
* DX (30)
* SMA (30, 60)

These capture **momentum, volatility, and trend structure**.

---

#### 3.2 Temporal Market Features

| Feature | Description    |
| ------- | -------------- |
| Open    | Opening price  |
| High    | Daily high     |
| Low     | Daily low      |
| Close   | Adjusted close |
| Volume  | Trading volume |

---

#### 3.3 Covariance Matrix (Risk Structure)

A **rolling 252-day covariance matrix** is computed:

```
Cov_t = Cov(returns_{t-252 : t})
```

Purpose:

* Capture **inter-stock dependency**
* Enable **risk-aware learning**
* Useful for graph-based and attention models

---

## 4. Dataset Construction Logic

Implemented in `stock_data_handle.py`.

### Input Tensor

```
X ∈ ℝ[Days × Stocks × Features]
```

### Label Tensor

```
Y ∈ ℝ[Prediction_Horizon × Days × Stocks]
```

Label definition:

```
Return_{t+h} = (P_{t+h} - P_t) / P_t
```

---

## 5. Trading Strategy Design

### 5.1 Prediction → Ranking

For each trading day:

1. Model predicts **future returns** for all stocks
2. Stocks are **ranked descending** by predicted return

---

### 5.2 Portfolio Construction (Top-K)

| Parameter | Meaning                   |
| --------- | ------------------------- |
| K         | Number of selected stocks |
| H         | Holding period (days)     |

Strategy:

* Buy **Top-K ranked stocks**
* Equal-weighted allocation
* Hold for **H days**

---

### 5.3 Rebalancing

* Portfolio is rebalanced after every holding period
* Prevents excessive turnover
* Mimics realistic trading constraints

---

## 6. Backtesting Methodology

### Assumptions

* No leverage
* No short selling
* Equal capital allocation
* Daily re-evaluation

### Metrics Computed

| Metric            | Interpretation            |
| ----------------- | ------------------------- |
| Sharpe Ratio      | Risk-adjusted return      |
| Sortino Ratio     | Downside risk             |
| Max Drawdown      | Worst peak-to-trough loss |
| Annualized Return | Normalized performance    |
| Information Ratio | Alpha vs Market           |

---

## 7. Why This System Is Robust

✔ Market-agnostic design
✔ No look-ahead bias
✔ Strict temporal splitting
✔ Risk-aware features
✔ Modular extensibility

---

## 8. Extension Ideas

* Transaction cost modeling
* Volatility targeting
* Reinforcement learning agent
* Intraday data support
* Regime detection

---

## 9. Intended Use

This repository is suitable for:

* Quantitative research
* ML for finance projects
* Academic experimentation
* Strategy prototyping

---

## 📈 Backtesting Framework

Backtesting is designed to simulate **real‑world trading constraints**.

### Strategy Logic

1. Rank stocks based on model predictions
2. Select **Top‑K** stocks
3. Hold positions for **N days**
4. Rebalance periodically

### Metrics Computed

* Sharpe Ratio
* Sortino Ratio
* Maximum Drawdown
* Annualized Return
* Total Return
* Information Ratio (vs Market)

📓 Example:

```python
results = model.backtest(topk=5, holding_period=10)
```

---

## 🧪 Research Notebooks

* **`backtest.ipynb`** → Strategy evaluation & visualization
* **`Code_Challenge.ipynb`** → Experiments, ideas & prototyping

## Trading Strategy Logic
- Prediction → Ranking → Allocation → Rebalancing


---

## ⚠️ Disclaimer

> This project is **strictly for educational and research purposes**.
> It **does NOT constitute financial advice**.
> Trading in financial markets involves significant risk.

---

## 🤝 Contributions

Contributions are welcome! 🚀

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

## 📜 License

This project is licensed under the **Eclipse Public License - v 2.0**.

---

## 🙏 Acknowledgements

* Yahoo Finance
* Open‑source Quant & ML community
* Academic research in financial time‑series modeling

---

## OWNER😎 Created By:- "**KARTIK PANDEY**"
For any Query, E-Mail:- **kartik0pandey00@gmail.com**

Visit My Website:- **https://kartik-pandey.github.io/My_Portfolio_Website/**

---

## ⭐ If you find this project useful

Give it a **star ⭐** on GitHub — it really helps!









