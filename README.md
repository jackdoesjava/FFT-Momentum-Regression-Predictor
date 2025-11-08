# Hybrid Stock Prediction Engine

A modular, production-style stock forecasting engine that combines multiple quantitative
methods—including polynomial regression, momentum modelling, FFT-based residual forecasting,
and optional LSTM neural networks—into a unified and reproducible pipeline.

The project is implemented entirely in `stockregressor.py` and includes clean data handling,
local caching, model evaluation utilities, and publication-quality plotting tools.

---

## 🚀 Features

- **Polynomial & Linear Regression** for long-term trend modelling  
- **Hybrid Momentum Model** to incorporate recent short-term movements  
- **FFT-based Cyclical Forecasting** using dominant Fourier harmonics  
- **Optional LSTM Model** (if TensorFlow/Keras installed)  
- **Automatic Train/Test Splitting**  
- **Model Evaluation** (R² and MSE on train/test)  
- **High-quality Visualization** of actual vs predicted prices  
- **Local Data Loading Only** (no live downloads)

---

## 📁 Repository Structure

/project_root
│── stockregressor.py
│── requirements.txt
└── README.md

## 📥 Installing Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
## 📊 Providing Your Own Stock Data (IMPORTANT)

This project **does NOT download data from the internet**. Instead, it reads from **local CSV files** placed inside a folder named:

Stocks/

Each CSV **must** follow this naming format:

Stock-<TICKER>-YYYY-MM-DD-YYYY-MM-DD.csv

Example:

Stock-GOOG-2015-01-01-2017-08-27.csv

Each file must contain at least:

Date, Open, High, Low, Close, Adj Close, Volume

✅ **Make sure your file’s date range fully covers:**
- Your chosen `start_date`
- Your chosen `end_date`
- Your prediction horizon (`predict_days`)

If the date range doesn't cover these, the script will raise a **“no local file found”** error.

## 🧠 Using the Regressor

```python
from stockregressor import ProductionStockRegressor

reg = ProductionStockRegressor(
    ticker='GOOG',
    start_date='2015-01-01',
    end_date='2017-01-01',
    predict_days=60,
    verbose=True
)

reg.train_regression_model(degrees=[1, 3, 5])
reg.train_momentum_model(lookback_days=30, momentum_weight=0.4)
reg.train_fft_model(num_harmonics=6)

reg.report_scores()
reg.plot_predictions(model_names='all')
```

The prediction horizon (predict_days)

Otherwise the script will raise a “no local file found” error.
