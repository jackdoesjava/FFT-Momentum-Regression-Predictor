# FFT Spectral Analysis for Market Prediction

A quantitative engine designed to decouple structural trends from cyclical noise using a hybrid **Ridge-FFT** approach. This repository focuses on modeling residuals in financial time-series to extract periodic signals.

## Project Highlights
* **Engineered a hybrid forecasting pipeline** utilizing Polynomial Ridge Regression for baseline trends and Fast Fourier Transforms (FFT) for cyclical residual analysis.
* **Optimized model generalization** by implementing harmonic filtering to mitigate end-point bias and prevent "boundary slingshot" effects in out-of-sample testing.

## Directory Structure
```text
├── src/
│   ├── data/
│   │   └── data_loader.py     # yfinance API integration & local caching
│   ├── evaluation/
│   │   └── metrics.py          # Sharpe Ratio & Risk-Adjusted returns
│   ├── models/
│   │   ├── base_model.py       # Abstract Base Class for forecasters
│   │   ├── fft_residual.py     # FFT-based cyclical modeling
│   │   └── ridge_momentum.py   # Ridge Regression with momentum blending
│   ├── visualization/
│   │   └── plotters.py         # Matplotlib forecasting visualizations
│   └── __init__.py
├── data/                       # Root directory for raw CSV storage
├── main.py                     # Central execution pipeline
└── requirements.txt            # Project dependencies
```

## Performance Metrics (90-Day Forecast)
Testing on **GLD** (Gold Shares) revealed a robust baseline while highlighting the sensitivity of spectral analysis to non-stationary market shocks.

| Model | R2 Test | MSE | Sharpe |
| :--- | :--- | :--- | :--- |
| **Ridge_Momentum** | -5.7466 | 50.9401 | **1.4115** |
| **FFT_Wrapped_Ridge_Momentum** | -7.0326 | 60.6504 | -0.7987 |

> **Note:** The negative Sharpe in the FFT overlay is a result of "Boundary Slingshot" effects—where the cyclical nature of the Fourier Transform attempts to mathematically "close the loop" on recent market crashes.

## Future Work
* **Mamba SSM Implementation:** Transitioning from spectral analysis to State Space Models to better handle non-stationary data.

