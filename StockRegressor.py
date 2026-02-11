import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from typing import List, Tuple, Dict, Optional, Union
import warnings
import os
import glob

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge # Added Ridge
from sklearn.metrics import r2_score, mean_squared_error
from IPython.display import display

try:
    import plotext as plt_term
    TERMINAL_PLOTTING = True
except ImportError:
    TERMINAL_PLOTTING = False
    print("⚠️ Install 'plotext' for nice terminal plots: pip install plotext")

try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping # Added for Anti-Overfitting
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    warnings.warn("Keras/TensorFlow not installed. LSTM functionality will be unavailable.", ImportWarning)

# Configs and Constraints
np.random.seed(42)
sns.set(style="whitegrid")
CACHE_DIR = 'Stocks'
DEFAULT_TICKER = 'GOOG'
DEFAULT_START_DATE = '2015-01-01'
DEFAULT_END_DATE = '2017-01-01'
DEFAULT_PREDICT_DAYS = 50
DEFAULT_TRAIN_TEST_RATIO = 0.8
DEFAULT_REGRESSION_DEGREES = [1] 

# Helper Funcs

def _handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing data using forward and backward fill."""
    df_clean = df.copy()
    df_clean = df_clean.ffill() 
    df_clean = df_clean.bfill()
    return df_clean

def _calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates annualized Sharpe Ratio."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)

def _get_cache_path(ticker: str, start_date: str, end_date: str) -> str:
    """Generates a standardized cache file path."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f'{ticker}_{start_date}_{end_date}.csv')

def _calculate_fft_waveform(residuals: np.ndarray, num_harmonics: int, total_samples: int) -> np.ndarray:
    """Calculates the FFT-based waveform for a given residual series."""
    n = len(residuals)
    if n == 0: return np.zeros(total_samples)

    fft_coeffs = np.fft.fft(residuals)
    frequencies = np.fft.fftfreq(n)

    # Sort by amplitude to find dominant frequencies
    sorted_indices = np.argsort(np.abs(fft_coeffs[1:n//2]))[::-1] + 1
    top_indices = [0] + list(sorted_indices[:num_harmonics])
    
    t = np.arange(total_samples)
    fft_reconstructed = np.zeros(total_samples)
    
    for i in top_indices:
        amplitude = np.abs(fft_coeffs[i]) / n
        phase = np.angle(fft_coeffs[i])
        frequency = frequencies[i]
        
        fft_reconstructed += amplitude * np.cos(2 * np.pi * frequency * t + phase)
        
        if i != 0 and i != n//2:
            neg_i = n - i
            neg_amplitude = np.abs(fft_coeffs[neg_i]) / n
            neg_phase = np.angle(fft_coeffs[neg_i])
            neg_frequency = frequencies[neg_i]
            fft_reconstructed += neg_amplitude * np.cos(2 * np.pi * neg_frequency * t + neg_phase)

    return fft_reconstructed

class ProductionStockRegressor:
    """
    A professional, modular class for fetching, modeling, and predicting stock prices.
    Includes protections against overfitting (Ridge, EarlyStopping).
    """

    def __init__(self,
                 ticker: str = DEFAULT_TICKER,
                 start_date: str = DEFAULT_START_DATE,
                 end_date: str = DEFAULT_END_DATE,
                 predict_days: int = DEFAULT_PREDICT_DAYS,
                 verbose: bool = True):
        self.ticker = ticker
        self.start_date = start_date
        self.train_end_date = end_date
        self.predict_days = predict_days
        self.verbose = verbose
        self.data: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.reg_models: Dict[int, Ridge] = {}
        self.predictions: Dict[str, pd.Series] = {}
        self.scores: Dict[str, Dict[str, float]] = {}
        self.data_end_date: str = self._calculate_data_end_date()
        self._load_or_fetch_data()
        self._prepare_regression_data()

    def _calculate_data_end_date(self) -> str:
        try:
            train_end = dt.datetime.strptime(self.train_end_date, "%Y-%m-%d")
            data_end = train_end + dt.timedelta(days=int(self.predict_days * 1.5))
            if data_end > dt.datetime.today():
                data_end = dt.datetime.today() - dt.timedelta(days=1)
            return data_end.strftime("%Y-%m-%d")
        except Exception as e:
            raise RuntimeError(f"Could not calculate data end date: {e}")

    def _load_or_fetch_data(self) -> None:
        """Loads data from local CSV or falls back to yfinance."""
        req_start_dt = dt.datetime.strptime(self.start_date, "%Y-%m-%d")
        req_end_dt = dt.datetime.strptime(self.data_end_date, "%Y-%m-%d")
        
        search_pattern = os.path.join(CACHE_DIR, f'Stock-{self.ticker}-*.csv')
        local_files = glob.glob(search_pattern)
        best_match_path = None
        
        for file_path in local_files:
            try:
                date_part = os.path.basename(file_path).replace(f'Stock-{self.ticker}-', '').replace('.csv', '')
                parts = date_part.split('-')
                if len(parts) == 6: 
                    cached_start = dt.datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                    cached_end = dt.datetime(int(parts[3]), int(parts[4]), int(parts[5]))
                    if req_start_dt >= cached_start and req_end_dt <= cached_end:
                        best_match_path = file_path
                        break
            except Exception:
                continue

        if best_match_path:
            try:
                self.data = pd.read_csv(best_match_path, index_col='Date', parse_dates=True)
                self.data = self.data[self.start_date : self.data_end_date] 
                self.data = _handle_missing_data(self.data)
                if self.verbose: print(f"✅ LOADED LOCAL DATA: {best_match_path}")
                return
            except Exception:
                pass # Fall through to download if local load fails

        # Fallback: Download
        if self.verbose: print(f"⚠️ Local file not found. Downloading {self.ticker}...")
        try:
            self.data = yf.download(self.ticker, start=self.start_date, end=self.data_end_date)
            self.data = _handle_missing_data(self.data)
            if self.data.empty: raise ValueError("Downloaded data is empty.")
        except Exception as e:
            raise IOError(f"Failed to download data: {e}")

    def _prepare_regression_data(self) -> None:
        if self.data is None: raise AttributeError("Data not loaded.")
            
        data_df = self.data.copy().reset_index()
        if 'Date' not in data_df.columns: data_df = data_df.reset_index()
            
        data_df = data_df[['Date', 'Adj Close']]
        data_df['Timeline'] = data_df.index.values

        train_end_dt = pd.to_datetime(self.train_end_date)
        train_mask = data_df['Date'] <= train_end_dt
        
        if train_mask.sum() == 0:
            self.split_point = int(DEFAULT_TRAIN_TEST_RATIO * len(data_df))
        else:
            self.split_point = data_df[train_mask].index.max() + 1
        
        total_samples = self.split_point + self.predict_days
        
        self.X_train = data_df['Timeline'].iloc[:self.split_point].values.reshape(-1, 1)
        self.y_train = data_df['Adj Close'].iloc[:self.split_point].values

        self.X_test = data_df['Timeline'].iloc[self.split_point:total_samples].values.reshape(-1, 1)
        self.y_test = data_df['Adj Close'].iloc[self.split_point:total_samples].values
        
        self.full_data = data_df[['Date', 'Adj Close']].iloc[:total_samples].copy()
        self.full_data.set_index('Date', inplace=True)
        
        if self.verbose:
            print(f"📊 Split Point: {self.split_point} | Train Size: {len(self.X_train)} | Test Size: {len(self.X_test)}")

    # Training methods

    def train_regression_model(self, degrees: List[int] = DEFAULT_REGRESSION_DEGREES, alpha: float = 50.0) -> None:
        """
        Trains Ridge Regression models (L2 Regularization) to prevent overfitting.
        :param alpha: Regularization strength (Higher = less overfitting).
        """
        if self.X_train is None: raise RuntimeError("Data not prepared.")

        self.reg_models = {}
        
        for deg in degrees:
            try:
                poly = PolynomialFeatures(degree=deg)
                X_train_poly = poly.fit_transform(self.X_train)
                X_full_poly = poly.transform(np.arange(len(self.full_data)).reshape(-1, 1))

                # Trying Ridge instead of LinearRegression
                model = Ridge(alpha=alpha) 
                model.fit(X_train_poly, self.y_train)

                pred_full = model.predict(X_full_poly)
                name = f'Ridge_Deg_{deg}'

                self.reg_models[deg] = model
                self.predictions[name] = pd.Series(pred_full, index=self.full_data.index, name=name)
                self._evaluate_model(name, pred_full)
                
            except Exception as e:
                warnings.warn(f"Failed to train degree {deg}: {e}")
        
        if self.verbose: print(f"✅ Trained Ridge Regressions (Alpha={alpha}): {list(self.reg_models.keys())}")

    def train_momentum_model(self, lookback_days: int = 15, momentum_weight: float = 0.3, alpha: float = 50.0) -> None:
        """Combines average trend with short-term momentum (using Ridge)."""
        # 1. Average of existing regression models
        reg_preds = [self.predictions[k].values for k in self.predictions if 'Ridge' in k]
        if not reg_preds:
            if self.verbose: print("⚠️ No Ridge models found. Skipping Momentum.")
            return
            
        avg_trend = np.mean(reg_preds, axis=0)

        # 2. Short-term Momentum
        start = len(self.X_train) - lookback_days
        X_mom = self.X_train[start:]
        y_mom = self.y_train[start:]
        
        # Using Ridge for momentum to avoid overreacting to noise
        model_mom = Ridge(alpha=alpha)
        model_mom.fit(X_mom, y_mom)
        X_full = np.arange(len(self.full_data)).reshape(-1, 1)
        mom_trend = model_mom.predict(X_full)

        # 3. Blend
        mom_start = max(0, self.split_point - lookback_days)
        combined = avg_trend.copy()
        
        # Weighted blend in the active window
        combined[mom_start:] = (1 - momentum_weight) * avg_trend[mom_start:] + \
                               momentum_weight * mom_trend[mom_start:]

        self.predictions['Reg_Momentum'] = pd.Series(combined, index=self.full_data.index, name='Reg_Momentum')
        self._evaluate_model('Reg_Momentum', combined)
        
        if self.verbose: print("✅ Trained Momentum Model")

    def train_fft_model(self, num_harmonics: int = 4) -> None:
        """FFT model on top of Momentum residuals."""
        if 'Reg_Momentum' not in self.predictions:
            if self.verbose: print("⚠️ Reg_Momentum missing. Skipping FFT.")
            return
            
        trend = self.predictions['Reg_Momentum'].values
        residuals = self.y_train - trend[:len(self.y_train)]

        waveform = _calculate_fft_waveform(residuals, num_harmonics, len(self.full_data))
        final_pred = trend + waveform

        self.predictions['FFT_Model'] = pd.Series(final_pred, index=self.full_data.index, name='FFT_Model')
        self._evaluate_model('FFT_Model', final_pred)
        
        if self.verbose: print(f"✅ Trained FFT Model (Harmonics={num_harmonics})")

    def train_lstm_model(self, sequence_length=50, units=50, epochs=50, batch_size=32):
        """Trains LSTM with EarlyStopping to prevent overfitting."""
        if not KERAS_AVAILABLE: return
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices = self.full_data['Adj Close'].values.reshape(-1, 1)
        scaled_prices = scaler.fit_transform(prices)

        X_seq, y_seq = [], []
        for i in range(len(scaled_prices) - sequence_length):
            X_seq.append(scaled_prices[i:i + sequence_length, 0])
            y_seq.append(scaled_prices[i + sequence_length, 0])

        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        X_seq = X_seq[:, :, np.newaxis]
        
        train_size = len(self.y_train) - sequence_length
        if train_size < 10: return

        X_train_lstm = X_seq[:train_size]
        y_train_lstm = y_seq[:train_size]
        X_remainder = X_seq[train_size:]
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(units, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=0)
        
        if self.verbose: print("Training LSTM with Early Stopping...")
        model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, 
                  verbose=0, callbacks=[es])
        lstm_preds = model.predict(X_remainder, verbose=0)
        dummy = np.zeros(shape=(len(lstm_preds), 1))
        dummy[:, 0] = lstm_preds[:, 0]
        descaled = scaler.inverse_transform(dummy)[:, 0]
        start_idx = len(y_seq) - len(descaled) + sequence_length
        full_pred = self.full_data['Adj Close'].values.copy()
        full_pred[start_idx:] = descaled
        self.predictions['LSTM_Model'] = pd.Series(full_pred, index=self.full_data.index, name='LSTM_Model')
        self._evaluate_model('LSTM_Model', full_pred)
        
        if self.verbose: print("✅ Trained LSTM Model")

    def _evaluate_model(self, model_name: str, full_prediction: np.ndarray) -> None:
        """Calculates R2, MSE, and Sharpe Ratio on the Test Set."""
        preds = full_prediction[:len(self.full_data)]
        
        if len(self.y_test) > 0:
            y_test_pred = preds[self.split_point : self.split_point + len(self.y_test)]
            
            r2_test = r2_score(self.y_test, y_test_pred)
            mse_test = mean_squared_error(self.y_test, y_test_pred)
            test_slice = self.full_data.iloc[self.split_point : self.split_point + len(self.y_test)].copy()
            test_slice['Pred'] = y_test_pred
            test_slice['Returns'] = test_slice['Adj Close'].pct_change()
            test_slice['Signal'] = np.where(test_slice['Pred'].diff() > 0, 1.0, 0.0)
            strat_ret = test_slice['Signal'].shift(1) * test_slice['Returns']
            sharpe = _calculate_sharpe_ratio(strat_ret.dropna())
        else:
            r2_test, mse_test, sharpe = 0.0, 0.0, 0.0

        self.scores[model_name] = {
            'R2 Test': r2_test,
            'MSE': mse_test,
            'Sharpe': sharpe
        }

    # Reporting and Plotting

    def plot_terminal(self, model_names='all', title=None) -> None:
        """Plots predictions directly in the terminal using Plotext."""
        if not TERMINAL_PLOTTING:
            print("❌ 'plotext' library missing. Cannot plot in terminal.")
            return

        if model_names == 'all':
            model_names = list(self.predictions.keys())
        dates_str = [d.strftime('%Y-%m-%d') for d in self.full_data.index]
        actuals = self.full_data['Adj Close'].tolist()
        
        plt_term.clf()
        plt_term.date_form('Y-m-d')
        plt_term.plot(dates_str, actuals, label='Actual', color='white')
        
        for name in model_names:
            if name in self.predictions:
                # Add Sharpe to legend
                sr = self.scores[name]['Sharpe'] if name in self.scores else 0
                label = f"{name} (SR: {sr:.2f})"
                plt_term.plot(dates_str, self.predictions[name].tolist(), label=label)

        # Split Line
        split_date = dates_str[self.split_point] if self.split_point < len(dates_str) else dates_str[-1]
        plt_term.vline(split_date, color='red')

        plt_term.title(title or f"{self.ticker} Predictions (Terminal Mode)")
        plt_term.theme('pro')
        plt_term.show()

    def plot_window(self, model_names='all', title=None):
        """Plots using the standard Matplotlib GUI window."""
        if self.full_data is None: return

        if model_names == 'all': model_names = list(self.predictions.keys())
        
        plt.figure(figsize=(14, 7))
        # Actuals
        plt.plot(self.full_data.index, self.full_data['Adj Close'], label='Actual', color='black', linewidth=2, alpha=0.7)
        # Predictions
        for name in model_names:
            if name in self.predictions:
                sr = self.scores[name]['Sharpe'] if name in self.scores else 0
                plt.plot(self.full_data.index, self.predictions[name], label=f"{name} (SR: {sr:.2f})", linestyle='--')
        # Split Line
        split_date = self.full_data.index[self.split_point]
        plt.axvline(split_date, color='red', linestyle=':', label='Train/Test Split')
        plt.title(title or f"{self.ticker} Predictions (GUI Mode)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def report_scores(self) -> None:
        if not self.scores:
            print("No models evaluated.")
            return

        scores_df = pd.DataFrame.from_dict(self.scores, orient='index')
        scores_df = scores_df.sort_values(by='Sharpe', ascending=False)
        
        print("\n--- Model Evaluation (Ranked by Sharpe Ratio) ---")
        print(scores_df.to_string(float_format="{:.4f}".format))
        print("-------------------------------------------------\n")

def main_example():
    print("--- Professional Stock Regressor (Terminal & GUI) ---")
    
    try:
        regressor = ProductionStockRegressor(
            ticker='GLD', # Matches local file preference
            start_date='2015-06-01',
            end_date='2017-01-01',
            predict_days=100,
            verbose=True
        )
    except Exception as e:
        print(f"Init Error: {e}")
        return

    # Train (Using settings that reduce overfitting)
    regressor.train_regression_model(degrees=[1, 2], alpha=50.0) 
    regressor.train_momentum_model(lookback_days=30, momentum_weight=0.4, alpha=50.0)
    regressor.train_fft_model(num_harmonics=6)

    if KERAS_AVAILABLE:
        regressor.train_lstm_model(sequence_length=60, epochs=20, units=100) # Epochs increased safely due to EarlyStopping
    
    regressor.report_scores()
    
    print("\n--- Plotting in Terminal ---")
    regressor.plot_terminal(model_names=['Reg_Momentum', 'FFT_Model', 'Ridge_Deg_1'])

    print("\n--- Opening Graph Window... ---")
    regressor.plot_window(model_names=['Reg_Momentum', 'FFT_Model', 'Ridge_Deg_1'])

if __name__ == '__main__':
    main_example()
