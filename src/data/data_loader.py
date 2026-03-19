import os
import glob
import warnings
import datetime as dt
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = 'stock_cache'

class DataLoader:
    """
    Handles fetching, caching, cleaning, and splitting financial time-series data.
    """
    def __init__(self, ticker: str, start_date: str, train_end_date: str, predict_days: int, verbose: bool = True):
        self.ticker = ticker
        self.start_date = start_date
        self.train_end_date = train_end_date
        self.predict_days = predict_days
        self.verbose = verbose
        
        self.data_end_date = self._calculate_data_end_date()
        self.full_data: Optional[pd.DataFrame] = None

    def _calculate_data_end_date(self) -> str:
        """Calculates how much future data we need based on prediction days."""
        try:
            train_end = dt.datetime.strptime(self.train_end_date, "%Y-%m-%d")
            data_end = train_end + dt.timedelta(days=int(self.predict_days * 1.5))
            if data_end > dt.datetime.today():
                data_end = dt.datetime.today() - dt.timedelta(days=1)
            return data_end.strftime("%Y-%m-%d")
        except Exception as e:
            raise RuntimeError(f"Could not calculate data end date: {e}")

    @staticmethod
    def _handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing data using forward and backward fill."""
        return df.copy().ffill().bfill()

    def fetch_data(self) -> pd.DataFrame:
        """Loads data from the local CSV cache or falls back to yfinance."""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        req_start_dt = dt.datetime.strptime(self.start_date, "%Y-%m-%d")
        req_end_dt = dt.datetime.strptime(self.data_end_date, "%Y-%m-%d")
        
        search_pattern = os.path.join(CACHE_DIR, f'Stock-{self.ticker}-*.csv')
        best_match_path = None
        
        for file_path in glob.glob(search_pattern):
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

        # 2. Load Local Data if found
        if best_match_path:
            try:
                data = pd.read_csv(best_match_path, index_col='Date', parse_dates=True)
                data = data[self.start_date : self.data_end_date] 
                self.full_data = self._handle_missing_data(data)
                if self.verbose: 
                    print(f"✅ LOADED LOCAL DATA: {best_match_path}")
                return self.full_data
            except Exception:
                pass # Fall through to download if local load fails

        # 3. Fallback: Download with yfinance
        if self.verbose: 
            print(f"⚠️ Local file not found. Downloading {self.ticker} via yfinance...")
        try:
            data = yf.download(self.ticker, start=self.start_date, end=self.data_end_date, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            self.full_data = self._handle_missing_data(data)
            if self.full_data.empty: 
                raise ValueError("Downloaded data is empty.")
            return self.full_data
        except Exception as e:
            raise IOError(f"Failed to download data: {e}")

    def prepare_data(self, train_test_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int]:
        """
        Formats the fetched data into X and y arrays for training and testing.
        Returns: X_train, y_train, X_test, y_test, full_data, split_point
        """
        if self.full_data is None:
            self.fetch_data()
            
        data_df = self.full_data.copy().reset_index()
        if 'Date' not in data_df.columns: 
            data_df = data_df.reset_index()
            
        data_df = data_df[['Date', 'Adj Close']]
        data_df['Timeline'] = data_df.index.values

        train_end_dt = pd.to_datetime(self.train_end_date)
        train_mask = data_df['Date'] <= train_end_dt
        
        if train_mask.sum() == 0:
            split_point = int(train_test_ratio * len(data_df))
        else:
            split_point = data_df[train_mask].index.max() + 1
        
        total_samples = split_point + self.predict_days
        
        # Arrays sliced
        X_train = data_df['Timeline'].iloc[:split_point].values.reshape(-1, 1)
        y_train = data_df['Adj Close'].iloc[:split_point].values

        X_test = data_df['Timeline'].iloc[split_point:total_samples].values.reshape(-1, 1)
        y_test = data_df['Adj Close'].iloc[split_point:total_samples].values
        
        # Prep the final dataframe struct used for indexing pred
        final_full_data = data_df[['Date', 'Adj Close']].iloc[:total_samples].copy()
        final_full_data.set_index('Date', inplace=True)
        
        if self.verbose:
            print(f"📊 Split Point: {split_point} | Train Size: {len(X_train)} | Test Size: {len(X_test)}")

        return X_train, y_train, X_test, y_test, final_full_data, split_point