import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates annualized Sharpe Ratio."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)

def evaluate_predictions(full_data: pd.DataFrame, predictions: dict, split_point: int) -> pd.DataFrame:
    """
    Evaluates model predictions on the unseen test dataset.
    Returns a formatted Pandas DataFrame ranked by Sharpe Ratio.
    """
    scores = {}
    
    # Extract the true vals for the test period
    test_data = full_data.iloc[split_point:].copy()
    y_test = test_data['Adj Close'].values
    
    for name, pred_array in predictions.items():
        if len(y_test) == 0 or len(pred_array) <= split_point:
            continue
            
        # Slice the pred arr to match the test period
        y_test_pred = pred_array[split_point : split_point + len(y_test)]
        
        # Ensure lengths match exactly to avoid numpy broadcasting errors
        min_len = min(len(y_test), len(y_test_pred))
        y_test_actual = y_test[:min_len]
        y_test_pred_trim = y_test_pred[:min_len]
        
        # Metrics
        r2 = r2_score(y_test_actual, y_test_pred_trim)
        mse = mean_squared_error(y_test_actual, y_test_pred_trim)
        
        # Metric: Trading Strat
        temp_df = pd.DataFrame({
            'Adj Close': y_test_actual,
            'Pred': y_test_pred_trim
        })
        temp_df['Returns'] = temp_df['Adj Close'].pct_change()
        # Buy signal if prediction goes up, flat if it goes down
        temp_df['Signal'] = np.where(temp_df['Pred'].diff() > 0, 1.0, 0.0)
        strat_ret = temp_df['Signal'].shift(1) * temp_df['Returns']
        sharpe = calculate_sharpe_ratio(strat_ret.dropna())
        
        scores[name] = {
            'R2 Test': r2, 
            'MSE': mse, 
            'Sharpe': sharpe
        }
        
    # Format / Sort results 
    scores_df = pd.DataFrame.from_dict(scores, orient='index')
    if not scores_df.empty:
        scores_df = scores_df.sort_values(by='Sharpe', ascending=False)
        
    return scores_df