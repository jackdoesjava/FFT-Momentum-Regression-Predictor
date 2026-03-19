import matplotlib.pyplot as plt
import pandas as pd


def plot_gui(full_data: pd.DataFrame, predictions: dict, split_point: int, title: str = "Model Predictions") -> None:
    """Generates a high-quality matplotlib visualization."""
    plt.figure(figsize=(14, 7))
    
    # Plot actual
    plt.plot(full_data.index, full_data['Adj Close'], label='Actual', color='black', linewidth=2, alpha=0.7)
    # Plot preds
    for name, pred_array in predictions.items():
        plt.plot(full_data.index[:len(pred_array)], pred_array, label=name, linestyle='--')
        
    # Mark the Train/Test barrier
    if split_point < len(full_data):
        split_date = full_data.index[split_point]
        plt.axvline(split_date, color='red', linestyle=':', label='Train/Test Split')
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
