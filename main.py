import warnings
import numpy as np
from src.data.data_loader import DataLoader
from src.models.ridge_momentum import RidgeMomentumModel
from src.models.fft_residual import FFTResidualModel
from src.evaluation.metrics import evaluate_predictions
from src.visualization.plotters import plot_gui

warnings.filterwarnings('ignore')

def run_pipeline():
    print("--- Initializing Quant Forecasting Engine ---\n")

    # 1. Load and Prep Data
    loader = DataLoader(
        ticker='GLD',
        start_date='2015-06-01',
        train_end_date='2017-01-01',
        predict_days=90,
        verbose=True
    )
    X_train, y_train, X_test, y_test, full_data, split_point = loader.prepare_data()

    # We need a full timeline array (0 to N) for the models to predict the future
    X_full = np.arange(len(full_data)).reshape(-1, 1)

    # 2. Init Models
    print("\n--- Training Models ---")
    ridge_model = RidgeMomentumModel(degrees=[1], alpha=100.0, lookback_days=30, momentum_weight=0.05)
    
    # Pass ridge into fft
    fft_model = FFTResidualModel(base_model=ridge_model, num_harmonics=6)

    models = [ridge_model, fft_model]
    predictions = {}

    # 3. Train and Pred
    for model in models:
        print(f"Fitting {model.name}...")
        model.fit(X_train, y_train)
        predictions[model.name] = model.predict(X_full)

    # 4. Eval Performance
    print("\n--- Model Evaluation (Ranked by Sharpe Ratio) ---")
    scores_df = evaluate_predictions(full_data, predictions, split_point)
    if not scores_df.empty:
        print(scores_df.to_string(float_format="{:.4f}".format))
    else:
        print("Not enough test data to evaluate metrics.")

    # 5. Visualize Results
    print("\n--- Generating Visualizations ---")
    
    # Pop up GUI
    plot_gui(full_data, predictions, split_point, title=f"{loader.ticker} Forecast vs Actuals")

if __name__ == '__main__':
    run_pipeline()