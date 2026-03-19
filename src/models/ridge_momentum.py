import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from src.models.base_model import BaseForecaster

class RidgeMomentumModel(BaseForecaster):
    """
    Blends an averaged Polynomial Ridge Regression with a short-term Momentum Ridge Regression.
    """
    def __init__(self, degrees: list = [1], alpha: float = 50.0, 
                 lookback_days: int = 30, momentum_weight: float = 0.4):
        super().__init__(name="Ridge_Momentum")
        self.degrees = degrees
        self.alpha = alpha
        self.lookback_days = lookback_days
        self.momentum_weight = momentum_weight
        
        self.reg_models = {}
        self.momentum_model = None
        self.X_train_len = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.X_train_len = len(X_train)
        
        # 1. Train base poly Ridge models
        for deg in self.degrees:
            poly = PolynomialFeatures(degree=deg)
            X_train_poly = poly.fit_transform(X_train)
            
            model = Ridge(alpha=self.alpha)
            model.fit(X_train_poly, y_train)
            self.reg_models[deg] = model

        # 2. Train short-term momentum model
        start = max(0, self.X_train_len - self.lookback_days)
        X_mom = X_train[start:]
        y_mom = y_train[start:]
        
        self.momentum_model = Ridge(alpha=self.alpha)
        self.momentum_model.fit(X_mom, y_mom)

    def predict(self, X_full: np.ndarray) -> np.ndarray:
        # 1. Get avg trend from base models
        reg_preds = []
        for deg, model in self.reg_models.items():
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=deg)
            X_full_poly = poly.fit_transform(X_full)
            reg_preds.append(model.predict(X_full_poly))
        
        avg_trend = np.mean(reg_preds, axis=0) if reg_preds else np.zeros(len(X_full))
        
        # 2. Get short-term momentum trend
        mom_trend = self.momentum_model.predict(X_full)
        
        # 3. Blend them with a short hinge
        mom_start = max(0, self.X_train_len - self.lookback_days)
        combined = avg_trend.copy()
        
        if mom_start < len(combined):
            # Create a short 10-day transition to smooth the cliff
            transition_len = min(10, len(combined) - mom_start)
            smooth_weights = np.linspace(0, self.momentum_weight, transition_len)
            
            window_end = mom_start + transition_len
            
            # Apply the curve ONLY to the 10-day hinge
            combined[mom_start:window_end] = (
                (1 - smooth_weights) * avg_trend[mom_start:window_end] + 
                smooth_weights * mom_trend[mom_start:window_end]
            )
            
            # Lock in a static, perfectly straight line for the rest of the forecast
            if window_end < len(combined):
                combined[window_end:] = (
                    (1 - self.momentum_weight) * avg_trend[window_end:] + 
                    self.momentum_weight * mom_trend[window_end:]
                )
            
        return combined