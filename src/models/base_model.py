from abc import ABC, abstractmethod
import numpy as np

class BaseForecaster(ABC):
    """
    Abstract Base Class that enforces a standard interface for all predictive models.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, X_full: np.ndarray) -> np.ndarray:
        """Generates predictions for the provided timeline."""
        pass