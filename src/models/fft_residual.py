import numpy as np
from src.models.base_model import BaseForecaster

class FFTResidualModel(BaseForecaster):
    """
    Models the cyclical residuals (errors) of a base trend model using Fast Fourier Transform.
    """
    def __init__(self, base_model: BaseForecaster, num_harmonics: int = 4):
        # Dynamically named so it's clear what base model it's wrapping
        super().__init__(name=f"FFT_Wrapped_{base_model.name}")
        self.base_model = base_model
        self.num_harmonics = num_harmonics
        
        # We will learn these during fit()
        self.amplitudes = []
        self.phases = []
        self.frequencies = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Calculates dominant frequencies from the base model's residuals."""
        # 1. Get the base trend preds
        base_preds = self.base_model.predict(X_train)
        
        # 2. Calc what is missed by base model
        residuals = y_train - base_preds
        n = len(residuals)
        
        if n == 0: return

        # 3. FFT on the residuals
        fft_coeffs = np.fft.fft(residuals)
        freqs = np.fft.fftfreq(n)

        # 4. Find the most dominant freqs (excluding DC component)
        sorted_indices = np.argsort(np.abs(fft_coeffs[1:n//2]))[::-1] + 1
        top_indices = [0] + list(sorted_indices[:self.num_harmonics])
        
        # 5. Store the paras so we can recreate the wave later
        self.amplitudes = []
        self.phases = []
        self.frequencies = []
        
        for i in top_indices:
            self.amplitudes.append(np.abs(fft_coeffs[i]) / n)
            self.phases.append(np.angle(fft_coeffs[i]))
            self.frequencies.append(freqs[i])
            
            # Add the mirrored negative freqs
            if i != 0 and i != n//2:
                neg_i = n - i
                self.amplitudes.append(np.abs(fft_coeffs[neg_i]) / n)
                self.phases.append(np.angle(fft_coeffs[neg_i]))
                self.frequencies.append(freqs[neg_i])

    def predict(self, X_full: np.ndarray) -> np.ndarray:
        """Extrapolates the learned waveform over a new time period."""
        # 1. Get the base trend for the whole future timeline
        base_preds = self.base_model.predict(X_full)
        
        # 2. Create the time arr for the waveform
        t = np.arange(len(X_full))
        fft_reconstructed = np.zeros(len(X_full))
        
        # 3. Reconstruct the wave using the learned harmonics
        for amp, freq, phase in zip(self.amplitudes, self.frequencies, self.phases):
            fft_reconstructed += amp * np.cos(2 * np.pi * freq * t + phase)
            
        # 4. Add the cyclical wave back onto the base trend
        return base_preds + fft_reconstructed