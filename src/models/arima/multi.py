from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

__all__ = ["MultiARIMA"]


class MultiARIMA(dict):
    def __init__(self, features: Iterable, history: np.ndarray, order: Optional[Tuple] = (0, 0, 0)):
        super().__init__()
        if len(features) != history.shape[1]:
            raise ValueError(f"Expected `features` and `history` to have the same length.")
        for col, history in zip(features, history):
            self[col] = ARIMA(history, order=order)

    def fit(self) -> Dict[str, ARIMAResults]:
        return MultiARIMAResults((k, v.fit()) for k, v in self.items())


class MultiARIMAResults(dict):
    def predict(self):
        return {k: v.forecast() for k, v in self.items()}