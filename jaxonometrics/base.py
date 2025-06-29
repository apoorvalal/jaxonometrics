from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import jax.numpy as jnp


class BaseEstimator(ABC):
    """Base class for all estimators in jaxonometrics."""

    def __init__(self):
        self.params: Optional[Dict[str, jnp.ndarray]] = None

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseEstimator":
        """Fit the model to the data."""
        raise NotImplementedError

    def summary(self) -> None:
        """Print a summary of the model results."""
        if self.params is None:
            print("Model has not been fitted yet.")
            return

        print(f"{self.__class__.__name__} Results")
        print("=" * 30)
        for param_name, param_value in self.params.items():
            print(f"{param_name}: {param_value}")
        print("=" * 30)
