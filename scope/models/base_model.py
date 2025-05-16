import numpy as np
from typing import Dict, Union
from abc import abstractmethod

from scope.distances import MatchingMethods


class BaseModel:
    start_key_value_matrix: str = 'ScOPEC_'
    start_key_value_sample: str = 'ScOPES_'

    matching_method: MatchingMethods = MatchingMethods()

    def __matching_score__(self, x1: np.ndarray, x2: np.ndarray, method: str = 'matching') -> float:
        return self.matching_method(method=method, x1=x1, x2=x2)

    @staticmethod
    def __softmax__(scores: np.ndarray) -> np.ndarray:
        # exp_scores = np.exp(scores - np.max(scores))  # **Softmax Stabilization**
        # return exp_scores/np.sum(exp_scores)
        return scores / np.sum(scores)

    @staticmethod
    def __gaussian_function__(x: np.ndarray, sigma: Union[np.ndarray, float]) -> np.ndarray:
        return np.exp(
            -0.5 * np.square(
                (x / sigma)
            )
        )

    @abstractmethod
    def forward(self, data_matrix: Dict[str, np.ndarray], softmax: bool = False) -> np.ndarray:
        raise NotImplementedError
