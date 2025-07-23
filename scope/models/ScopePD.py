import numpy as np
from scope.models.base_model import BaseModel


class ScOPEPD(BaseModel):
    
    def __init__(self, use_matching_method: bool = False, distance_metric: str = "cosine") -> None:
        self._is_matching_method: bool = use_matching_method
        self.epsilon: float = 1e-12

        self.supported_distance_metrics = {
            "cosine": lambda x1, x2: self.__cosine_distance__(x1, x2),
            "euclidean": lambda x1, x2: self.__euclidean_distance__(x1, x2),
            "manhattan": lambda x1, x2: self.__manhattan_distance__(x1, x2),
            "chebyshev": lambda x1, x2: self.__chebyshev_distance__(x1, x2),
            "canberra": lambda x1, x2: self.__canberra_distance__(x1, x2),
            "minkowski": lambda x1, x2: self.__minkowski_distance__(x1, x2, p=3),
            "braycurtis": lambda x1, x2: self.__braycurtis_distance__(x1, x2),
            "hamming": lambda x1, x2: self.__hamming_distance__(x1, x2),
            "correlation": lambda x1, x2: self.__correlation_distance__(x1, x2)
        }
        
        if distance_metric not in self.supported_distance_metrics:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        self.distance_metric = self.supported_distance_metrics[distance_metric]

    @staticmethod
    def __calc_prototype__(data: np.ndarray) -> np.ndarray:
        return np.mean(data, axis=0)
    
    @staticmethod
    def __cosine_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    
    @staticmethod
    def __euclidean_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        return np.linalg.norm(x1 - x2)
    
    @staticmethod
    def __manhattan_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sum(np.abs(x1 - x2))
    
    @staticmethod
    def __chebyshev_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        return np.max(np.abs(x1 - x2))
    
    @staticmethod
    def __canberra_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        numerator = np.abs(x1 - x2)
        denominator = np.abs(x1) + np.abs(x2)
        mask = denominator != 0
        return np.sum(numerator[mask] / denominator[mask])
    
    @staticmethod
    def __minkowski_distance__(x1: np.ndarray, x2: np.ndarray, p: float = 2) -> float:
        return np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p)
    
    @staticmethod
    def __braycurtis_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sum(np.abs(x1 - x2)) / np.sum(np.abs(x1 + x2))
    
    @staticmethod
    def __hamming_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        return np.mean(x1 != x2)
    
    @staticmethod
    def __correlation_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        correlation = np.corrcoef(x1, x2)[0, 1]
        return 1 - correlation

    def __forward__(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> np.ndarray:
        if self._is_matching_method:
            raise ValueError("Matching method is not enabled. Set use_matching_method to False.")
        
        prototype: np.ndarray = self.__calc_prototype__(current_cluster)
        
        score: float = self.distance_metric(current_sample, prototype)
        
        return score