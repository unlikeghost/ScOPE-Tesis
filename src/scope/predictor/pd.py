import numpy as np
from scope.predictor.base import _BasePredictor


class ScOPEPD(_BasePredictor):

    def __init__(self, distance_metric: str = "cosine", use_prototypes: bool = False, use_softmax: bool = False, epsilon: float = 1e-12) -> None:
        self.use_prototypes = use_prototypes

        self.supported_distance_metrics = {
            "cosine": lambda x1, x2: self.__cosine_distance__(x1, x2),
            "euclidean": lambda x1, x2: self.__euclidean_distance__(x1, x2),
            "manhattan": lambda x1, x2: self.__manhattan_distance__(x1, x2),
            "chebyshev": lambda x1, x2: self.__chebyshev_distance__(x1, x2),
            "canberra": lambda x1, x2: self.__canberra_distance__(x1, x2),
            "minkowski": lambda x1, x2: self.__minkowski_distance__(x1, x2, p=3),
            "braycurtis": lambda x1, x2: self.__braycurtis_distance__(x1, x2),
            "hamming": lambda x1, x2: self.__hamming_distance__(x1, x2),
            "correlation": lambda x1, x2: self.__correlation_distance__(x1, x2),
            "dot_product": lambda x1, x2: self.__dot_product__(x1, x2)
        }
        
        if distance_metric not in self.supported_distance_metrics:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        self.distance_metric = self.supported_distance_metrics[distance_metric]

        super().__init__(use_softmax=use_softmax, epsilon=epsilon)

    @staticmethod
    def __calc_prototype__(data: np.ndarray) -> np.ndarray:
        return np.mean(data, axis=0)
    
    def __cosine_distance__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Robust cosine distance with zero-vector handling"""
        # Ensure inputs are 1D
        x1 = np.asarray(x1).flatten()
        x2 = np.asarray(x2).flatten()
        
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        
        # Handle zero vectors
        if norm1 < self.epsilon or norm2 < self.epsilon:
            return 1.0  # Maximum distance for zero vectors
        
        cosine_sim = np.dot(x1, x2) / (norm1 * norm2)
        
        # Ensure scalar and clip to valid range
        if np.isscalar(cosine_sim):
            cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        else:
            cosine_sim = np.clip(cosine_sim.item(), -1.0, 1.0)
        
        return float(1.0 - cosine_sim)
    
    @staticmethod
    def __euclidean_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        x1, x2 = np.asarray(x1).flatten(), np.asarray(x2).flatten()
        return float(np.linalg.norm(x1 - x2))
    
    @staticmethod
    def __manhattan_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        x1, x2 = np.asarray(x1).flatten(), np.asarray(x2).flatten()
        return float(np.sum(np.abs(x1 - x2)))
    
    @staticmethod
    def __chebyshev_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        x1, x2 = np.asarray(x1).flatten(), np.asarray(x2).flatten()
        return float(np.max(np.abs(x1 - x2)))
    
    def __canberra_distance__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Robust Canberra distance with zero-denominator handling"""
        x1, x2 = np.asarray(x1).flatten(), np.asarray(x2).flatten()
        
        numerator = np.abs(x1 - x2)
        denominator = np.abs(x1) + np.abs(x2)
        
        # Only compute for non-zero denominators
        mask = denominator > self.epsilon
        if not np.any(mask):
            return 0.0  # All elements are zero
        
        return float(np.sum(numerator[mask] / denominator[mask]))
    
    @staticmethod
    def __minkowski_distance__(x1: np.ndarray, x2: np.ndarray, p: float = 2) -> float:
        x1, x2 = np.asarray(x1).flatten(), np.asarray(x2).flatten()
        return float(np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p))
    
    def __braycurtis_distance__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Robust Bray-Curtis distance with zero-denominator handling"""
        x1, x2 = np.asarray(x1).flatten(), np.asarray(x2).flatten()
        
        numerator = np.sum(np.abs(x1 - x2))
        denominator = np.sum(np.abs(x1 + x2))
        
        if denominator < self.epsilon:
            return 0.0  # Both vectors are zero
        
        return float(numerator / denominator)
    
    @staticmethod
    def __hamming_distance__(x1: np.ndarray, x2: np.ndarray) -> float:
        x1, x2 = np.asarray(x1).flatten(), np.asarray(x2).flatten()
        return float(np.mean(x1 != x2))
    
    def __correlation_distance__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Robust correlation distance with constant-vector handling"""
        x1, x2 = np.asarray(x1).flatten(), np.asarray(x2).flatten()
        
        # Check for constant vectors (zero variance)
        if np.var(x1) < self.epsilon or np.var(x2) < self.epsilon:
            # If both are constant and equal, distance is 0
            if np.allclose(x1, x2, atol=self.epsilon):
                return 0.0
            # If one or both are constant but different, distance is 1
            return 1.0
        
        try:
            correlation_matrix = np.corrcoef(x1, x2)
            correlation = correlation_matrix[0, 1]
            
            # Handle NaN case (shouldn't happen with the variance check, but just in case)
            if np.isnan(correlation):
                return 1.0
            
            # Clip correlation to valid range
            correlation = np.clip(correlation, -1.0, 1.0)
            return float(1.0 - correlation)
            
        except Exception:
            # Fallback: if correlation fails for any reason
            return 1.0

    @staticmethod
    def __dot_product__(x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute dot product distance with zero-vector handling"""
        x1, x2 = np.asarray(x1).flatten(), np.asarray(x2).flatten()
        
        if np.linalg.norm(x1) < 1e-12 or np.linalg.norm(x2) < 1e-12:
            return 1.0  # Maximum distance for zero vectors
        
        dot_prod = np.dot(x1, x2)
        return float(1.0 - dot_prod)
    
    def __forward__(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> float:
        """
        Compute distance between sample and cluster prototype.
        
        Args:
            current_cluster: Cluster data matrix
            current_sample: Sample data matrix
            
        Returns:
            Distance score as float
        """
        
        if self.use_prototypes:
            prototype = self.__calc_prototype__(current_cluster)
            score = self.distance_metric(current_sample, prototype)
        
        else:
            
            scores = []
            for kw_sample in current_cluster:
                score = self.distance_metric(current_sample, kw_sample)
                scores.append(score)
                
            score = np.sum(scores)
            
    
        # Ensure we have a float (handle both numpy scalars and Python floats)
        if hasattr(score, 'item'):
            score = score.item()
        else:
            score = float(score)
        
        # Validate result
        if np.isnan(score) or np.isinf(score):
            raise ValueError(f"Invalid score calculated: {score}. Check input data for NaN or Inf values.")
        
        return score