import ot
import numpy as np

from warnings import warn, filterwarnings

from scope.distances import MatchingMethods
from scope.models.base_model import BaseModel

filterwarnings("once")


class ScOPEOT(BaseModel):


    def __init__(self, use_matching_method: bool = False, matching_method_name: str = None) -> None:
        self._is_matching_method: bool = use_matching_method
        self.epsilon: float = 1e-12
        
        if use_matching_method:
            if matching_method_name is None:
                matching_method_name = 'dice'
                warn(f"No matching method provided. Using default: {matching_method_name}")
                
            self.matching_method_name: str = matching_method_name
            self.matching_method: MatchingMethods = MatchingMethods()

        else:
            warn("Using OT method, matching method will be ignored.")


    def __matching_score__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if not self._is_matching_method:
            raise ValueError("Matching method is not enabled. Set use_matching_method to True.")
        return self.matching_method(method=self.matching_method_name, x1=x1, x2=x2)

    def __calc_cost_matrix__(self, sample: np.ndarray, kw_samples: np.ndarray) -> np.ndarray:

        if self._is_matching_method:
            cost_matrix: np.ndarray = np.zeros(shape=(len(kw_samples), 1))
            for index, kw_sample in enumerate(kw_samples):
                kw_sample: np.ndarray = kw_sample.reshape(1, -1)
                this_cost = self.__matching_score__(x1=sample, x2=kw_sample)
                cost_matrix[index] = this_cost
        
        else:
            cost_matrix: np.ndarray = ot.dist(kw_samples, sample, metric='euclidean')

        return cost_matrix

    @staticmethod
    def __wasserstein_distance__(sample_weights: np.ndarray, cluster_weights: np.ndarray,
                                 cost_matrix: np.ndarray) -> float:
        return ot.emd2(cluster_weights, sample_weights, cost_matrix)
    
    def __forward__(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> np.ndarray:

        cost_matrix: np.ndarray = self.__calc_cost_matrix__(sample=current_sample, kw_samples=current_cluster)
        
        current_cluster_weights: np.ndarray = np.ones(current_cluster.shape[0]) / current_cluster.shape[0]
        current_sample_weights: np.ndarray = np.ones(current_sample.shape[0]) / current_sample.shape[0]

        score: float = self.__wasserstein_distance__(
            cost_matrix=cost_matrix,
            cluster_weights=current_cluster_weights,
            sample_weights=current_sample_weights,
        )
        
        return score