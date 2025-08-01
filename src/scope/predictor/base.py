
import numpy as np
from typing import Dict, Union, Any, List
from abc import abstractmethod, ABC



class _BasePredictor(ABC):
    start_key_value_matrix: str = 'ScOPE_KwSamples_'
    start_key_value_sample: str = 'ScOPE_UkSample_'
    
    def __init__(self, use_softmax: bool = False, epsilon: float = 1e-12):
        self.use_softmax = use_softmax
        self.epsilon = epsilon

    @staticmethod
    def __softmax__(scores: np.ndarray) -> np.ndarray:
        # exp_scores = np.exp(scores - np.max(scores))  # **Softmax Stabilization**
        # return exp_scores/np.sum(exp_scores)
        return np.clip(scores / np.sum(scores), 0.0, 1.0, dtype=np.float32)

    @staticmethod
    def __gaussian_function__(x: np.ndarray, sigma: Union[np.ndarray, float]) -> np.ndarray:
        return np.exp(
            -0.5 * np.square(
                (x / sigma)
            )
        )
    
    @abstractmethod
    def __forward__(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> float:
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    
    def forward(self, list_of_data: List[Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:

        if not isinstance(list_of_data, list):
            raise ValueError("Input should be a list of dictionaries containing data matrices.")
        
        if not list_of_data:
            return []
        
        output: List[Dict[str, Any]] = []
        
        for data_matrix in list_of_data:
            
            cluster_keys: list = list(
                filter(
                    lambda x: x.startswith(self.start_key_value_matrix),
                    data_matrix.keys()
                )
            )

            sample_keys: list = list(
                filter(
                    lambda x: x.startswith(self.start_key_value_sample),
                    data_matrix.keys()
                )
            )
            
            this_output: Dict[str, Any] = {
                'scores': {
                    cluster_key[len(self.start_key_value_matrix):]: 0.0
                    for cluster_key in cluster_keys
                },
                'predicted_class': None
            }
            
            for cluster_key in cluster_keys:
                real_cluster_name: str = cluster_key[len(self.start_key_value_matrix):]
                current_sample_key: str = list(
                    filter(
                        lambda x: x.endswith(real_cluster_name),
                        sample_keys)
                )[0]

                current_cluster: np.ndarray = data_matrix[cluster_key]
                current_sample: np.ndarray = data_matrix[current_sample_key]
                
                if np.allclose(current_sample, 0, atol=self.epsilon):
                    current_sample = current_sample + np.random.normal(0, self.epsilon, current_sample.shape)
                
                if np.allclose(current_cluster, 0, atol=self.epsilon):
                    current_cluster = current_cluster + np.random.normal(0, self.epsilon, current_cluster.shape)

                if data_matrix.get("best_sigma"):
                    current_cluster = self.__gaussian_function__(
                        x=current_cluster,
                        sigma=data_matrix["best_sigma"]
                    )
                    current_sample = self.__gaussian_function__(
                        x=current_sample,
                        sigma=data_matrix["best_sigma"]
                    )

                score = self.__forward__(
                    current_cluster,
                    current_sample
                )
                                                
                this_output['scores'][real_cluster_name] = score

            if self.use_softmax:
                score_values: list = list(this_output['scores'].values())
                # compute reciprocal distances:
                similarity_scores = 1 / (np.array(score_values) + self.epsilon)

                softmax_scores: np.ndarray = self.__softmax__(np.array(similarity_scores))

                this_output['softmax'] = {
                    cluster_key[len(self.start_key_value_matrix):]: float(np.squeeze(softmax_value))
                    for cluster_key, softmax_value in zip(cluster_keys, softmax_scores)
                }

                this_output['predicted_class'] = cluster_keys[
                    np.argmax(softmax_scores)
                ].replace(self.start_key_value_matrix, '')

            this_output['predicted_class'] = this_output['predicted_class'] if this_output['predicted_class'] else cluster_keys[
                np.argmax(
                    np.argmin(
                        list(this_output['scores'].values())
                        )
                    )
                ].replace(self.start_key_value_matrix, '')
            
            output.append(this_output)
            
        return output

    def __call__(self, list_of_data: Union[List[Dict[str, np.ndarray]], Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:

        if not list_of_data:
            return []

        if isinstance(list_of_data, dict):
            list_of_data = [list_of_data]
            
        return self.forward(list_of_data)