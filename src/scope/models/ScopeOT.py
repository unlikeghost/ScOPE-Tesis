import ot
import numpy as np
from typing import Dict, Any

from scope.models.base_model import BaseModel


class ScOPEOT(BaseModel):

    def __init__(self, use_matching_method: bool = False) -> None:
        self._is_matching_method: bool = use_matching_method

        self.cost_matrix_function: callable = self.__calc_cost_ot_matrix if not use_matching_method else self.__calc_cost_matching_matrix

        self.epsilon: float = 1e-12

    @staticmethod
    def __calc_cost_ot_matrix(sample: np.ndarray, kw_samples: np.ndarray) -> np.ndarray:
        cost_matrix: np.ndarray = ot.dist(kw_samples, sample, metric='euclidean')

        return cost_matrix

    def __calc_cost_matching_matrix(self, sample: np.ndarray, kw_samples: np.ndarray) -> np.ndarray:
        cost_matrix: np.ndarray = np.zeros(shape=(len(kw_samples), 1))

        for index, kw_sample in enumerate(kw_samples):
            kw_sample: np.ndarray = kw_sample.reshape(1, -1)
            this_cost = self.__matching_score__(method='dice',
                                                x1=sample,
                                                x2=kw_sample)
            cost_matrix[index] = this_cost

        return cost_matrix

    @staticmethod
    def __wasserstein_distance__(sample_weights: np.ndarray, cluster_weights: np.ndarray,
                                 cost_matrix: np.ndarray) -> float:
        return ot.emd2(cluster_weights, sample_weights, cost_matrix)

    def forward(self, data_matrix: Dict[str, np.ndarray], softmax: bool = False) -> Dict[str, float]:
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

        # scores: Dict[str, float] = {
        #     cluster_key[len(self.start_key_value_matrix):]: 0.0
        #     for cluster_key in cluster_keys
        # }

        output: Dict[str, Any] = {
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

            if data_matrix.get("best_sigma"):
                current_cluster = self.__gaussian_function__(
                    x=current_cluster,
                    sigma=data_matrix["best_sigma"]
                )
                current_sample = self.__gaussian_function__(
                    x=current_sample,
                    sigma=data_matrix["best_sigma"]
                )

            cost_matrix: np.ndarray = self.cost_matrix_function(
                current_sample, current_cluster
            )

            current_cluster_weights: np.ndarray = np.ones(current_cluster.shape[0]) / current_cluster.shape[0]
            current_sample_weights: np.ndarray = np.ones(current_sample.shape[0]) / current_sample.shape[0]

            score: float = self.__wasserstein_distance__(
                cost_matrix=cost_matrix,
                cluster_weights=current_cluster_weights,
                sample_weights=current_sample_weights,
            )

            output['scores'][real_cluster_name] = score

        if softmax:
            score_values: list = list(output['scores'].values())
            # compute reciprocal distances:
            similarity_scores = 1 / (np.array(score_values) + self.epsilon)

            softmax_scores: np.ndarray = self.__softmax__(np.array(similarity_scores))

            output['softmax'] = {
                cluster_key[len(self.start_key_value_matrix):]: float(softmax_value)
                for cluster_key, softmax_value in zip(cluster_keys, softmax_scores)
            }

            output['predicted_class'] = cluster_keys[np.argmax(softmax_scores)].replace('ScOPEC_', '')

        output['predicted_class'] = output['predicted_class'] if output['predicted_class'] else cluster_keys[np.argmax(np.argmin(list(output['scores'].values())))].replace('ScOPEC_', '')

        return output


if __name__ == '__main__':
    from scope.compressors import ZStandardCompressor as Compressor
    from scope.matrix import MatrixFactory

    test_kw_samples: dict = {
        'class_0': ['Hola', 'Hola!', 'Holaa'],
        'class_1': ['Adios', 'Adios!', 'Adioss']
    }
    test_sample: str = 'Hola'
    matrix_factory: MatrixFactory = MatrixFactory(
        compressor_module=Compressor(),
        name_distance_function='ncd',
        str_separator=' '
    )
    matrix_result: Dict[str, np.ndarray] = matrix_factory(test_sample, test_kw_samples, get_best_sigma=True)

    model = ScOPEOT(use_matching_method=True)
    print(model.forward(matrix_result, softmax=True))
