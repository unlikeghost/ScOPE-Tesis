import copy
import numpy as np
from typing import Union, Tuple, Dict, List, Any

from scope.matrix.matrix_generator_base import MatrixFactoryBase


class MatrixFactoryV2(MatrixFactoryBase):

    def calc_compression_matrix(self, samples: list) -> np.ndarray:

        # Precompute compression lengths for all samples
        compression_lengths = np.array([
            self.compress_data(sample)[2]
            for sample in samples
        ])

        compression_matrix: np.ndarray = np.zeros(
            shape=(
                len(samples),
                1,
                len(samples) - 1
            ),
            dtype=np.float32
        )

        for index_i, sample in enumerate(samples):

            x1_len: np.ndarray = compression_lengths[index_i]

            # keep samples that are different from current index
            current_samples_index: list = list(
                filter(lambda values: values[0] != index_i, enumerate(samples))
            )
            current_samples: list = [item for _, item in current_samples_index]

            for index_j, current_sample in enumerate(current_samples):

                _, _, x2_len = self.compress_data(current_sample)

                joined_sample = self.join(sample, current_sample)

                _, _, joined_len = self.compress_data(joined_sample)

                compression_metric: float = self.distance_module(
                    distance=self.name_distance_function,
                    x1=x1_len,
                    x2=x2_len,
                    x1x2=joined_len
                )

                compression_matrix[index_i, 0, index_j] = compression_metric

        return compression_matrix

    def build_matrix(self,
                     sample: Union[str, np.ndarray],
                     kw_samples: Union[
                         Dict[str, Union[str, np.ndarray]],
                         Tuple[list]
                     ],
                     get_best_sigma: bool = False) -> Dict[str, np.ndarray]:

        if isinstance(kw_samples, dict):
            cluster_samples = copy.deepcopy(kw_samples)
        else:
            cluster_samples = {
                index: value for index, value in enumerate(kw_samples)
            }

        results = {}

        all_sigmas = []

        for cluster, cluster_samples_list in cluster_samples.items():
            if isinstance(cluster_samples_list, list):
                cluster_samples_list += [sample]
            elif isinstance(cluster_samples_list, np.ndarray):
                cluster_samples_list = np.append(cluster_samples_list, [sample])

            compression_matrix = self.calc_compression_matrix(cluster_samples_list)
            results[f"ScOPEC_{cluster}"] = compression_matrix[0:-1, :, :].transpose(1, 0, 2).squeeze()
            results[f"ScOPES_{cluster}"] = compression_matrix[-1, :, :]

            if get_best_sigma:
                sigma = self.get_best_sigma(sample, *cluster_samples_list[:-1])
                all_sigmas.append(sigma)

        if get_best_sigma:
            results["best_sigma"] = np.mean(all_sigmas)

        return results


if __name__ == "__main__":
    from scope.compressors import BZ2Compressor as Compressor

    test_kw_samples: dict = {
        'class_0': ['Holi', 'Hola!', 'Holaa'],
        'class_1': ['Adios', 'Adios!', 'Adioss']
    }
    test_sample: str = 'Hola'

    factory = MatrixFactoryV2(
        compressor_module=Compressor(),
        name_distance_function='ncd',
        str_separator='-'
    )
    result = factory(test_sample, test_kw_samples, get_best_sigma=True)
    print(result)
