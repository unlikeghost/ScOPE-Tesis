import copy
import numpy as np
from typing import Union, Tuple, Dict, List

from scope.matrix.matrix_generator_base import MatrixFactoryBase


class MatrixFactory(MatrixFactoryBase):

    def calc_compression_matrix(self, samples: list) -> np.ndarray:
        """
        Computes the compression distance matrix.

        Parameters:
        samples: list
            List of samples (strings or np.ndarray).

        Returns:
        np.ndarray
            Symmetric matrix containing the distances.
        """
        sample_count = len(samples)

        # Precompute compression lengths for all samples
        compression_lengths = np.array([
            self.compress_data(sample)[2] for sample in samples
        ])

        # Create an empty matrix for the distances
        distance_matrix = np.zeros((sample_count, sample_count), dtype=np.float32)

        # Compute distances for each pair
        for i in range(sample_count):
            for j in range(i, sample_count):
                joined_sample = self.join(samples[i], samples[j])
                _, _, joined_len = self.compress_data(joined_sample)

                distance = self.distance_module(
                    distance=self.name_distance_function,
                    x1=compression_lengths[i],
                    x2=compression_lengths[j],
                    x1x2=joined_len
                )
                distance_matrix[i, j] = distance_matrix[j, i] = distance

        return distance_matrix

    def build_matrix(self,
                     sample: Union[str, np.ndarray],
                     kw_samples: Union[
                         Dict[str, Union[str, np.ndarray]],
                         Tuple[list]
                     ],
                     get_best_sigma: bool = False) -> Dict[str, np.ndarray]:
        """
        Builds compression matrices for the provided samples.

        Parameters:
        sample : str or np.ndarray
            Main sample to consider.
        kw_samples : dict or tuple
            Samples grouped into clusters.
        get_best_sigma : bool
            If True, computes the best sigma.

        Returns:
        dict
            Dictionary containing compression matrices and optionally the best sigma.
        """
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

            # Generate matrices
            compression_matrix = self.calc_compression_matrix(cluster_samples_list)
            results[f"ScOPEC_{cluster}"] = compression_matrix[:-1, :]
            results[f"ScOPES_{cluster}"] = compression_matrix[-1:, :]

            if get_best_sigma:
                sigma = self.get_best_sigma(sample, *cluster_samples_list[:-1])
                all_sigmas.append(sigma)

        if get_best_sigma:
            results["best_sigma"] = np.mean(all_sigmas)

        return results


if __name__ == "__main__":
    from scope.compressors import GZIPCompressor

    test_samples = {
        'class_0': ['Hola', 'Adios', 'Buenos dias'],
        'class_1': ['Hello', 'Goodbye', 'Good morning']
    }
    test_sample = 'Hello'
    factory = MatrixFactory(
        compressor_module=GZIPCompressor(),
        name_distance_function='ncd',
        str_separator='-'
    )
    result = factory(test_sample, test_samples, get_best_sigma=True)
    print(result)
