import copy
import numpy as np
from typing import Union, Tuple, Dict, List

from scope.distances import CompressionDistance
from scope.compressors import BaseCompressor


class MatrixFactory:
    def __init__(self,
                 compressor_module: BaseCompressor,
                 name_distance_function: str,
                 str_separator: str = ' ') -> None:
        """
        Initializes the MatrixFactory instance with the compressor, distance function, and separator.

        Parameters:
        compressor_module: BaseCompressor
            Compression module used for compression operations.
        name_distance_function: str
            Name of the distance function to be used (e.g., NCD).
        str_separator: str, optional
            Separator for joining strings. Defaults to a single space ' '.
        """
        self.join_str_separator: str = str_separator
        self.name_distance_function: str = name_distance_function
        self.distance_module: CompressionDistance = CompressionDistance()
        self.compressor_module: BaseCompressor = compressor_module

    def get_best_sigma(self, sample: str, *kw_samples: str) -> float:
        """
        Computes the average sigma for the provided data.

        Parameters:
        sample: str
            Base sample for comparison.
        *kw_samples: str
            Additional samples.

        Returns:
        float
            The average sigma.
        """
        all_data = np.array([sample] + list(kw_samples), dtype=object)

        def compute_sigma(sequence: str) -> float:
            """Computes the sigma for a specific sequence."""
            _, _, compression_len = self.compress_data(sequence)
            repeated = self.join(sequence, sequence)
            _, _, double_compression_len = self.compress_data(repeated)

            return self.distance_module(
                distance=self.name_distance_function,
                x1=compression_len,
                x2=compression_len,
                x1x2=double_compression_len
            )

        # Vectorize the function to operate on arrays
        compute_sigma_vec = np.vectorize(compute_sigma)
        sigmas = compute_sigma_vec(all_data)

        return np.mean(sigmas)

    def join(self, x1: Union[str, np.ndarray], x2: Union[str, np.ndarray]) -> Union[str, np.ndarray]:
        """
        Joins two pieces of data of the same type (str or np.ndarray).

        Returns:
        str or np.ndarray
            The joined data.
        """
        if not isinstance(x1, type(x2)):
            raise ValueError("Both pieces of data must be of the same type")

        if isinstance(x1, str):
            return f"{self.join_str_separator}".join([x1, x2])
        elif isinstance(x1, np.ndarray):
            return np.concatenate([x1, x2])

        raise ValueError("Unsupported data type")

    def compress_data(self, x: Union[str, np.ndarray]) -> Tuple:
        """
        Compresses the given data.

        Parameters:
        x: str or np.ndarray
            Data to compress.

        Returns:
        Tuple
            Original data, compressed data, and compression length.
        """
        return self.compressor_module(x)

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

    def __call__(self, sample: Union[str, np.ndarray],
                 kw_samples: Union[Dict[str, Union[List[str], np.ndarray]], list],
                 get_best_sigma: bool = False) -> Dict[str, np.ndarray]:
        return self.build_matrix(sample, kw_samples, get_best_sigma=get_best_sigma)


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
