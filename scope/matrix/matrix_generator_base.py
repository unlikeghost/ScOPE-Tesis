import numpy as np
from typing import Union, Tuple, Any

from scope.distances import CompressionDistance
from scope.compressors import BaseCompressor


class MatrixFactoryBase:
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

    def calc_compression_matrix(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def build_matrix(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        return self.build_matrix(*args, **kwargs)
