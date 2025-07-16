# -*- coding: utf-8 -*-
"""
compression_distance.py

This module provides the `CompressionDistance` class, which implements methods for calculating
various normalized compression-based distances used in data analysis and comparison tasks.

Features:
- `ncd`: Normalized Compression Distance (NCD), compares compressibility of two inputs.
- `cdm`: Compression Distance Metric (CDM), measures shared information between inputs.
- `clm`: Compression Length Measure (CLM), focuses on the compressed size relative to individual inputs.
- Supports callable usage to dynamically select and compute a distance metric.

Classes:
- CompressionDistance: A class containing static methods for compression-based distance
  calculations and a callable interface for easier metric selection.

Example Usage:
    compression_distance = CompressionDistance()
    result = compression_distance(distance='ncd', x1=10.2, x2=12.5, x1x2=18.7)
    print(result)

Supported Distance Metrics:
- ncd: Normalized Compression Distance
- cdm: Compression Distance Metric
- clm: Compression Length Measure

Encoding: UTF-8
Author: Jesus Alan Hernandez Galvan
Date: 01/09/2025
"""


class CompressionDistance:

    @staticmethod
    def ncd(x1: float, x2: float, x1x2: float) -> float:
        """
            Computes the Normalized Compression Distance (NCD) between two entities.

            This method calculates a similarity measure between two data entities
            based on their compression sizes. It uses compressed sizes of the
            individual entities and their concatenated compression size to compute
            the ratio.

            Args:
                x1: float
                    The compressed size of the first entity.
                x2: float
                    The compressed size of the second entity.
                x1x2: float
                    The compressed size of the concatenation of the two entities.

            Returns:
                float
                    The calculated normalized compression distance.

            Raises:
                None
        """
        denominator: float = max(x1, x2)
        numerator: float = x1x2 - min(x1, x2)
        return numerator / denominator

    @staticmethod
    def cdm(x1: float, x2: float, x1x2: float) -> float:
        """
            Calculate the dependency measure (cdm) for given probabilities.

            This method computes a statistical measure often used in entropy or
            probability-based calculations to evaluate the dependency or
            relationship between two events.

            Args:
                x1: The probability or measure associated with the first event.
                x2: The probability or measure associated with the second event.
                x1x2: The joint probability or measure of the intersection of
                    both events.

            Returns:
                The result of the cdm calculation as a floating-point number.

            Raises:
                ZeroDivisionError: If the sum of x1 and x2 (denominator) is 0.
        """
        denominator: float = x1 + x2
        numerator: float = x1x2
        return numerator / denominator

    @staticmethod
    def clm(x1: float, x2: float, x1x2: float) -> float:
        """
            Computes a specific metric using the given inputs.

            This method calculates a ratio based on the inputs `x1`, `x2`, and `x1x2`. It returns
            the computed value as a floating-point number. The inputs are expected to represent probabilities
            or similar values in the interval of [0, 1]. The formula ensures that the calculation adheres
            to this probabilistic interpretation.

            Args:
                x1: A floating-point number representing the first input probability value.
                x2: A floating-point number representing the second input probability value.
                x1x2: A floating-point number representing the joint probability of `x1` and `x2`.

            Returns:
                A float value that is the result of the computed ratio.
        """
        denominator: float = x1x2
        numerator: float = 1 - (x1 + x2 - x1x2)
        return numerator / denominator
    
    @staticmethod
    def mse(x1: float, x2: float, x1x2: float) -> float:
        """
        Computes the Mean Squared Error (MSE) between two values.

        This method calculates the squared difference between two numerical values
        and returns the average of these squared differences. It is often used to
        measure the accuracy of predictions in regression tasks.

        Args:
            x1: float
                The first value for comparison.
            x2: float
                The second value for comparison.
            x1x2: float
                The concatenated value of x1 and x2.

        Returns:
            float
                The computed Mean Squared Error value.
        """
        return (x1 - x2) ** 2

    def __call__(self, distance: str, **kwargs) -> float:
        """
        Callable object that calculates the similarity or distance measure based
        on the provided method name. Supports multiple methods defined within
        the object. The desired method is selected using a string identifier
        and additional parameters specific to the method can be passed as keyword
        arguments.

        Parameters
        ----------
        distance: str
            A string identifier representing the method to be used for
            calculation. Must be one of the keys in the `allowed_methods`
            dictionary.
        **kwargs
            Additional keyword arguments passed to the selected method
            during execution.

        Raises
        ------
        ValueError
            If the provided `distance` method name is not listed in the
            `allowed_methods` dictionary.

        Returns
        -------
        float
            The result of the calculation performed by the selected method.
        """
        allowed_methods: dict = {
            'ncd': self.ncd,
            'cdm': self.cdm,
            'clm': self.clm,
            'mse': self.mse
        }

        if distance not in allowed_methods:
            raise ValueError(f'Method "{distance}" is not allowed')

        return allowed_methods[distance](**kwargs)

if __name__ == '__main__':
    from scope.compressors import LZ77Compressor as Compressor

    compressor = Compressor()
    _, _, test_x1 = compressor("Hola")
    _, _, test_x2 = compressor("Hola")
    _, _, test_x1x2 = compressor("Hola Hola")

    compression_distance = CompressionDistance()
    result = compression_distance(distance="ncd", x1=test_x1, x2=test_x2, x1x2=test_x1x2)
    print(result)
