# -*- coding: utf-8 -*-
"""
matching_methods.py

This module provides a class `MatchingMethods` that implements various similarity/distance
calculation methods for numerical arrays (or lists/tuples) using mathematical concepts.

Features:
- `matching`: Computes the sum of minimum values between two arrays.
- `jaccard`: Calculates the Jaccard distance between two arrays.
- `dice`: Computes the Dice distance metric for the arrays.
- `overlap`: Measures the overlap distance based on array values.
- Supports callable usage to dynamically select a method (e.g., 'matching', 'jaccard', etc.).

Classes:
- MatchingMethods: Contains static and class methods for the above calculations and
  supports callable operations to execute a specific method based on user input.

Example Usage:
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = np.array([1])

    mm = MatchingMethods()
    result = mm('dice', x1, x2)  # Calculate the Dice distance.
    print(result)

Encoding: UTF-8
Author: Jesus Alan Hernandez Galvan
Date: 01/09/2025
"""

import numpy as np
from typing import Dict, Union


class MatchingMethods:

    @staticmethod
    def matching(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.sum(np.minimum(x1, x2), dtype=np.float32)

    @classmethod
    def jaccard(cls, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        _matching = cls.matching(x1, x2)
        return 1.0 - (_matching / np.sum(np.maximum(x1, x2)))

    @classmethod
    def dice(cls, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        _matching = 2 * cls.matching(x1, x2)
        return 1.0 - (_matching / (np.sum(x1) + np.sum(x2)))

    @classmethod
    def overlap(cls, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        _matching = cls.matching(x1, x2)
        return 1.0 - (_matching / np.sum(x1) + np.sum(x2))

    def __call__(self, method: str, x1: Union[np.ndarray, list, tuple], x2: Union[np.ndarray, list, tuple]):

        if isinstance(x1, (list, tuple)):
            x1 = np.array(x1)
        if isinstance(x2, (list, tuple)):
            x2 = np.array(x2)

        allowed_methods: Dict[str, object] = {
            'matching': self.matching(x1, x2),
            'jaccard': self.jaccard(x1, x2),
            'dice': self.dice(x1, x2),
            'overlap': self.overlap(x1, x2)
        }

        if method not in allowed_methods:
            raise ValueError(f'Method "{method}" is not allowed')

        return allowed_methods[method]


if __name__ == '__main__':
    x1_test: np.ndarray = np.array([[1, 2, 3, 4, 5],
                                    [1, 2, 3, 4, 5]
                                    ])
    x2_test: np.ndarray = np.array([[1, 1, 1, 1, 1]])

    mm_str = MatchingMethods()
    print(mm_str('jaccard', x1_test, x2_test))
