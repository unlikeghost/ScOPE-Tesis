# -*- coding: utf-8 -*-

import numpy as np
from enum import Enum
from typing import Protocol, Union


class MatchingType(Enum):
    MATCHING = "matching"
    JACCARD = "jaccard"
    DICE = "dice"
    OVERLAP = "overlap"


class MatchingStrategy(Protocol):
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        ...


class _MatchingBase:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _matching(x1: np.ndarray, x2: np.ndarray) -> float:
        return float(np.sum(np.minimum(x1, x2), dtype=np.float32))


class MatchingMetric(_MatchingBase):
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return self._matching(x1, x2)


class JaccardMetric(_MatchingBase):
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        m = self._matching(x1, x2)
        return 1.0 - (m / float(np.sum(np.maximum(x1, x2))))


class DiceMetric(_MatchingBase):
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        m = 2 * self._matching(x1, x2)
        return 1.0 - (m / float(np.sum(x1) + np.sum(x2)))


class OverlapMetric(_MatchingBase):
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        m = self._matching(x1, x2)
        return 1.0 - (m / float(np.sum(x1) + np.sum(x2)))


MATCHING_STRATEGIES: dict[MatchingType, type[MatchingStrategy]] = {
    MatchingType.MATCHING: MatchingMetric,
    MatchingType.JACCARD: JaccardMetric,
    MatchingType.DICE: DiceMetric,
    MatchingType.OVERLAP: OverlapMetric,
}


def get_matching_metric(name: Union[str, MatchingType]) -> MatchingStrategy:
    if isinstance(name, str):
        try:
            name_enum = MatchingType(name.lower())
        except ValueError:
            allowed = sorted(t.value for t in MatchingType)
            raise ValueError(
                f"'{name}' is not a valid matching metric. "
                f"Expected one of: {', '.join(allowed)}"
            )
    elif isinstance(name, MatchingType):
        name_enum = name
    else:
        raise TypeError("Expected 'name' to be str or MatchingType.")

    return MATCHING_STRATEGIES[name_enum]()


# Test case
if __name__ == '__main__':
    x1_test = np.array([[1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5]])
    x2_test = np.array([[1, 1, 1, 1, 1]])

    metric = get_matching_metric("jaccard")
    print("Jaccard:", metric.compute(x1_test, x2_test))

    metric = get_matching_metric("dice")
    print("Dice:", metric.compute(x1_test, x2_test))

    metric = get_matching_metric("overlap")
    print("Overlap:", metric.compute(x1_test, x2_test))

    metric = get_matching_metric("matching")
    print("Matching:", metric.compute(x1_test, x2_test))
