import numpy as np
from typing import Tuple


def generate_samples(data: np.ndarray, labels: np.ndarray, num_samples: int = 5) -> Tuple[np.ndarray]:
    """
    Generates samples from the provided dataset with the specified number of samples for
    each unique class. The function ensures that the generated samples do not include the
    index of the current sample being processed. It is intended for use in scenarios
    such as k-fold sampling or instance-based analysis.

    :param data: The dataset to sample from. Each row represents an individual data point.
    :type data: np.ndarray
    :param labels: The corresponding class labels for the dataset. Must have the same length
        as the number of rows in the data.
    :type labels: np.ndarray
    :param num_samples: The number of samples to generate for each instance from the dataset.
        Must be a positive integer less than the number of data points in the dataset.
    :type num_samples: int
    :return: A generator yielding a tuple containing the sample to predict, expected label
        for the current sample, and the generated samples of other instances within each
        unique class.
    :rtype: Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]
    :raises ValueError: If num_samples is less than or equal to zero.
    :raises ValueError: If num_samples is greater than or equal to the number of data points
        in the dataset.
    """

    if num_samples > len(data):
        raise ValueError('kw_samples must be less than the number of instances -1')

    if num_samples <= 0:
        raise ValueError('kw_samples must be greater than 0')

    unique_classes: np.ndarray = np.unique(labels)

    for index in range(len(data)):
        expected_label: np.ndarray = labels[index]
        sample_to_predict: np.ndarray = data[index]

        x_kw_samples: np.ndarray = np.zeros(
            shape=(num_samples, data.shape[-1])
        )

        for current_index_class in range(len(unique_classes)):
            mask: np.ndarray = np.where(labels == unique_classes[current_index_class])[0]
            while True:
                random_samples_index: np.ndarray = np.random.choice(mask, size=num_samples, replace=True)
                if index not in random_samples_index:
                    break
            x_kw_samples: np.ndarray = data[random_samples_index]

        yield sample_to_predict, expected_label, x_kw_samples


if __name__ == '__main__':
    x_test: np.ndarray = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    )

    y_test: np.ndarray = np.array(
        [1, 1, 1, 2, 2]
    )

    for test_x, test_y, kw_samples in generate_samples(x_test, y_test, num_samples=1):
        print(f'Test instance: {test_x}')
        print(f'Target: {test_y}')
        print(f'Know samples: {kw_samples}')
        print('\n')
