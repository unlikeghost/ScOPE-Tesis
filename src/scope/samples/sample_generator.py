import numpy as np
from typing import Tuple, Dict


def generate_samples(data: np.ndarray, labels: np.ndarray, num_samples: int = 5) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generates samples of features for each class to be used in prediction or other operations.
    This function iterates over the dataset and for each instance, generates specified number
    of samples for each unique class from the dataset while ensuring that the original instance
    is not included in its own class's sample set.

    :param data: The dataset containing the feature values of the instances.
    :type data: np.ndarray
    :param labels: The corresponding class labels for the instances in the dataset.
    :type labels: np.ndarray
    :param num_samples: Number of samples to generate for each class for each instance.
        Must be less than the number of instances in the dataset and greater than zero.
    :type num_samples: int
    :return: A generator yielding tuples containing:
        - The current sample to predict.
        - Its corresponding expected label.
        - A dictionary of samples for each class.
    :rtype: Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]
    :raises ValueError: If `num_samples` is less than or equal to zero or larger than the
        number of instances in the dataset minus one.
    """

    if num_samples > len(data):
        raise ValueError('kw_samples must be less than the number of instances -1')

    if num_samples <= 0:
        raise ValueError('kw_samples must be greater than 0')

    unique_classes: np.ndarray = np.unique(labels)

    for index in range(len(data)):
        expected_label: np.ndarray = labels[index]
        sample_to_predict: np.ndarray = data[index]

        current_kw_samples: Dict[str, np.ndarray] = {
            f'class_{key}':  np.zeros(shape=(num_samples, data.shape[-1]))
            for key in unique_classes
        }

        for current_index_class in range(len(unique_classes)):
            mask: np.ndarray = np.where(labels == unique_classes[current_index_class])[0]
            while True:
                random_samples_index: np.ndarray = np.random.choice(mask, size=num_samples, replace=True)
                if index not in random_samples_index:
                    break
            # x_kw_samples: np.ndarray = data[random_samples_index]
            current_kw_samples[f'class_{unique_classes[current_index_class]}'] = data[random_samples_index]

        yield sample_to_predict, expected_label, current_kw_samples


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
        print(f'know samples: {kw_samples}')
        print('\n')
