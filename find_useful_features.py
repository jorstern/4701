import pickle
from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression


def _calculate_feature_coefficients(data: csr_matrix, labels: List[int]) -> np.ndarray:
    classifier = LogisticRegression(solver='liblinear', max_iter=1000)
    classifier.fit(data, labels)
    return classifier.coef_[0]


def _find_largest_coefficient_indices(coefficients: np.ndarray, number_of_features: int) -> np.ndarray:
    # feature indices of the features with highest absolute coefficients, in descending order
    return np.abs(coefficients).argsort()[-number_of_features:][::-1]


def find_most_useful_feature_indices(preprocessed_data: csr_matrix, labels: List[int], number_of_features: int):
    """
    Determines which features are most useful for determining the political leaning of an instance by training a
    linear classifier. Returns the features that were assigned the highest (absolute) coefficients.

    :param preprocessed_data: A list of preprocessed instances
    :param labels: A list of integers indicating the political leaning for each instance
    :param number_of_features: The number of features to be returned
    :return: The indices of the number_of_features most useful features, in descending order of usefulness
    """

    coefficients = _calculate_feature_coefficients(preprocessed_data, labels)
    feature_indices = _find_largest_coefficient_indices(coefficients, number_of_features)
    return feature_indices


def _print_features(feature_indices: np.ndarray, coefficients: np.ndarray, feature_names: List[str]):
    feature_names += ['P(left-leaning | sub)', 'P(right-leaning | sub)']

    print('Coefficient \t\t Leaning \t Feature')
    for i in feature_indices:
        leaning = "Left" if coefficients[i] < 0 else "Right"
        print(f'{coefficients[i]} \t {leaning} \t\t {feature_names[i]}')


def _test():
    # read preprocessed data from disk
    with open('data_preprocessed.pickle', 'rb') as data_file:
        data_dict = pickle.load(data_file)

    coefficients = _calculate_feature_coefficients(data_dict['data'], data_dict['labels'])
    feature_indices = _find_largest_coefficient_indices(coefficients, 200)
    _print_features(feature_indices, coefficients, data_dict['feature_names'])


# _test()
