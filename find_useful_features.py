import pickle
from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression


def calculate_feature_coefficients(data: csr_matrix, labels: List[int]) -> np.ndarray:
    classifier = LogisticRegression(solver='liblinear', max_iter=1000)
    classifier.fit(data, labels)
    return classifier.coef_[0]


def print_n_most_useful_features(coefficients: np.ndarray, feature_names: List[str], n: int):
    # feature indices of the n features with highest absolute coefficients, in descending order
    feature_indices = np.abs(coefficients).argsort()[-n:][::-1]

    print('Coefficient \t\t Leaning \t Feature')
    for i in feature_indices:
        leaning = "Left" if coefficients[i] < 0 else "Right"
        print(f'{coefficients[i]} \t {leaning} \t\t {feature_names[i]}')


def main():
    # read preprocessed data from disk
    with open('data_preprocessed.pickle', 'rb') as data_file:
        data_dict = pickle.load(data_file)

    coefficients = calculate_feature_coefficients(data_dict['data'], data_dict['labels'])
    print_n_most_useful_features(coefficients, data_dict['feature_names'], 50)


main()
