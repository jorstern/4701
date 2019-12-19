import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import pickle
from typing import List

from find_useful_features import find_most_useful_feature_indices
from data_processing import DEFAULT_IN_FILE_PATHS as DEFAULT_RAW_DATA_FILE_NAMES, read_raw_data_from_files


def get_subreddit_names() -> List[str]:
    raw_data = read_raw_data_from_files(DEFAULT_RAW_DATA_FILE_NAMES)
    return [sub[0] for sub in raw_data]


def plot_pca(data, labels):
    pca = PCA(n_components=2)
    pca_coordinates = pca.fit_transform(data.toarray())

    plt.figure(figsize=(8, 3))

    for i in range(len(labels)):
        color = 'b' if labels[i] == 0 else 'r'
        plt.scatter(x=pca_coordinates[i, 0], y=pca_coordinates[i, 1], s=50, c=color)

    # for name, a, b in zip(get_subreddit_names(), pca_coordinates[:, 0], pca_coordinates[:, 1]):
    #     plt.annotate(name, (a, b))

    plt.show()


def _main():
    # read preprocessed data from disk
    with open('data_preprocessed.pickle', 'rb') as data_file:
        data_dict = pickle.load(data_file)

    # remove all but the most important features
    feature_indices = find_most_useful_feature_indices(data_dict['data'], data_dict['labels'], 200)
    data = data_dict['data'][:, feature_indices]

    # PCA before feature selection
    # plot_pca(data_dict['data'], data_dict['labels'])
    # PCA after feature selection
    plot_pca(data, data_dict['labels'])


_main()
