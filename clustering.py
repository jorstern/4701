import numpy as np
from sklearn.cluster import SpectralClustering

import pickle
from typing import List

from data_processing import DEFAULT_IN_FILE_PATHS as DEFAULT_RAW_DATA_FILE_NAMES, read_raw_data_from_files
from find_useful_features import find_most_useful_feature_indices
from evaluation import print_metrics


NUMBER_OF_CLUSTERS = 5


def get_subreddit_names() -> List[str]:
    # TODO there is probably a better way of doing this (maybe store the subreddit names with the preprocessed data)
    raw_data = read_raw_data_from_files(DEFAULT_RAW_DATA_FILE_NAMES)
    return [sub[0] for sub in raw_data]


def _main():
    # read preprocessed data from disk
    with open('data_preprocessed.pickle', 'rb') as data_file:
        data_dict = pickle.load(data_file)

    # remove all but the most important features
    feature_indices = find_most_useful_feature_indices(data_dict['data'], data_dict['labels'], 200)
    data = data_dict['data'][:, feature_indices]
    print(data.shape)

    clustering = SpectralClustering(n_clusters=NUMBER_OF_CLUSTERS, n_components=NUMBER_OF_CLUSTERS)
    # for instance i, cluster_indices[i] is the index of the cluster the instance is in
    cluster_indices = clustering.fit_predict(data)

    # restructure the cluster information so that clusters[i] contains the indices of all instances in cluster i
    clusters = []
    for i in range(NUMBER_OF_CLUSTERS + 5):
        clusters.append(np.argwhere(cluster_indices == i).flatten())

    # replace indices with respective subreddit names
    subreddit_names = get_subreddit_names()
    clusters_with_names = [[subreddit_names[sub] for sub in cluster] for cluster in clusters]

    print("Clusters:")
    for i, cluster in enumerate(clusters_with_names):
        print(f"{i}: {', '.join(cluster)}")

    # evaluate clustering
    print_metrics(cluster_indices.tolist(), data_dict['labels'])


_main()
