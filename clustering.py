from find_useful_features import find_most_useful_feature_indices

import pickle

from sklearn.cluster import KMeans


def _main():
    # read preprocessed data from disk
    with open('data_preprocessed.pickle', 'rb') as data_file:
        data_dict = pickle.load(data_file)

    # remove all but the most important features
    feature_indices = find_most_useful_feature_indices(data_dict['data'], data_dict['labels'], 200)
    data = data_dict['data'][:, feature_indices]
    print(data.shape)

    k_means = KMeans(n_clusters=4, max_iter=1000)
    cluster_indices = k_means.fit_predict(data)
    # TODO somehow show the results


_main()
