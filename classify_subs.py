import pickle
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import data_praw
import data_processing
import subreddit_labels


GROUND_TRUTH = [0]*len(subreddit_labels.LEFT_TEST_SUBS) + [1]*len(subreddit_labels.RIGHT_TEST_SUBS)
TEST_IN_PATHS = ['data_left_test.txt', 'data_right_test.txt']
TEST_OUT_PATH = 'test_preprocessed.pickle'

def classify(test_data, training_data, training_labels):
    classifier = LogisticRegression(solver='liblinear', max_iter=1000)
    classifier.fit(training_data, training_labels)
    print("Trained model!")
    predictions = classifier.predict(test_data)
    pred_nums = classifier.predict_proba(test_data)
    pred_results = []
    for r in pred_nums:
        if r[0] > r[1]:
            pred_results.append(str(r[0]/r[1]) + "x more likely to be left")
        else:
            pred_results.append(str(r[1]/r[0]) + "x more likely to be right")
    labels = subreddit_labels.LEFT_TEST_SUBS + subreddit_labels.RIGHT_TEST_SUBS
    res = zip(labels, pred_results)
    for r in res:
        print(r)
    return predictions

def remove_extra_features(training_data):
    size = training_data['data'].shape[1]
    extra_features = [size-2, size-1]
    idx_to_drop = np.unique(extra_features)
    C = training_data['data'].tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()


def main():
    data_praw.main()
    print("Downloaded data!")
    data_processing.main()
    print("Extracted features from training!")
    data_processing.test(TEST_IN_PATHS, TEST_OUT_PATH)
    print("Extracted features from test!")
    with open('data_preprocessed.pickle', 'rb') as data_file:
        training_data = pickle.load(data_file)
    with open('test_preprocessed.pickle', 'rb') as data_file:
        test_data = pickle.load(data_file)
    with open('classifier_model.pickle', 'rb') as model_file:
        model = pickle.load(model_file)
    training_labels = training_data['labels']
    new_training_data = remove_extra_features(training_data)
    results = classify(test_data['data'], new_training_data, training_labels)
    score = 0
    for idx, res in enumerate(results):
        if res == GROUND_TRUTH[idx]:
            score += 1
    print("Accuracy: ", score/len(GROUND_TRUTH))

main()