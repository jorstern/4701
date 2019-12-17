import pickle
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression


#import data_praw
import data_processing
import subreddit_labels


TEST_IN_PATHS = ['data_left_test.txt', 'data_right_test.txt']
TEST_OUT_PATH = 'test_preprocessed.pickle'

def classify(model, test_data):
    predictions = model.predict_proba(test_data['data'])
    print(predictions)

def main():
    data_processing.test(TEST_IN_PATHS, TEST_OUT_PATH)

    with open('test_preprocessed.pickle', 'rb') as data_file:
        test_data = pickle.load(data_file)

    print(test_data['data'].shape)
    with open('classifier_model.pickle', 'rb') as model_file:
        model = pickle.load(model_file)
    classify(model, test_data)

main()