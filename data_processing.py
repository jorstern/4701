import json
import pickle
from typing import List, Tuple

from scipy.sparse import csr_matrix, vstack, hstack
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import math


DEFAULT_IN_FILE_PATHS = ['data_left.txt', 'data_right.txt']
DEFAULT_OUT_FILE_PATH = 'data_preprocessed.pickle'

nlp = spacy.load("en")


def read_raw_data_from_files(file_paths: List[str]) -> List:
	json_data = []

	for file_path in file_paths:
		with open(file_path) as file:
			json_data += json.load(file)

	return json_data


def _load_post_instances(json_data: List) -> Tuple:
	data = []
	labels = []
	scores = []
	subreddits = []
	for subreddit in json_data:
		for post in subreddit[1]['posts']:
			data.append(post['title'])
			labels.append(0 if subreddit[1]['leaning'] == 'left' else 1)
			scores.append(post['score'])
			subreddits.append(subreddit)
		for comment in subreddit[1]['comments']:
			data.append(comment['body'])
			labels.append(0 if subreddit[1]['leaning'] == 'left' else 1)
			scores.append(comment['score'])
			subreddits.append(subreddit)
	return data, labels, scores, subreddits


def _spacy_tokenizer(data: List, remove_punct=True, remove_stopwords=True, lemmatize=True) -> List:
	return [
		(token.lemma_ if lemmatize else token.orth_) for token in nlp(data)
		if (not token.is_punct if remove_punct else True)
		and not token.is_space
		and (not token.is_stop if remove_stopwords else True)
	]


def _bag_of_words(data, ngram_range=(1, 2), tokenizer=_spacy_tokenizer, is_training=True) -> Tuple:
	# by default, CountVectorizer will convert tokens to lower case.
	if not is_training:
		with open('vocabulary.pickle', 'rb') as file:
			VOCABULARY = pickle.load(file)
		vectorizer = CountVectorizer(ngram_range=ngram_range, tokenizer=tokenizer, vocabulary=VOCABULARY)
	else:
		vectorizer = CountVectorizer(ngram_range=ngram_range, tokenizer=tokenizer)

	result = vectorizer.fit_transform(data)
	return result, vectorizer.get_feature_names()


def _named_entity_tokenizer(data):
	return [
		token.orth_ for token in nlp(data)
		if not token.is_punct
		and not token.is_space
	]


def _bag_of_named_entity(data, entity_max_len=3):
	named_entity = set()
	types = {"PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}

	for item in data:
		doc = nlp(item)
		for ent in doc.ents:
			if ent.label_ in types and ent.text.lower() not in named_entity and len(ent.text.split()) <= entity_max_len:
				named_entity.add(ent.text.lower())
	vectorizer = CountVectorizer(tokenizer=_named_entity_tokenizer, vocabulary=named_entity)
	result = vectorizer.fit_transform(data)

	return result, vectorizer.get_feature_names()


def _group_by_subreddits(data, subreddits, labels, scores=None):
	# group the data from the same subreddit into one, weighted by the sqrt(scores).
	# e.g. data = [[1,1], [1,2], [2,1], [2,2]], subreddits = ['xx', 'xx', 'yy', 'yy'], scores = [1, 4, 9, 16]
	# Then the grouped_data will be [[1,1]*1 + [1,2]*sqrt(4), [2,1]*sqrt(9) + [2,2]*sqrt(16)] = [[3,5], [14,11]]
	last_index = 0

	grouped_data = None
	grouped_labels = []

	for i in range(len(subreddits) + 1):
		if i == len(subreddits) or subreddits[i] != subreddits[last_index]:
			multiplier = csr_matrix([0 if j < last_index or j >= i else (round(math.sqrt(scores[j])) if scores and scores[j] > 0 else 1) for j in range(len(subreddits))])
			subreddit_data = multiplier.dot(data)
			if grouped_data is None:
				grouped_data = subreddit_data
			else:
				grouped_data = vstack((grouped_data, subreddit_data))
			grouped_labels.append(labels[last_index])

			last_index = i  # update the last index
	return grouped_data, grouped_labels


def _tf_idf(data: csr_matrix) -> csr_matrix:
	transformer = TfidfTransformer()
	return transformer.fit_transform(data)


def load_data(file_paths: List[str] = None, instance_type: str = 'posts', named_entity=False, is_training=True) \
		-> Tuple[csr_matrix, List[int], List[str]]:
	"""
	Reads raw data in JSON form from file_paths and preprocesses it.

	:param file_paths: Files to read the raw data from
	:param instance_type: 'posts' if each post and each comment should be a separate instance, or 'subs' if each sub
		should constitute one instance.
	:param named_entity: True if named entity recognition should be used as feature
	:return: A tuple containing the list of preprocessed instances, the list of ground truth labels for their political
		leanings and a mapping from feature indices to their respective names.
	"""

	if file_paths is None:
		file_paths = DEFAULT_IN_FILE_PATHS
	raw_data = read_raw_data_from_files(file_paths)

	data, labels, scores, subreddits = _load_post_instances(raw_data)
	if named_entity:
		data, feature_names = _bag_of_named_entity(data)
	else:
		data, feature_names = _bag_of_words(data, is_training=is_training)
	if instance_type == 'subs':
		data, labels = _group_by_subreddits(data, subreddits, labels, scores)
	data = _tf_idf(data)
	print(f"Preprocessed data has shape {data.get_shape()}")

	return data, labels, feature_names


def classify_as_feature(data, labels):
	classifier = LogisticRegression(solver='liblinear', max_iter=1000)
	classifier.fit(data, labels)
	with open('classifier_model.pickle', 'wb') as outfile:
		pickle.dump(classifier, outfile)

	return classifier.predict_proba(data)


def _store_preprocessed_data(
	data: csr_matrix, labels: List[int], feature_names: List[str],
	out_file_path: str = DEFAULT_OUT_FILE_PATH, is_training=True):
	if is_training:
		with open('vocabulary.pickle', 'wb') as out_file:
			pickle.dump(feature_names, out_file)

		lr_results = classify_as_feature(data, labels)
		data = csr_matrix(hstack([data, lr_results]))

	print(data.get_shape())
	with open(out_file_path, 'wb') as out_file:
		stored_data = {
			'data': data,
			'labels': labels,
			'feature_names': feature_names,
		}
		pickle.dump(stored_data, out_file)

def main():
	# _store_preprocessed_data(*load_data(DEFAULT_IN_FILE_PATHS, 'subs', named_entity=True))
	_store_preprocessed_data(*load_data(DEFAULT_IN_FILE_PATHS, 'subs'))

def test(test_file_paths, test_out_path):
	_store_preprocessed_data(*load_data(test_file_paths, 'subs', is_training=False), out_file_path=test_out_path, is_training=False)

	main()

