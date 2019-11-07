import json
import pickle
from typing import List, Tuple

from scipy.sparse import csr_matrix
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


DEFAULT_IN_FILE_PATHS = ['data_left.txt', 'data_right.txt']
DEFAULT_OUT_FILE_PATH = 'data_preprocessed.pickle'

nlp = spacy.load("en")


def read_raw_data_from_files(file_paths: List[str]) -> List:
	json_data = []

	for file_path in file_paths:
		with open(file_path) as file:
			json_data += json.load(file)

	return json_data


def _load_post_instances(json_data: List) -> Tuple[List, List[int]]:
	data = []
	labels = []
	for subreddit in json_data:
		for post in subreddit[1]['posts']:
			data.append(post['title'])
			labels.append(0 if subreddit[1]['leaning'] == 'left' else 1)
		for comment in subreddit[1]['comments']:
			data.append(comment['body'])
			labels.append(0 if subreddit[1]['leaning'] == 'left' else 1)
	return data, labels


def _load_subreddit_instances(json_data: List) -> Tuple[List, List[int]]:
	data = []
	labels = []

	for subreddit in json_data:
		posts = [post['title'] for post in subreddit[1]['posts']]
		comments = [comment['body'] for comment in subreddit[1]['comments']]
		subreddit_text = " ".join(posts) + " " + " ".join(comments)

		data.append(subreddit_text)
		labels.append(0 if subreddit[1]['leaning'] == 'left' else 1)

	return data, labels


def _spacy_tokenizer(data: List, remove_punct=True, remove_stopwords=True, lemmatize=True) -> List:
	return [
		(token.lemma_ if lemmatize else token.orth_) for token in nlp(data)
		if (not token.is_punct if remove_punct else True)
		and not token.is_space
		and (not token.is_stop if remove_stopwords else True)
	]


def _bag_of_words(data, ngram_range=(1, 2), tokenizer=_spacy_tokenizer) -> Tuple:
	# by default, CountVectorizer will convert tokens to lower case.
	vectorizer = CountVectorizer(ngram_range=ngram_range, tokenizer=tokenizer)
	result = vectorizer.fit_transform(data)
	return result, vectorizer.get_feature_names()


def _tf_idf(data: csr_matrix) -> csr_matrix:
	transformer = TfidfTransformer()
	return transformer.fit_transform(data)


def load_data(file_paths: List[str] = None, instance_type: str = 'posts') \
		-> Tuple[csr_matrix, List[int], List[str]]:
	"""
	Reads raw data in JSON form from file_paths and preprocesses it.

	:param file_paths: Files to read the raw data from
	:param instance_type: 'posts' if each post and each comment should be a separate instance, or 'subs' if each sub
		should constitute one instance.
	:return: A tuple containing the list of preprocessed instances, the list of ground truth labels for their political
		leanings and a mapping from feature indices to their respective names.
	"""

	if file_paths is None:
		file_paths = DEFAULT_IN_FILE_PATHS
	raw_data = read_raw_data_from_files(file_paths)

	if instance_type == 'posts':
		data, labels = _load_post_instances(raw_data)
	elif instance_type == 'subs':
		data, labels = _load_subreddit_instances(raw_data)
	else:
		raise ValueError("type is neither 'posts' nor 'subs'")

	data, feature_names = _bag_of_words(data)
	data = _tf_idf(data)
	print(f"Preprocessed data has shape {data.get_shape()}")

	return data, labels, feature_names


def _store_preprocessed_data(
	data: csr_matrix, labels: List[int], feature_names: List[str],
	out_file_path: str = DEFAULT_OUT_FILE_PATH
):
	with open(out_file_path, 'wb') as out_file:
		stored_data = {
			'data': data,
			'labels': labels,
			'feature_names': feature_names,
		}
		pickle.dump(stored_data, out_file)


_store_preprocessed_data(*load_data(DEFAULT_IN_FILE_PATHS, 'subs'))
