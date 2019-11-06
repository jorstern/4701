import json
import pickle
from typing import List

import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def load_data(file_paths: List[str]):
	# load raw data from disks
	json_data = []
	for file_path in file_paths:
		with open(file_path) as file:
			json_data += json.load(file)

	data = []
	label = []
	for subreddit in json_data:
		for post in subreddit[1]['posts']:
			data.append(post['title'])
			label.append(0 if subreddit[1]['leaning'] == 'left' else 1)
		for comment in subreddit[1]['comments']:
			data.append(comment['body'])
			label.append(0 if subreddit[1]['leaning'] == 'left' else 1)
	return data, label

def spacy_tokenizer(data, remove_punct=True, remove_stopwords=True, lemmatize=True):
	return [(token.lemma_ if lemmatize else token.orth_) for token in nlp(data) if (not token.is_punct if remove_punct else True) and not token.is_space and (not token.is_stop if remove_stopwords else True)]

def bag_of_words(data, ngram_range=(1,2), tokenizer=spacy_tokenizer):
	# by default, CountVectorizer will convert word to lower case.
	vectorizer = CountVectorizer(ngram_range=ngram_range, tokenizer=tokenizer)
	result = vectorizer.fit_transform(data)
	# print(vectorizer.vocabulary_)
	return result, vectorizer.get_feature_names()

def tfidf(data):
	transformer = TfidfTransformer()
	return transformer.fit_transform(data)

nlp = spacy.load("en")
data, label = load_data(['data_left.txt', 'data_right.txt'])
data, feature_names = bag_of_words(data)
data = tfidf(data)
print(f"Preprocessed data has shape {data.get_shape()}")


# write preprocessed data and labels to disk
with open('data_preprocessed.pickle', 'wb') as out_file:
	stored_data = {
		'data': data,
		'labels': label,
		'feature_names': feature_names,
	}
	pickle.dump(stored_data, out_file)
