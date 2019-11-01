import json
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def load_data(file_path):
	with open(file_path) as file:
	    json_data = json.load(file)
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
	vectorizer = CountVectorizer(ngram_range=ngram_range, tokenizer=tokenizer)
	result = vectorizer.fit_transform(data)
	# print(vectorizer.vocabulary_)
	return result

def tfidf(data):
	transformer = TfidfTransformer()
	return transformer.fit_transform(data)

nlp = spacy.load("en")
data, label = load_data('data.txt')
data = bag_of_words(data)
data = tfidf(data)
print(data.toarray().shape)