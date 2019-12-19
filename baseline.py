def read_news_data(csv_file):
	data = []
	labels = []
	subreddits = []
	with open(csv_file, encoding='utf8') as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for row in readCSV:
			if len(row) == 6 and row[5] in pubs:
				data.append(row[4])
				labels.append(0 if pubs[row[5]] < 0 else 1)
				subreddits.append(row[5])
	return data, labels, subreddits


raw_data = read_raw_data_from_files(file_paths)

data, labels, scores, subreddits = _load_post_instances(raw_data)

news_data, news_labels, news_source = read_news_data(NEWS_CSV_FILE)
data += news_data
labels += news_labels
subreddits += news_source

data, feature_names = _bag_of_words(data, ngram_range=(1, 1))

data, labels, subreddits = _group_by_subreddits(data, subreddits, labels)
data = _tf_idf(data)

ground_truth = []
prediction = []
for i in range(len(subreddits)- 10):
	shortest_distance = math.inf
	closest_subreddit = None
	for j in range(len(subreddits) - 10, len(subreddits)):
		distance = sklearn.metrics.pairwise.pairwise_distances(data[i],data[j])
		if distance < shortest_distance:
			shortest_distance = distance
			closest_subreddit = subreddits[j]
	print("{} is matched to {}".format(subreddits[i], closest_subreddit))

	if subreddits[i] in RIGHT_LEANING_SUBS:
		ground_truth.append(1)
	else:
		ground_truth.append(0)
	if pubs[closest_subreddit] > 0:
		prediction.append(1)
	else:
		prediction.append(0)
	print(ground_truth)
	print(prediction)
