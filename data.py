import requests
import datetime

start = int(datetime.datetime(2019,8,11,0,0).timestamp())
end = int(datetime.datetime(2019,10,11,0,0).timestamp())
comment_per_subreddit = 1000
posts_per_subreddit = 1000

subreddits = ['conservative']
comments = {}
posts = {}

for subreddit in subreddits:
	comments[subreddit] = []
	posts[subreddit] = []
	r = requests.get('https://api.pushshift.io/reddit/search/comment/?subreddit={}&sort=desc&sort_type=created_utc&after={}&before={}&size={}'.format(subreddit, start, end, comment_per_subreddit))
	for data in r.json()['data']:
		comments[subreddit].append(data['body'])

	r = requests.get('https://api.pushshift.io/reddit/search/submission/?subreddit={}&sort=desc&sort_type=created_utc&after={}&before={}&size=1000'.format(subreddit, start, end, posts_per_subreddit))
	for data in r.json()['data']:
		posts[subreddit].append(data['title'])
