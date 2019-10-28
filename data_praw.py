import praw
# reddit = praw.Reddit(client_id='gl9u8v7CW8bV7w', client_secret="7U-0BRdmb0GE46kZl2uR6i5YGqg",
#                      password='4701project', user_agent='4701project by 4701project',
#                      username='4701project')

# subreddit = reddit.subreddit('conservative')
# submissions = []
# for submission in reddit.subreddit('politics').top('all', limit=2000):
#     submissions.append(submission)

# print(len(submissions))

# print(subreddit.display_name)  # Output: redditdev
# print(subreddit.title)         # Output: reddit Development
# print(subreddit.description)   # Output: A subreddit for discussion of ..

# import requests
# base_url = 'https://www.reddit.com/'
# data = {'grant_type': 'password', 'username': '4701project', 'password': '4701project'}
# auth = requests.auth.HTTPBasicAuth('gl9u8v7CW8bV7w', '7U-0BRdmb0GE46kZl2uR6i5YGqg')
# r = requests.post(base_url + 'api/v1/access_token',
#                   data=data,
#                   headers={'user-agent': '4701project by 4701project'},
#           auth=auth)
# d = r.json()

# token = 'bearer ' + d['access_token']

# base_url = 'https://oauth.reddit.com'
# headers = {'Authorization': token, 'User-Agent': '4701project by 4701project'}
# response = requests.get(base_url + '/api/v1/me', headers=headers)

# if response.status_code == 200:
#     print(response.json()['name'], response.json()['comment_karma'])
def get_data(subs, leaning, num):
    for subreddit in subs:
        posts = []
        comments = []
        posts_res = list(api.search_submissions(after=start_epoch,
                                    subreddit=subreddit,
                                    filter=['url', 'title', 'score'],
                                    limit=num))

        for post in posts_res:
            posts.append(post[-1])
        comments_res = list(api.search_comments(after=start_epoch,
                                    subreddit=subreddit,
                                    filter=['body','score'],
                                    limit=num))
        for comment in comments_res:
            comments.append(comment[-1])
        subreddits.append((subreddit, {'leaning': leaning, 'posts': posts, 'comments': comments}))


from psaw import PushshiftAPI
import json
import datetime as dt

api = PushshiftAPI()

start_epoch=int(dt.datetime(2017, 1, 1).timestamp())

left_subreddits = [
'anarchocommunism',
'antifascistsofreddit',
'centerleftpolitics',
'democrats',
'democraticsocialism',
'elizabethwarren',
'greenparty',
'progressive',
'socialism',
'toiletpaperusa'
]

right_subreddits = [
'askaconservative',
'conservative',
'cringeanarchy',
'republican',
'shitpoliticssays',
'the_donald',
'prolife',
'progun',
'rightwinglgbt',
'libertarian'
]

subreddits = []

# get_data(left_subreddits, 'left', 10)
# get_data(right_subreddits, 'right', 10)
# get_data(['conservative'], 'right', 1)
# print(subreddits)
with open('data.txt', 'w') as outfile:
    json.dump(subreddits, outfile)