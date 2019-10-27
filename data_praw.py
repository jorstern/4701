import praw
reddit = praw.Reddit(client_id='gl9u8v7CW8bV7w', client_secret="7U-0BRdmb0GE46kZl2uR6i5YGqg",
                     password='4701project', user_agent='4701project by 4701project',
                     username='4701project')

subreddit = reddit.subreddit('conservative')

print(subreddit.display_name)  # Output: redditdev
print(subreddit.title)         # Output: reddit Development
print(subreddit.description)   # Output: A subreddit for discussion of ..

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
