from psaw import PushshiftAPI

import datetime as dt
import json
from typing import Any, Dict, List, Tuple

from subreddit_labels import LEFT_LEANING_SUBS, RIGHT_LEANING_SUBS, RIGHT_TEST_SUBS, LEFT_TEST_SUBS


def get_data(
    subreddit_names: List[str],
    leaning: str,
    num: int,
    api: PushshiftAPI,
    start_epoch: int,
    end_epoch: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    subreddits = []

    for subreddit in subreddit_names:
        posts = []
        comments = []
        posts_res = list(api.search_submissions(
            after=start_epoch,
            before=end_epoch,
            subreddit=subreddit,
            filter=['url', 'title', 'score'],
            limit=num)
        )

        for post in posts_res:
            posts.append(post[-1])

        comments_res = list(api.search_comments(
            after=start_epoch,
            before=end_epoch,
            subreddit=subreddit,
            filter=['body', 'score'],
            limit=num)
        )

        for comment in comments_res:
            comments.append(comment[-1])

        subreddits.append((subreddit, {'leaning': leaning, 'posts': posts, 'comments': comments}))

    return subreddits


def main():
    api = PushshiftAPI()
    start_epoch = int(dt.datetime(2018, 1, 1).timestamp())
    end_epoch = int(dt.datetime(2019, 10, 1).timestamp())
    num = 30

    subs = get_data(LEFT_LEANING_SUBS, 'left', num, api, start_epoch, end_epoch)
    with open('data_left.txt', 'w') as outfile:
        json.dump(subs, outfile)

    subs = get_data(RIGHT_LEANING_SUBS, 'right', num, api, start_epoch, end_epoch)
    with open('data_right.txt', 'w') as outfile:
        json.dump(subs, outfile)

    subs = get_data(RIGHT_TEST_SUBS, 'right', num, api, start_epoch, end_epoch)
    with open('data_right_test.txt', 'w') as outfile:
        json.dump(subs, outfile)

    subs = get_data(LEFT_TEST_SUBS, 'left', num, api, start_epoch, end_epoch)
    with open('data_left_test.txt', 'w') as outfile:
        json.dump(subs, outfile)
