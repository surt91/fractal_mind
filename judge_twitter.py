import os
import sys
import urllib.request

import tweepy

from keys_and_secrets import keys_and_secrets
from classify import Classifier

auth = tweepy.OAuthHandler(keys_and_secrets["consumer_key"],
                           keys_and_secrets["consumer_secret"])
auth.set_access_token(keys_and_secrets["access_token_key"],
                      keys_and_secrets["access_token_secret"])

# wait if we hit twitters rate limit (15 requests in 15 minutes)
# this way all tweets will be accepted and we have a rudimentary DOS protection
# if this bot is too successful. Twitter itself will protect us from malicious DOS
api = tweepy.API(auth, wait_on_rate_limit=True)


def tweet_pic(path, text=None, reply_to=None):
    api.update_with_media(path, text, in_reply_to_status_id=reply_to)


def obtain_tweets_from(screen_name):
    try:
        with open("last_id.dat", "r") as f:
            last_id = int(f.read().strip())
    except:
        last_id = 1

    print(last_id)

    todo = []
    tweets = api.user_timeline(screen_name, since_id=last_id)
    for i in tweets:
        if i.text[:3] == "RT ":
            # this is a retweet, ignore it
            print("ignored retweet: @" + i.user.screen_name, ":", i.text)
            continue
        print("@" + i.user.screen_name, ":", i.text)
        last_id = max(i.id, last_id)
        todo.append(i)

    with open("last_id.dat", "w") as f:
        f.write(str(last_id))

    return todo


def get_my_handle():
    myself = api.me()
    return myself.screen_name


my_handle = get_my_handle()

print("constructing global classifier")
classifier = Classifier()


class MyStreamListener(tweepy.StreamListener):
    def __init__(self, action, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = action
        try:
            with open("last_id.dat", "r") as f:
                self.last_id = int(f.read().strip())
        except:
            self.last_id = 0

    def on_status(self, status):
        print(status.text)
        print("@" + status.user.screen_name, ":", status.text)
        self.action(status)

    def on_error(self, status_code):
        # if we hit the rate limit, raise an error
        # this will crash this program an send a mail (if started via cron)
        if status_code == 420:
            raise Exception("hit streaming API rate limit")


def judge_tweet(tweet):
    print("judging", tweet.id)
    # download image or continue
    media = tweet.entities.get('media', [])
    for i, m in enumerate(media):
        url = m['media_url']
        fname = "{}_{}.png".format(tweet.id, i)
        urllib.request.urlretrieve(url, fname)

        # judge image
        good = classifier.is_good(fname)
        os.remove(fname)

        print("good" if good else "bad")
        # if good: like and retweet
        if good:
            api.create_favorite(tweet.id)
            api.retweet(tweet.id)


def answerMentions():
    """Answers mentions with images of graphs.
    """
    stalkee = "AFractalADay"
    try:
        # are there new mentions while we were not listening?
        todo = obtain_tweets_from(stalkee)
        print(len(todo), "new messages")
        for t in todo:
            judge_tweet(t)
    except:
        print("something went wrong", sys.exc_info())

    # listen for new mentions
    print("listening for tweets from", stalkee)
    myStreamListener = MyStreamListener(judge_tweet)
    myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
    myStream.filter(track=[stalkee])


if __name__ == '__main__':
    my_handle = get_my_handle()
    answerMentions()
