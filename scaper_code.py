import praw

from csv import DictWriter, DictReader
import csv
from datetime import datetime
from time import gmtime

SUBREDDITS = ['panicparty', 'worldnews', 'history', 'mentalhealth', 'sports', 'askreddit']
POST_KEYS = ['title','created_utc','score','subreddit','domain','is_self','over_18','selftext', 'id']
CSV_KEYS = ['title','created_utc','score','subreddit','domain','is_self','over_18','selftext', 'author_name', 'id', 'a_id']
SCRAPE_AUTHORS = True

processed_users = {}
def get_author_info(a):
    if a:
        try:
            if a.id in processed_users:
                return processed_users[a.id]
            else:
                d = {}
                d['author_name'] = a.name
                d['a_id'] = a.id
                processed_users[a.id] = d
                return d
        except AttributeError:
            return {'author_name': ''}

    else:
        return {'author_name':''}

def process_post(post):
    d = {}
    postdict = vars(post)
    for key in POST_KEYS:
        val = postdict[key]
        try:
            val = val.lower()
        except:
            pass
        d[key.encode('ascii')] = val

    if SCRAPE_AUTHORS:
        try:
            author_dict = get_author_info(post.author)
        except praw.errors.NotFound:
            return None
        for key,val in author_dict.iteritems():
            d[key] = val
    return d

if __name__ == '__main__':
    r = praw.Reddit('Reddit Dataset builder')
    ids = []
    posts = []
    filename = 'reddit_'+ '+'.join(SUBREDDITS) + '_' + '.csv'

    # with open(filename, 'rb') as csvfile:
    #     reader = DictReader(csvfile)
    #     id = 0
    #     for row in reader:
    #         ids.append(row['id'])
    #         d = {}
    #         d['author_name'] = row['author_name']
    #         a_id = row['a_id']
    #         d['a_id'] = a_id
    #         processed_users[a_id] = d
    #
    #
    #
    # with open(filename, 'a') as fid:
    with open(filename, 'w') as fid:
        csv_writer = DictWriter(fid, CSV_KEYS)
        csv_writer.writeheader()

        if len(SUBREDDITS) > 0:
            for subreddit in SUBREDDITS:
                print 'scraping subreddit:',subreddit
                sub = r.get_subreddit(subreddit)
                i = 0
                print 'scraping new posts...'
            #    posts =  [process_post(p) for p in sub.get_new(limit=1000)]
            #    ids = [p['id'] for p in posts]
                for post in sub.get_new(limit=100):
                    if post.id not in ids:
                        print i
                        print post.title
                        result = process_post(post)
                        if result is not None:
                            posts.append(result)
                            try:
                                csv_writer.writerow(result)
                            except (UnicodeEncodeError, ValueError):
                                pass

                            ids.append(post.id)
                            i+=1

                print 'scraping top posts...'
                for post in sub.get_top_from_all(limit=100):
                    if post.id not in ids:
                        print i
                        print post.title
                        result = process_post(post)
                        if result is not None:
                            posts.append(result)
                            try:
                                csv_writer.writerow(result)
                            except (UnicodeEncodeError, ValueError):
                                pass
                            ids.append(post.id)
                            i += 1

                print 'scraping controversial posts...'
                for post in sub.get_controversial_from_all(limit=100):
                    if post.id not in ids:
                        print i
                        print post.title
                        result = process_post(post)
                        if result is not None:
                            posts.append(result)
                            try:
                                csv_writer.writerow(result)
                            except (UnicodeEncodeError, ValueError):
                                pass
                            ids.append(post.id)
                            i += 1
        else:
            print 'scraping frontpage...'
            SUBREDDITS = ['frontpage']
            for post in r.get_front_page(limit=1000):
                print post.title
                posts.append(process_post(post))

    print 'scraped ',len(posts),' posts'
