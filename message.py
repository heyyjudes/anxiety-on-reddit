#this file will include message class and author class
import nltk

class RedMessage:
    """Class for Reddit message"""
    def __init__(self, id, title, body, author, subreddit, date_time):
        self.title = title
        self.body = body
        self.author = author
        self.subreddit = date_time
        self.id = id
        self.sentences = []
        self.sentences_in_words = []
        self.words = []

    def tokenize_words(self):
        self.words = nltk.word_tokenize(self.body)

    def tokenize_sentence(self):
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.sentences = sent_detector.tokenize(self.body)
        for sent in self.sentences:
            s_words = nltk.word_tokenize(sent)
            self.sentences_in_words.append(s_words)

class RedAuthor:
    """Class for Reddit author"""
    def __init__(self, username):
        self.name = username
        self.messages = []
        self.anxiety = False

