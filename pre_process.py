#this file will handle the pre-processing after scraping data from Reddit
import nltk
import csv
import message
import sys
import gensim
import logging
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i had",
    "i'll": "i will",
    "im" : "i am",
    "i'm" : "i am",
    "ive" : "i have",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had ",
    "it'd've": "it would have",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they had",
    "they'll": "they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": " where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}

class PreProccessor:
    """ class for pre processing"""
    def __init__(self, input_csv):
        self.input_file = input_csv
        self.message_arr = []
        self.author_arr = []
        self.texts = []
        with open(input_csv, 'rb') as csvfile:
            reader = csv.DictReader(csvfile)
            id = 0
            for row in reader:
                new_message = message.RedMessage(id, row['title'], row['selftext'], row['author_name'], row['subreddit'], row['created_utc'])
                self.message_arr.append(new_message)
                self.texts.append(new_message.body)
                self.texts.append(new_message.title)
                id += 1

    def filter_message(self):
        #filter by length
        new_arr = []
        for Rmsg in self.message_arr:
            if len(Rmsg.body) > 60:
                new_arr.append(Rmsg)
            # else:
            #     print "too short"
            #     print Rmsg.body
        self.message_arr = new_arr

    def build_dictionary(self):
        for message in self.message_arr:
            for word in message.body:
                print
        from collections import defaultdict
        frequency = defaultdict(int)

        # here we want to build a dictionary of words
        return

    def tokenize(process):
        print("tokenizing")
        tokenizer2 = RegexpTokenizer(r'\w+')
        tokenizer1 = WhitespaceTokenizer()
        docs = process.texts
        tokens = []
        for doc in docs:
            raw = doc.lower()

            #white space tokenize
            token = tokenizer1.tokenize(raw)

            #extending contractions
            for i in range(0, len(token)):
                if token[i] in contractions.keys():
                    token[i] = contractions[str(token[i])]

            raw = " ".join(token)

            #regex tokenizing
            token = tokenizer2.tokenize(raw)
            for i in range(0, len(token)):
                if token[i] == "r":
                    print ('found')
                    print(raw)
                    print token
            tokens.append(token)
        return tokens


    def remove_stop(tokens):
        print("removing stops")
        en_stop = get_stop_words('en')
        en_stop.append('just')
        stopped_tokens = []
        for token in tokens:
            s_token = [i for i in token if not i in en_stop]
            stopped_tokens.append(s_token)
        return stopped_tokens



