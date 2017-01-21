#this file will handle the pre-processing after scraping data from Reddit
import nltk
import csv
import message
import re
import os
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
POST_KEYS = ['title','created_utc','score','subreddit','domain','is_self','over_18','selftext']
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
        self.words = []
        self.unique_words = []
        self.to_write = []
        self.vocab_size = 0
        with open(input_csv, 'rb') as csvfile:
            reader = csv.DictReader(csvfile)
            id = 0
            for row in reader:
                new_message = message.RedMessage(id, row['title'], row['selftext'], row['author_name'], row['subreddit'], row['created_utc'])
                self.message_arr.append(new_message)
                combined_post = new_message.title + " " + new_message.body
                self.texts.append(combined_post)
                # self.texts.append(new_message.body)
                # self.texts.append(new_message.title)
                id += 1
            print "Total posts", id

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

    def tokenize(self):
        print("Tokenizing")
        tokenizer2 = RegexpTokenizer(r'\w+')
        tokenizer1 = WhitespaceTokenizer()
        tokens = []
        for i in range(len(self.texts)):
            raw = self.texts[i].lower()
            #print('raw txt')
            #print(raw)

            # white space tokenize
            token = tokenizer1.tokenize(raw)

            # extending contractions
            for i in range(0, len(token)):
                if token[i] in contractions.keys():
                    token[i] = contractions[str(token[i])]
                # removing links
                if (re.search('http', token[i])):
                    token[i] = ''

            raw = " ".join(token)
            # regex tokenizing
            token = tokenizer2.tokenize(raw)
            for i in range(0, len(token)):
                if token[i].isalnum() == False:
                    token[i] = ''
                if(token[i] not in self.unique_words):
                    self.vocab_size += 1
                    self.unique_words.append(token[i])
                self.words.append(token[i])

            tokens.append(token)
            raw = " ".join(token)
            self.to_write.append(raw)
        return tokens

    def tokenize_txt(self):
        print("Tokenizing")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = []
        for i in range(len(self.texts)):
            raw = self.texts[i].lower()
            # print('raw txt')
            # print(raw)

            # white space tokenize
            token = tokenizer1.tokenize(raw)

            # extending contractions
            for i in range(0, len(token)):
                if token[i] in contractions.keys():
                    token[i] = contractions[str(token[i])]
                # removing links
                if (re.search('http', token[i])):
                    token[i] = ''

            raw = " ".join(token)
            # regex tokenizing
            token = tokenizer2.tokenize(raw)
            for i in range(0, len(token)):
                if token[i].isalnum() == False:
                    token[i] = ''
                if (token[i] not in self.unique_words):
                    self.vocab_size += 1
                    self.unique_words.append(token[i])
                self.words.append(token[i])

            tokens.append(token)
            raw = " ".join(token)
            self.to_write.append(raw)
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

    def save_txt(self, str):
        print('Writing file')
        txt_str = str + '.txt'
        if(os.path.isfile(txt_str)):
            with open (txt_str, 'a') as f:
                for token in self.to_write:
                    f.write(token)
                    f.write('\n')
        else:
            with open(txt_str, 'w') as f:
                for token in self.to_write:
                    f.write(token)
                    f.write('\n')

        f.close()

class PreProccessor_text:
    """ class for pre processing"""

    def __init__(self, input_txt):
        self.input_file = input_txt
        self.texts = []
        self.words = []
        self.unique_words = []
        self.vocab_size = 0
        f = open(input_txt, 'rb')
        messages = f.readlines()
        for m in messages:
            self.texts.append(m)

    def tokenize_txt(self):
        print("Tokenizing")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = []
        for i in range(len(self.texts)):
            raw = self.texts[i].lower()

            token = tokenizer.tokenize(raw)
            for i in range(0, len(token)):
                if token[i].isalnum() == False:
                    token[i] = ''
                if (token[i] not in self.unique_words):
                    self.vocab_size += 1
                    self.unique_words.append(token[i])
                self.words.append(token[i])

            tokens.append(token)
        return tokens

    def train_test_split(self, text_size):
        length = len(self.texts)
        cutoff = int(length*text_size)

        test_str = 'test_set.txt'
        with open(test_str, 'w') as f:
            for i in range(0, cutoff):
                #print(self.texts[i])
                if len(self.texts[i]) > 3:
                    f.write(self.texts[i])

                else:
                    print('empty')
                    print(len(self.texts[i]))
                    print(self.texts[i])

        f.close()

        train_str = 'train_set.txt'
        with open(train_str, 'w') as f:
            for i in range(cutoff, length):
                if len(self.texts[i]) > 3:
                    f.write(self.texts[i])
                else:
                    print('empty')
                    print(len(self.texts[i]))
                    print(self.texts[i])

        f.close()

def merge_csv(input_csv1, input_csv2):
    ids=[]
    added = 0
    with open(input_csv1, 'r+') as csvfile1:
        reader = csv.DictReader(csvfile1)
        for row in reader:
            ids.append(row['created_utc'])

        with open(input_csv2, 'r') as csvfile2:
            reader = csv.DictReader(csvfile2)
            writer = csv.DictWriter(csvfile1, POST_KEYS)
            for row in reader:
                if row['created_utc'] not in ids:
                    writer.writerow(row)
                    print added
                    added +=1
                else:
                    print "match"



if __name__ == "__main__":
    #mixed files
    #newProcess = PreProccessor('data/reddit_askscience+writingprompts+atheism_.csv')
    #newProcess = PreProccessor('data/reddit_showerthoughts+lifeprotips+personalfinance_.csv')
    #newProcess = PreProccessor('data/reddit_theoryofreddit+randomkindness+relationships_.csv')
    #newProcess = PreProccessor('data/reddit_christianity+teaching+parenting_.csv')
    #newProcess = PreProccessor('data/reddit_jokes+writing+fitness_.csv')
    #newProcess = PreProccessor('data/reddit_talesfromretail+talesfromtechsupport+talesfromcallcenters_.csv')
    #newProcess = PreProccessor('data/reddit_suicidewatch_.csv')
    #newProcess = PreProccessor('data/reddit_panicparty+socialanxiety_.csv')
    #newProcess = PreProccessor('data/reddit_books+askdocs+legaladvice_.csv')
    newProcess = PreProccessor('data/reddit_frugal+youshouldknow+nostupidquestions_.csv')
    #newProcess = PreProccessor('data/reddit_panicparty+worldnews+history+mentalhealth+sports+askreddit_.csv')
    newProcess.tokenize()
    newProcess.save_txt('data/mixed_content')

    #splitProcess = PreProccessor_text('data/allsub_content.txt')
    #splitProcess.train_test_split(0.25)

    # newProcess = PreProccessor('data/reddit_anxiety_.csv')
    # newProcess.tokenize()
    # newProcess.save_txt('data/anxiety_content')



