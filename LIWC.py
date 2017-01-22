''' Make LIWC feature extractor into class
'''
from nltk.tokenize import WhitespaceTokenizer
import re
import pickle
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit
liwcPath = 'LIWC2007_English080730.dic'
def makeLIWCDictionary(liwcPath, picklePath):
    '''
        Make lookup data structure from LIWC dictionary file
    '''
    LIWC_file = open(liwcPath, 'rb') # LIWC dictionary
    catNames = {}
    LIWC_file.readline() #skips first '%' line
    line = LIWC_file.readline()
    lookup = []
    while '%' not in line:
    	keyval = line.split('\t')
    	key = keyval[0]
    	value = keyval[1].strip()
    	catNames[key] = {'name' : value,
                         'words' : []}
    	line = LIWC_file.readline()
    mapCategoriesToNumbers = catNames.keys()
    line = LIWC_file.readline() # skips second '%' line

    #return mapCategoriesToNumbers
    while line: #iterate through categories
    	data = line.strip().split('\t')
    	reString = '^'+data[0].replace('*', '.*') + '$'
        indeces = [mapCategoriesToNumbers.index(d) for d in data[1:]]
    	lookupCell = (re.compile(reString), indeces)
        lookup.append(lookupCell)
        for cat in data[1:]:
            catNames[cat]['words'] += (data[0], reString)
    	cats = data[1:]
    	line = LIWC_file.readline()
    toPickle = {'categories' : catNames, 'lookup' : lookup, 'cat_to_num' : mapCategoriesToNumbers}
    pickle.dump(toPickle, open(picklePath, 'w'))
    return toPickle

class liwcExtractor():
    def __init__(self,
                tokenizer=None,
                ignore=None,
                dictionary=None,
                newCategories=None,
                keepNonDict=True,
                liwcPath=None):
        self.liwcPath = liwcPath
        self.dictionary = dictionary
        if tokenizer is None:
            self.tokenizer = WhitespaceTokenizer
        if liwcPath is not None:
            self.dictionary = makeLIWCDictionary(liwcPath, './liwcDictionary.pickle')
            self.lookup = self.dictionary['lookup']
            self.categories = self.dictionary['categories']
            self.mapCategoriesToNumbers = self.dictionary['cat_to_num']
        elif self.dictionary==None:
            self.dictionary = makeLIWCDictionary(liwcPath, './liwcDictionary.pickle')
            self.lookup = self.dictionary['lookup']
            self.categories = self.dictionary['categories']
            self.mapCategoriesToNumbers = self.dictionary['cat_to_num']
        self.ignore = ignore
        self.newCategories = newCategories
        self.nonDictTokens = []
        self.keepNonDict = keepNonDict

    def getCategoryIndeces(self):
        indeces = [x['name'] for x in self.categories.values()]
        indeces += ['wc', 'sixltr','dic','punc','emoticon'] # These last two are not built yet.
        return indeces

    def extract(self, corpus):
        corpusFeatures = []
        for doc in corpus:
            features = self.extractFromDoc(doc)
            corpusFeatures.append(features)
        return corpusFeatures

    def extractFromDoc(self, document):
        tokens = document
        features = [0] * 70 # 66 = wc, total word count
                            # 67 = sixltr, six letter words
                            # 68 = dic, words found in LIWC dictionary
                            # 70 = punc, punctuation
                            # 71 = emoticon
        features[66] = len(tokens)

        for t in tokens: #iterating through tokens of a message
            #print "Token : " + t
            if len(t) > 6: # check if more than six letters
                features[67] += 1
            inDict = False
            for pattern, categories in self.lookup:
                if len(pattern.findall(t)) > 0:
                    inDict = True
                    for c in categories:
                        features[int(c)] += 1
            if inDict:
                features[68] += 1
            else:
                self.nonDictTokens.append(t)
        return features

def cleanText(corpus):
    corpus = [z.lower().replace('\n', '').split() for z in corpus]
    return corpus

def return_vec(liwc, doc):
    results = liwc.extractFromDoc(doc)
    preportions = [str(round(x/float(results[66]),3)) for x in results]
    return preportions

if __name__ == "__main__":


    liwc = liwcExtractor(liwcPath=liwcPath)

#translating to liwc vectors
    with open('data/anxiety_content.txt', 'r') as infile:
        dep_posts = infile.readlines()

    with open('data/liwc_anxious.txt', 'w') as outfile:
        for post in dep_posts:
            if len(post) > 5:
                result_vec = return_vec(liwc, post)
                outfile.write('-'.join(result_vec))
    print "done anxious"

    with open('data/mixed_content.txt', 'r') as infile:
        reg_posts = infile.readlines()

    with open('data/liwc_mixed.txt', 'w') as outfile:
        for post in dep_posts:
            if len(post) > 5:
                result_vec = return_vec(liwc, post)
                outfile.write('-'.join(result_vec))

    print "done mixed"








