"""
Bajan
~~~~~~~~

:copyright: (c) 2012 by Sean Braithwaite

"""

__title__ = 'bajan'
__version__ = '0.0.1'
__author__ = 'Sean Braithwaite'
__copyright__ = 'Copyright 2012 Sean Braithwaite'

import os
import pickle
import re
import pprint
import math
import operator
import sys

pp = pprint.PrettyPrinter(indent=4)

from collections import defaultdict

DEFAULT_WEIGHT = 0.001

class Knowledge:
    def __init__(self, default):
        self.store = {}
        self.default = default

    def __getitem__(self, token):
        if token in self.store.keys():
            return self.store[token]
        else:
            return self.default

    def __setitem__(self, token, value):
        self.store[token] = value

    def keys(self):
        return self.store.keys()

    def values(self):
        return self.store.values()

    def __str__(self):
        return str(self.store)
       

BLACKLIST = ["the", "a", "it", "its", "for", "to", "is", "that"]

def display_token(token, score):
        color = 46 + (score * 12)
        sys.stdout.write(" \x1b[38;5;%dm" % color + token + "\x1b[0m")

def tokenize(text):
    # lowercase 
    # remove useless charecters
    # remove useless words
    
    text   = re.compile("[!@#\?\.,]").sub('', text)
    tokens = re.compile("\s+").split(text.lower().strip())

    acc = defaultdict(lambda: 0)
    for t in tokens:
        if t not in BLACKLIST:
            acc[t] = acc[t] + 1

    return acc.items()
    
class Classifier:
    def __init__(self):
        self.spammy_weight = 1.0
        self.hammy_weight = 1.0

        self.spam_threshhold = 0

        self.label_counts = Knowledge(0)
        self.token_counts = {}
        self.document_counts = Knowledge(0)

        self._document_count = False
        self._token_count = False


    @staticmethod
    def build(filename):
        if os.path.exists(filename):
            return pickle.load(open(filename, 'rb'))
        else: 
            return Classifier()

    def display(self, text, classifier_view="normal"):
        """ Display text with classification style """ 

        if classifier_view == "normal":
            sys.stdout.write(text)
        elif classifier_view == "color-tokens":
            for token, occurances in tokenize(text):
                display_token(token, self.calculate_spammyness(token))
            print

        elif classifier_view == "color":
            scores = defaultdict(lambda: 0.4)
            for token, occurances in tokenize(text):
                scores[token] = self.calculate_spammyness(token)

            for word in text.split(' '):
                display_token(word, scores[word])

        else:
            raise "invalid classifier view %s " % classifier_view 
    

    def train(self, label, text):
        """ Update counters """

        self.label_counts[label] = self.label_counts[label] + 1

        tokens = tokenize(text)

        for token, count in tokens:
            self.document_counts[token] = self.document_counts[token]+1

            if not label in self.token_counts.keys():
                self.token_counts[label] = Knowledge(0)
            
            self.token_counts[label][token] = self.token_counts[label][token] + count


    @property
    def document_count(self):
        if not self._document_count:
            self._document_count = sum(self.label_counts.values())

        return self._document_count


    @property            
    def token_count(self):
        if not self._token_count:
            self._token_count = sum(reduce(lambda x,y: x + y, ([t.values() for t in self.token_counts.values()])))

        return self._token_count


    def calc_tf_idf(self, token, local_occurances):
        """ Term frequences, inverse document frequency """

        global_occurances    = max(sum([subset[token] for subset in self.token_counts.values()]), local_occurances)
        documents_containing = max(self.document_counts[token], 1)

        score = local_occurances / (1.0 * global_occurances * (math.log(1.0 * self.document_count/(documents_containing))))

        return score


    def calculate_spammyness(self, token):
        """ Spammyness measure as the difference of spam score and ham score """

        return (self.spammy_weight * self.token_counts["spam"][token]/self.label_counts["spam"]) - (self.hammy_weight*self.token_counts["ham"][token]/self.label_counts["ham"])


    def classify_spammyness(self, text):
        """ Classify text based on spammyness measure. This differs from alternative classification as it's a score and not a probability """

        document_spammyness = 0
        for token, occurances in tokenize(text):
            token_spammyness = self.calculate_spammyness(token)
            document_spammyness = document_spammyness + (token_spammyness * self.calc_tf_idf(token, occurances))

        if document_spammyness > self.spam_threshhold:
            return "spam"
        else: 
            return "ham"

    def classify(self, text):
        """ Calculate posterior and predict a label """

        def stipulate(label):
            prior = 1.0 * self.label_counts[label] / sum(self.label_counts.values())
            posterior = prior

            for token, occurances in tokenize(text):
                tf_idf = self.calc_tf_idf(token, occurances)
                
                weight = max(self.token_counts[label][token], DEFAULT_WEIGHT)
                posterior = posterior * tf_idf * weight / self.label_counts[label]

            return [label, posterior]

        stipulations = [stipulate(label) for label in self.label_counts.keys()]
        stipulations.sort(key=lambda x: x[1], reverse=True)

        return stipulations[0][0]

    def debug(self):
        print "Counts"
        pp.pprint(self.label_counts)

        for label in self.label_counts.keys():
            print "Label: %s" % label
            pp.pprint(self.token_counts[label].store)


    def persist(self, filename):
        """ Pickle the model and persist it to a file """

        pickle.dump(self, open(filename, 'wb'))
