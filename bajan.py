#!/usr/bin/env python

import sys

from optparse import OptionParser
from bajan import Classifier

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename", default="knowledge.pkl",
                  help="persist model to filename")

parser.add_option("-v", "--view", dest="view", default="normal",
                  help="""Decorate the output\n
                       color: color the charecters based on the classifier score
                       color-tokens: colors the actual tokens that the classifier sees
                        """)

(options, args) = parser.parse_args()

if len(args) != 1:
    print "requires a mode, either \"train\" or \"classify\""
    exit(1)
else:
    mode = args[0]


def run_training():
    try:
        for line in sys.stdin:
            label, text = line.split("\t")
            classifier.train(label, text)

    except KeyboardInterrupt:
        print "EXIT"
        exit(0)

    classifier.persist(options.filename)


def run_test():
    success = 0
    failure = 0

    try:
        for line in sys.stdin:
            label, text = line.split("\t")
            prediction = classifier.classify(text)

            if prediction == label:
                success = success +1
            else:
                failure = failure + 1

    except KeyboardInterrupt:
        print "EXIT"
        exit(0)

    # Evaluation
    print "Accuracy %f" % (1.0 * success / (success + failure))

def run_filter(target_label):
    try:
        for line in sys.stdin:
            split = line.split("\t")
            if len(split) == 1:
                text = split[0]
            else:
                text = split[1]

            prediction = classifier.classify(text)

            if prediction == target_label:
                classifier.display(text, options.view)

    except KeyboardInterrupt:
        print "EXIT"
        exit(0)


classifier = Classifier.build(options.filename)

classifier.spammy_weight = 1.0
classifier.hammy_weight = 1.0

if mode == "train":
    run_training()
elif mode == "test":
    run_test()
elif mode == "spam":
    run_filter("spam")
elif mode == "ham":
    run_filter("ham")
elif mode == "debug":
    classifier.debug()
else:
    print "Unknown mode, try train or classify"
