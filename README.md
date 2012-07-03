Bajan
-----
Simple, Naive Bayesian Classifier


Training
========

Bajan can train, test and classify messages as spam or ham based on input from stdin.
The format is assumed to be one tab seperated example per line. The first field is the label, spam or ham, and the second field is the text. See data/corpus.txt as an example


```shell
$ head -n 1000 data/corpus.txt|python bajan.py train
```

By default, the model will be persisted (via pickle) into a file called knowledge.pkl.

```shell
$ head -n 2000 data/corpus.txt|python bajan.py train -f large_training.pkl
```

The -f flag allows you to specify the that will be persisted. This is handy for trying out models based on
different samplings.


Testing
=======

Putting bajan in test mode will read all the labeled data from stdin and output an accuracy score. Percentage of successful classifications.

```shell
$ tail -n 1000 data/corpus.txt|python bajan.py test
```

Classifying 
===========

With a trained model, you can send unlabeled data and bajan will filter by the specified label

```shell
$ echo "Hey, it's me mom. Happy birthday"|python bajan.py ham
```

To get a better idea of what the classifyer is doing, you print the text colored based on the classification score.

```shell
$ tail -n 20|python bajan.py ham -v color
```

Alternatively, the color-tokens level of verbosity will show you the tokens the classifier actually see's. This differs from the full text as
the classifyer will remove certain words and charecters as well as lowercase words before scoring them.

```shell
$ tail -n 20 data/corpus.txt|python bajan.py ham -v color-tokens
```
