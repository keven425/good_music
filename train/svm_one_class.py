# transform pitch matrix into string of keys, e.g. a b c d e ...
# 12 letters to represent 12 semi tones

import csv
import sys

from sklearn.feature_extraction.text import *
from sklearn import svm
from sklearn.metrics import *


csv.field_size_limit(sys.maxsize)

def read():
    with open('./preprocessed/training/preprossed_after_2010_top_500.csv', 'rb') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        array = []
        for row in csv_reader:
            array.append(row)
    return array

def learn(songs):
    # all labels are positive, since it's one class svm
    raw_features = [song[2] for song in songs]
    vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer='word', token_pattern='\w+')
    features = vectorizer.fit_transform(raw_features)
    labels = [1 for i in range(len(songs))]
    # features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.0)

    clf = svm.OneClassSVM(kernel="linear", tol=10.)
    clf.fit(features)
    labels_predicted = clf.predict(features)
    print('positive test examples:')
    accuracy = accuracy_score(labels, labels_predicted)
    print('accuracy: ' + str(accuracy))


learn(read())


def learn_simple():
    features_train = [[1, 1],
                      [1, 2],
                      [2, 1],
                      [2, 2]]
    labels_train = [1, 1, 1, 1]

    clf = svm.OneClassSVM(kernel="rbf")
    clf.fit(features_train)
    labels_predicted = clf.predict([[1.5 , 1.5],
                                    [1.99, 1.99],
                                    [1.01, 1.99]])
    print('positive test examples:')
    accuracy = accuracy_score(labels_train, labels_predicted)
    print('accuracy: ' + str(accuracy))

# learn_simple()