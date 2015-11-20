# transform pitch matrix into string of keys, e.g. a b c d e ...
# 12 letters to represent 12 semi tones

import csv
import sys
import random
import json
from itertools import groupby
# import demjson
import ast
import math
from operator import add

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import *
from sklearn import cross_validation
from sklearn import svm
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import mixture
from sklearn.metrics import *
from sklearn.utils import shuffle

import numpy as np
import scipy as sp
import Utils.File
import Billboard.train.instrument_mapping as instrument_mapping


def get_song_flat_features(list):
    return [note for track in list for note in track]


def deserialize_feature_from_song(song):
    # return ast.literal_eval(song[4])
    return json.loads(song[2])


def get_features(songs):
    # return map(lambda song: get_song_flat_features(song), songs)

    return [deserialize_feature_from_song(song) for song in songs]
    # return [json.loads(song[4]) for song in songs]


# return f1 score
def get_metrics(labels_predicted, labels_actual):
    # print('metrics:')
    percent_positive_actual = float(sum(labels_actual)) / len(labels_actual)
    # print('percent labels positive actual: ' + str(percent_positive_actual))
    percent_positive_predicted = float(sum(labels_predicted)) / len(labels_predicted)
    # print('percent labels positive predicted: ' + str(percent_positive_predicted))
    accuracy = accuracy_score(labels_actual, labels_predicted)
    # print('accuracy: ' + str(accuracy))
    precision = precision_score(labels_actual, labels_predicted)
    # print('precision: ' + str(precision))
    recall = recall_score(labels_actual, labels_predicted)
    # print('recall: ' + str(recall))
    f1 = f1_score(labels_actual, labels_predicted)
    # print('f1: ' + str(f1))

    return f1, precision, recall



ngram_range=3
count = 0


def get_ngram(notes, ngram_range=3):
    # notes = [chr(ord('a') + note) for note in notes]
    # notes = ''.join(notes)
    features = []
    for i in range(len(notes) - ngram_range + 1):
        window = notes[i:i + ngram_range]
        # normalize chord = [note, note, note]
        offset = window[0]
        # normalized_cord = ':'.join([str(note - offset) for note in window])
        normalized_cord = ':'.join([str(note) for note in window])
        features.append(normalized_cord)
    return features

# absolute difference between two consequtive notes
def get_interval(notes):
    features = []
    for i in range(len(notes) - 2 + 1):
        window = notes[i:i + 2]
        # normalize chord = [note, note, note]
        difference = abs(window[1] - window[0])
        features.append(chr(ord('a') + difference))
    return features

# def get_interval_beat(notes):
#     features = []
#     for i in range(len(notes) - 2 + 1):
#         window = notes[i:i + 2]
#         # normalize chord = [note, note, note]
#         difference = abs(window[1] - window[0])
#         features.append(chr(ord('a') + difference))
#     return features

def analyzer(notes):
    global count
    count = count + 1
    print("processed " + str(count) + " samples")

    window = ''
    # remove consecutive duplicates
    # notes = [x[0] for x in groupby(notes)]

    # flatten notes
    # notes = get_song_flat_features(notes)
    return get_ngram(notes)

def instrument_2_notes_analyzer(channel_2_track):
    features = []
    global count
    count = count + 1
    print("processed " + str(count) + " samples")

    for channel, track in channel_2_track.iteritems():
        # skip percussion track
        if (channel == 9):
            continue

        instrument = track['instrument']
        notes_times = track['notes']

        notes = [note_time[0] for note_time in notes_times]
        two_notes = get_interval(notes)
        notes = [note % 12 for note in notes] # mod 12
        three_notes = get_ngram(notes, 3)
        notes = two_notes + three_notes
        features += [instrument_mapping.instruments[instrument] + ':' + note for note in notes]
    return features


def round_tempo(delta):
    if (delta == 0):
        return 0
    magnitude = 10 ** (int(math.log(abs(delta), 10)) - 1)
    return int(delta) / magnitude * magnitude

def get_tempos(times):
    time_delta = [j - i for i, j in zip(times[:-1], times[1:])]
    if (len(time_delta) <= 0):
        return[]
    if (isinstance(time_delta[0], ( int, long ))):  # no tempo info present, defaulted to 1 second per tick
        # http://stackoverflow.com/questions/2038313/midi-ticks-to-actual-playback-seconds-midi-music,  a 120 BPM track would have a MIDI time of (60000 / (120 * 192)) or 2.604 ms for 1 tick
        # default to 120 beat per minute
        time_delta = [delta * 0.002604 for delta in time_delta]
    time_delta = [delta for delta in time_delta if delta > 0]
    delta_int = [int(60.0 / delta) for delta in time_delta]
    delta_int = [delta for delta in delta_int if delta > 0]
    return [round_tempo(delta) for delta in delta_int]

def beats_analyzer(channel_2_track):
    features = []
    global count
    count = count + 1
    print("processed " + str(count) + " samples")

    for channel, track in channel_2_track.iteritems():
        channel = int(channel)
        instrument = track['instrument']
        notes_times = track['notes']
        times = [note_time[1] for note_time in notes_times]
        # percussion track
        tempos = get_tempos(times)
        tempos = get_ngram(tempos, 3)
        if (channel == 9):
            features += ['beats:' + str(tempo) for tempo in tempos]
        else:
            features += tempos

    return features

def interval_beat_analyzer(channel_2_track):
    global count
    count = count + 1
    print("processed " + str(count) + " samples")

    beats_features = beats_analyzer(channel_2_track)
    instruments_notes_features = instrument_2_notes_analyzer(channel_2_track)

    return beats_features + instruments_notes_features

def get_features_labels(positive_songs, negative_songs, analyzer=instrument_2_notes_analyzer):
    raw_features_positive = get_features(positive_songs)
    raw_features_negative = get_features(negative_songs)
    vectorizer = CountVectorizer(analyzer=analyzer)
    all_features = vectorizer.fit_transform(raw_features_positive + raw_features_negative)
    positive_features = all_features[:len(positive_songs)]
    negative_features = all_features[len(negative_songs):]
    positive_labels = [1 for i in range(len(positive_songs))]
    negative_labels = [0 for j in range(len(negative_songs))]
    return positive_features, negative_features, positive_labels, negative_labels, vectorizer.get_feature_names()


def learn_classify(positive_features, negative_features, positive_labels, negative_labels, clf=svm.LinearSVC()):
    # group positive, negative features and labels
    features = sp.sparse.vstack((positive_features, negative_features))
    labels = positive_labels + negative_labels
    features, labels = shuffle(features, labels)

    kf = cross_validation.KFold(features.shape[0], n_folds=10)

    f1s = []
    precisions = []
    recalls = []
    for train_index, test_index in kf:
        features_train, features_test = features[train_index], features[test_index]
        labels = np.array(labels)
        labels_train, labels_test = labels[train_index], labels[test_index]
        clf.fit(features_train.toarray(), labels_train)
        labels_predicted = clf.predict(features_test.toarray())

        f1, precision, recall = get_metrics(labels_predicted, labels_test)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    mean_f1 = sum(f1s) / len(f1s)
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    return mean_f1, mean_precision, mean_recall

def learn_classify_logistic_regression(positive_features, negative_features, positive_labels, negative_labels, cutoff = 0.5, C=1.0):
    # group positive, negative features and labels
    features = sp.sparse.vstack((positive_features, negative_features))
    labels = positive_labels + negative_labels
    features, labels = shuffle(features, labels)

    kf = cross_validation.KFold(features.shape[0], n_folds=10)
    # clf = svm.LinearSVC()
    # clf = naive_bayes.GaussianNB()
    clf = linear_model.LogisticRegression(C=C)

    f1s = []
    precisions = []
    recalls = []
    for train_index, test_index in kf:
        features_train, features_test = features[train_index], features[test_index]
        labels = np.array(labels)
        labels_train, labels_test = labels[train_index], labels[test_index]
        clf.fit(features_train.toarray(), labels_train)
        prob_predicted = clf.predict_proba(features_test.toarray())
        labels_predicted = [1 if (prob[1] > cutoff) else 0 for prob in prob_predicted]

        f1, precision, recall = get_metrics(labels_predicted, labels_test)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    mean_f1 = sum(f1s) / len(f1s)
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    return mean_f1, mean_precision, mean_recall

# take both beat and instrument to train separately
global_cutoff = 0.5
def should_be_positive_two_prob(prob_instrument, prob_beat):
    global global_cutoff
    if (prob_instrument[1] >= global_cutoff and prob_beat[1] >= global_cutoff):
        return 1
    return 0


def print_top_features(features_weight, instrument_feature_names, message):
    top100 = np.argsort(features_weight)[-100:]
    print('==============')
    print(message)
    for i in top100:
        print instrument_feature_names[i]


def learn_classify_ensemble(beats_features, instrument_features, labels, instrument_feature_names, beat_feature_names, cutoff=0.5):
    global global_cutoff
    global_cutoff = cutoff

    kf = cross_validation.KFold(instrument_features.shape[0], n_folds=10)
    # clf = svm.LinearSVC(C=1.0)
    # clf = naive_bayes.GaussianNB()
    clf = linear_model.LogisticRegression()
    f1s = []
    precisions = []
    recalls = []
    instrument_features_weight = np.zeros(instrument_features.shape[1])
    beats_features_weight = np.zeros(beats_features.shape[1])
    for train_index, test_index in kf:
        instrument_features_train, instrument_features_test = instrument_features[train_index], instrument_features[
            test_index]
        beats_features_train, beats_features_test = beats_features[train_index], beats_features[test_index]

        labels = np.array(labels)
        labels_train, labels_test = labels[train_index], labels[test_index]

        clf.fit(instrument_features_train.toarray(), labels_train)
        instrument_prob_predicted = clf.predict_proba(instrument_features_test.toarray())
        instrument_features_weight += clf.coef_[0]

        clf.fit(beats_features_train.toarray(), labels_train)
        beats_prob_predicted = clf.predict_proba(beats_features_test.toarray())
        beats_features_weight += clf.coef_[0]

        # add probabilities element wise
        # prob_predicted = map(add, instrument_prob_predicted, beats_prob_predicted)
        # labels_predicted = [1 if (prob[1] > cutoff) else 0 for prob in prob_predicted]
        labels_predicted = map(should_be_positive_two_prob, instrument_prob_predicted, beats_prob_predicted)

        f1, precision, recall = get_metrics(labels_predicted, labels_test)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    mean_f1 = sum(f1s) / len(f1s)
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    print_top_features(instrument_features_weight, instrument_feature_names, 'top instrument features:')
    print_top_features(beats_features_weight, beat_feature_names, 'top beats features:')
    return mean_f1, mean_precision, mean_recall

def learn_outlier_detection(positive_features, negative_features, cutoff = 0.5):
    print('===============')
    # features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=int(random.random() * 100))

    kf = cross_validation.KFold(positive_features.shape[0], n_folds=10)
    # clf = mixture.GMM(n_components=1)
    clf = svm.OneClassSVM(kernel="linear")

    f1s = []
    precisions = []
    recalls = []
    for train_index, test_index in kf:
        features_train = positive_features[train_index]
        features_test = sp.sparse.vstack((positive_features[test_index], negative_features[test_index]))
        labels_test = [1 for i in range(len(test_index))] + [0 for j in range(len(test_index))]
        clf.fit(features_train.toarray())
        # prob_predicted = clf.predict_proba(features_test.toarray())
        # labels_predicted = [1 if (prob[0] > cutoff) else 0 for prob in prob_predicted]
        labels_predicted = clf.predict(features_test.toarray())
        f1, precision, recall = get_metrics(labels_predicted, labels_test)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    mean_f1 = sum(f1s) / len(f1s)
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    return mean_f1, mean_precision, mean_recall


def classify_precision_recall_change_cutoff(positive_features, negative_features, positive_labels, negative_labels):
    print("cutoff,\tmean precision,\tmean recall,\tmean f1")
    cutoffs = [x / 100.0 for x in range(4, 100, 4)] + [x / 1000.0 for x in range(962, 1000, 2)]
    for cutoff in cutoffs:
        mean_f1, mean_precision, mean_recall = learn_classify_logistic_regression(positive_features, negative_features, positive_labels, negative_labels, cutoff)
        print(str(cutoff) + ',\t' + str(mean_precision) + ',\t' + str(mean_recall) + ',\t' + str(mean_f1))


def print_header():
    print("cutoff,\tmean precision,\tmean recall,\tmean f1")


def get_separate_feature_labels(negative_songs, positive_songs):
    positive_instruments_features, negative_instruments_features, positive_labels, negative_labels, instrument_feature_names = get_features_labels(
        positive_songs, negative_songs, instrument_2_notes_analyzer)
    positive_beats_features, negative_beats_features, positive_labels, negative_labels, beat_feature_names = get_features_labels(
        positive_songs, negative_songs, beats_analyzer)
    # group positive, negative features and labels
    instrument_features = sp.sparse.vstack((positive_instruments_features, negative_instruments_features))
    beats_features = sp.sparse.vstack((positive_beats_features, negative_beats_features))
    labels = positive_labels + negative_labels
    instrument_features, beats_features, labels = shuffle(instrument_features, beats_features, labels)
    return beats_features, instrument_features, labels, beat_feature_names, instrument_feature_names


def classify_ensemble_precision_recall_change_cutoff(positive_songs, negative_songs):
    beats_features, instrument_features, labels, beat_feature_names, instrument_feature_names = get_separate_feature_labels(
        negative_songs, positive_songs)

    print_header()
    cutoffs = [x / 100.0 for x in range(4, 100, 4)] + [x / 1000.0 for x in range(962, 1000, 2)]
    # cutoffs = [0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 1.0]
    # cutoffs = [0.0, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001] + [x / 1000.0 for x in range(2, 38, 2)]
    for cutoff in cutoffs:
        mean_f1, mean_precision, mean_recall = learn_classify_ensemble(beats_features, instrument_features, labels, instrument_feature_names, beat_feature_names, cutoff)
        print(str(cutoff) + ',\t' + str(mean_precision) + ',\t' + str(mean_recall) + ',\t' + str(mean_f1))

def classify_ensemble_top_features(positive_songs, negative_songs):
    beats_features, instrument_features, labels, beat_feature_names, instrument_feature_names = get_separate_feature_labels(
        negative_songs, positive_songs)

    learn_classify_ensemble(beats_features, instrument_features, labels, instrument_feature_names, beat_feature_names, cutoff=0.998)

def classify_precision_recall_change_regularization(positive_features, negative_features, positive_labels, negative_labels):
    print("regularization,\tmean precision,\tmean recall,\tmean f1")
    regularizations = [0.1, 0.2, 0.5, 0.75, 1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50]
    for regularization in regularizations:
        mean_f1, mean_precision, mean_recall = learn_classify_logistic_regression(positive_features, negative_features, positive_labels, negative_labels, C=1.0/regularization)
        print(str(regularization) + ',\t' + str(mean_precision) + ',\t' + str(mean_recall) + ',\t' + str(mean_f1))

def outlier_detection_precision_recall_change_cutoff(positive_features, negative_features):
    print_header()
    cutoffs = [x / 100.0 for x in range(4, 100, 4)] + [x / 1000.0 for x in range(962, 1000, 2)]
    for cutoff in cutoffs:
        mean_f1, mean_precision, mean_recall = learn_outlier_detection(positive_features, negative_features, cutoff)
        print(str(cutoff) + ',\t' + str(mean_precision) + ',\t' + str(mean_recall) + ',\t' + str(mean_f1))

def random_baseline_precision_recall(positive_songs, negative_songs):
    length = len(positive_songs)
    labels = [1 for i in range(0, length)] + [0 for j in range(0, length)]
    print_header()
    for i in range(0, 11, 1):
        predicted_labels = [(random.random() * 10 > i) and 1 or 0 for j in range(0, length * 2)]
        mean_f1, mean_precision, mean_recall = get_metrics(predicted_labels, labels)
        print(str(i) + ',\t' + str(mean_precision) + ',\t' + str(mean_recall) + ',\t' + str(mean_f1))


features_dir = '../../free_midi/features/'
positive_features_file = 'positive_pitch_duration_serialized.csv'
negative_features_file = 'negative_pitch_duration_serialized.csv'

positive_songs = Utils.File.read_csv(features_dir + positive_features_file, False)
negative_songs = Utils.File.read_csv(features_dir + negative_features_file, False)
negative_songs = negative_songs[:len(positive_songs)]
# random_baseline_precision_recall(positive_songs, negative_songs)
# positive_features, negative_features, positive_labels, negative_labels = get_features_labels(positive_songs, negative_songs, beats_analyzer)
# outlier_detection_precision_recall_change_cutoff(positive_features, negative_features)
# classify_precision_recall_change_cutoff(positive_features, negative_features, positive_labels, negative_labels)
# classify_ensemble_precision_recall_change_cutoff(positive_songs, negative_songs)
classify_ensemble_top_features(positive_songs, negative_songs)

# print_header()
# mean_f1, mean_precision, mean_recall = learn_outlier_detection(positive_features, negative_features)
# print(',\t' + str(mean_precision) + ',\t' + str(mean_recall) + ',\t' + str(mean_f1))
# mean_f1, mean_precision, mean_recall = learn_classify(positive_features, negative_features, positive_labels, negative_labels, svm.LinearSVC())
# print(',\t' + str(mean_precision) + ',\t' + str(mean_recall) + ',\t' + str(mean_f1))
# print('naive_bayes')
# mean_f1, mean_precision, mean_recall = learn_classify(positive_features, negative_features, positive_labels, negative_labels, naive_bayes.GaussianNB())
# print(',\t' + str(mean_precision) + ',\t' + str(mean_recall) + ',\t' + str(mean_f1))

# toy example
# clf = linear_model.LogisticRegression()
# clf.fit([[0, 1], [0, 0]], [0, 1])
# print(clf.coef_[0])
#
# clf.fit([[0, 0], [0, 1]], [1, 0])
# print(clf.coef_[0])