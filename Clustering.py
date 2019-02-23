from pandas import read_table
from pyannote.core import Segment, Annotation
import numpy as np
import dlib
import pandas


class Clustering(object):
    def __init__(self, data_path):
        names = ['time', 'track']

        for i in range(128):
            names += ['d{0}'.format(i)]
        #
        self.data = read_table(data_path, delim_whitespace=True,
                               header=None, names=names)

        self.data.sort_values(by=['track', 'time'], inplace=True)

        # create a descriptor list with dlibs descriptor vector
        descriptors = []
        embeddings = self.data.iloc[:, 2:].values
        for each_i in embeddings:
            face_descriptor = dlib.vector(each_i)
            descriptors.append(face_descriptor)

        # returns series of labels [0 0 2 2 2] for each row of embeddings
        labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
        # put the series into a column
        self.data['cluster'] = pandas.Series(labels, index=self.data.index)
        # TODO: this can be improved by taking highest count of label in each track
        # get the label for each track
        track_label = self.data.groupby(by='track', as_index=False).first()[
            ['track', 'cluster']].values

        # get unique labels
        self.labels = np.unique(track_label[:][:, [1]])

        self.starting_point = Annotation(modality='face')

        for track, segment in self.data.groupby('track').apply(_to_segment).iteritems():
            if not segment:
                continue
            self.starting_point[segment, track] = track_label[track][1]

    def get_labels(self):
        return self.labels

    def get_embeddings(self):
        return self.data

    def get_starting_point(self):
        return self.starting_point


def _to_segment(group):
    return Segment(np.min(group.time), np.max(group.time))
