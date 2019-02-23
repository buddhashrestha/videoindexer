from Database import FileSystem
from utils import *
import numpy as np
from segments import Segments
from FaissWrapper import FaissWrapper


class VideoIndexer:
    def __init__(self, fs_path="./data/"):
        self.fs = FileSystem(fs_path)

    def search(self, query_array, query_type):

        query_array = self.check_query(query_array)
        videos = FileSystem().findVideos(query_array)
        timings = {}

        for each_video in videos:
            # get labels and embeddings for that video
            person_labels, df_embeddings = self.fs.get_video_embeddings(
                each_video)
            indexes = FaissWrapper().search(df_embeddings, query_array)

            # if face is not present: then add to the list
            if indexes == [[]]:
                print("No faces found for video number :{}".format(each_video))
            else:
                # this returns the actual label from the row
                nearest_labels = [person_labels[i[0]][0]
                                  for i in indexes]  # i is a 2d array

            # get bitmaps for that video (each_video)
            df_person_bitmap = self.fs.get_video_bitmap(each_video)
            # get bitmap for the nearest labels
            person_bitmap = self.fs.get_person_bitmap(
                df_person_bitmap, nearest_labels)

            # get timing segments for that video
            segments = self.fs.get_video_segments(each_video)
            # transform the segments into [[start,end]] list
            start_end_pairs = compute_start_end_pairs(segments)

            if query_type == 'next':
                timings[each_video] = self.next(
                    person_bitmap[0], person_bitmap[1])
            if query_type == 'eventually':
                timings[each_video] = self.eventually(
                    person_bitmap[0], person_bitmap[1])
            if query_type == 'interval':
                timings[each_video] = self.interval(
                    person_bitmap, start_end_pairs)

        return timings

    def check_query(self, query_object):
        try:
            return np.array(query_object).astype('float32')
        except:
            raise ValueError(
                'The query object is not valid numpy array with type float32')

    def interval(self, person_bitmap, start_end_pairs):  # total 30us
        offsets = list_anding(person_bitmap)
        # calculate segment now
        segs = Segments().find_continous_segments(offsets, start_end_pairs)
        return segs

    def next(self, p1, p2):
        compute = p1 & (p2 << 1)
        if compute:
            return True
        else:
            return False

    def oring(self, person_bitmap, start_end_pairs):
        offsets = list_oring(person_bitmap)
        segs = Segments().find_continous_segments(offsets, start_end_pairs)
        return segs

    def eventually(self, p1, p2):
        if p1 > p2:
            return True
        else:
            return False
