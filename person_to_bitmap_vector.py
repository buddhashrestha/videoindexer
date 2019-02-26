import numpy
import pandas as pd
import faiss
import ast
import os
import io
from Clustering import Clustering


def check_intersection(segment_start, segment_end, each_interval):
    start_end_points = each_interval.split(":")
    start = float(start_end_points[0])
    end = float(start_end_points[1])
    if start >= segment_start and end <= segment_end:  # then we have that person in that interval
        return True


def generate_face_timeline(sorted_segments, vid_num):
    current_directory = os.getcwd()
    df_person = pd.DataFrame(
        columns=['index', 'segment'])

    for i, val in enumerate(sorted_segments):
        if i == len(sorted_segments)-1:
            break
        df_len = len(df_person)
        # add a person to the index
        df_person = df_person.append({'index': int(i)}, ignore_index=True)
        df_person.loc[[df_len], 'segment'] = str(
            sorted_segments[i])+":"+str(sorted_segments[i+1])
    df_person = df_person.astype({"index": int})
    file_name = "./data/" + str(vid_num) + '/person_segment.csv'
    with open(file_name, 'a') as f:
        return df_person.to_csv(file_name, sep='\t', index=False)


def cluster_and_save(embeddings_file, vid_num):

    import os
    import sys
    sys.path.append('../')
    import pandas as pd
    import numpy as np
    import utils

    from FaissWrapper import FaissWrapper

    clust2 = Clustering(embeddings_file)
    emb = clust2.get_embeddings()

    # video duration in seconds
    # TODO: this should be optimized!!!
    video_duration = int(emb['time'].iloc[-1])
    ar = [0] * (video_duration + 2)
    embeddings_labels_mappings = {}
    person = {}

    faiss = FaissWrapper()

    data = None
    data_matrix = "./data/person_to_video_matrix.csv"
    if (os.path.exists(data_matrix)):
        df_matrix = pd.read_csv(data_matrix, sep='\t')
        df_matrix = df_matrix.loc[:, ~
                                  df_matrix.columns.str.contains('^Unnamed')]

        cols = list(df_matrix)
        # cols.insert(0, cols.pop(cols.index('person')))
        df_matrix = df_matrix.ix[:, cols]

        data = utils.conv_to_np_float32(df_matrix['person'])

        video_num = vid_num
        # by default every video is zero
        df_matrix[video_num] = 0
        # searching section

        faiss.train(data)

        # ****************** this part can be reduced if dlib responds ******************
        for each_label in clust2.labels:
            p = emb.loc[emb['cluster'] == each_label].iloc[:, 2:-1].values
            r = []
            for i, t in enumerate(p):
                temp = np.array(list(t))
                temp = temp.astype('float32')
                temp = temp.reshape(1, 128)
                r.append(temp)

            dist = -1
            min = 1000
            real_pos = 0
            for j in r:
                I = faiss.searcher(j)  # actual search
            # if face is not present: then add to the list
                if not I == [[]]:
                    pos = I[0][0]
                    z = data[pos]
                    dist = numpy.linalg.norm(z - j)
                    if dist < min:
                        real_pos = pos
                        min = dist

            if min < 0.578:
                df_matrix.iloc[real_pos,
                               df_matrix.columns.get_loc(video_num)] = 1
            else:

                df_len = len(df_matrix)
                df_matrix.loc[df_len] = 0
                df_matrix.at[df_len, video_num] = 1
                df_matrix.at[df_len, 'person'] = r[0].tolist()


            # *************************************************************************
    else:
        df_matrix = pd.DataFrame(columns=['person', 1])
        video_num = 1
        for each_label in clust2.get_labels():
            q = np.array(
                list(emb.loc[emb['cluster'] == each_label].iloc[0, 2:-1]))
            q = q.astype('float32')

            df_len = len(df_matrix)
            df_matrix = df_matrix.append(
                {'person': [], 1: int(1)}, ignore_index=True)
            df_matrix = df_matrix.fillna(0)
            df_matrix.at[df_len, 'person'] = q.tolist()
    print(df_matrix)
    exit(0)
    cols = list(df_matrix)
    # cols.insert(0, cols.pop(cols.index('person')))
    df_matrix = df_matrix.ix[:, cols]
    print("DF_matrix after: ", df_matrix)
    file_name = './data/person_to_video_matrix.csv'
    # #
    with open(file_name, 'a') as f:
        df_matrix.to_csv(data_matrix, sep='\t', index=False)

    if not data.any():
        data = utils.conv_to_np_float32(df_matrix['person'])

    sorted_segments = []
    starting_point = clust2.starting_point
    for each_segment in starting_point.itersegments():
        sorted_segments.append(float(each_segment.start) + float(0.00005))
        sorted_segments.append(float(each_segment.end))
    sorted_segments = sorted(list(set(sorted_segments)))

    generate_face_timeline(sorted_segments, vid_num)

    person_segment_path = "./data/" + str(vid_num) + "/person_segment.csv"
    if (os.path.exists(person_segment_path)):
        person_segment = pd.read_csv(person_segment_path, sep='\t')

    df_person = pd.DataFrame(columns=person_segment['index'].tolist())

# ********************* this much part can be replaced if dlib provides the embeddings *************
    for each_label in clust2.labels:
        if (os.path.exists(data_matrix)):
            p = emb.loc[emb['cluster'] == each_label].iloc[:, 2:-1].values
            r = []
            for i, t in enumerate(p):
                temp = np.array(list(t))
                temp = temp.astype('float32')
                temp = temp.reshape(1, 128)
                r.append(temp)

            dist = -1
            min = 1000
            real_pos = 0
            for j in r:
                I = faiss.searcher(j)  # actual search
                # if face is not present: then add to the list
                if not I == [[]]:
                    pos = I[0][0]
                    z = data[pos]
                    print("z type : ", type(z))
                    print("j type : ", type(j))
                    exit(0)
                    dist = numpy.linalg.norm(z - j)
                    if dist < min:
                        real_pos = pos
                        min = dist
            if dist == -1:
                print("theres no person there!!")
                embeddings_labels_mappings[each_label] = list(
                    emb.loc[emb['cluster'] == each_label].iloc[0, 2:-1])

            else:
                if min < 0.6:
                    embeddings_labels_mappings[each_label] = df_matrix.iloc[real_pos, df_matrix.columns.get_loc(
                        'person')]
                else:
                    embeddings_labels_mappings[each_label] = list(
                        emb.loc[emb['cluster'] == each_label].iloc[0, 2:-1])

        else:
            embeddings_labels_mappings[each_label] = list(
                emb.loc[emb['cluster'] == each_label].iloc[0, 2:-1])

# **********************************

        df_len = len(df_person)
        # add a person to the index
        df_person = df_person.append(
            {'person_label': each_label}, ignore_index=True)

        label_segments = starting_point.label_timeline(each_label)
        for segment in label_segments:
            print("Something")
            for index, row in person_segment.iterrows():
                if check_intersection(float(segment.start), float(segment.end), row['segment']):
                    df_person.loc[[df_len], row['index']] = int(1)
        df_person = df_person.fillna(0)
        df_person = df_person.astype(int)

    df_embeddings = pd.DataFrame(list(embeddings_labels_mappings.items()), columns=[
                                 'person_label', 'Embeddings'])

    utils.check_directory("./data/" + str(vid_num))

    file_name = "./data/" + str(vid_num) + '/person_bitmap_vector.csv'
    with open(file_name, 'a') as f:
        df_person.to_csv(file_name, sep='\t', index=False)

    file_name = "./data/" + str(vid_num) + '/person_embeddings_mapping.csv'
    with open(file_name, 'a') as f:
        df_embeddings.to_csv(file_name, sep='\t', index=False)

    #cluster_and_save_naive(embeddings_file, vid_num)
    print("Done with naive ... ")


def cluster_and_save_naive(embeddings_file, vid_num):

    import os
    import sys
    sys.path.append('../')
    from pyannoteVideo.pyannote.video.face.clustering import FaceClustering
    import pandas as pd
    import numpy as np

    clustering = FaceClustering(threshold=0.6)
    face_tracks, embeddings = clustering.model.preprocess(embeddings_file)
    #
    result = clustering(face_tracks, features=embeddings)

    clust2 = Clustering(embeddings_file)
    #
    current_directory = os.getcwd()

    person_segment_path = "./data/" + \
        str(vid_num) + "/person_segment_naive.csv"
    person_segment_naive = pd.DataFrame(columns=['person', 'segment'])

    starting_point = clust2.get_starting_point()
    for each_label in clust2.get_labels():
        print(each_label)
        segments = starting_point.label_timeline(each_label)
        dictionary = {}
        r = []
        for each_segment in segments:
            l = []
            l.append(each_segment._get_start())
            l.append(each_segment._get_end())
            r.append(l)

        df_len = len(person_segment_naive)
        person_segment_naive = person_segment_naive.append(
            {'person': each_label, 'segment': []},  ignore_index=True)
        person_segment_naive.at[df_len, 'segment'] = r

    with open(person_segment_path, 'a') as f:
        person_segment_naive.to_csv(person_segment_path, sep='\t', index=False)


cluster_and_save("./data/" + str(2) + "/" + "bbt1.embedding.txt", 2)
# cluster_and_save("./data/" + str(1) + "/" + "friends1_720.embedding.txt", 1)
#
