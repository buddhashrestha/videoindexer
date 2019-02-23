import pandas as pd
import ast
import numpy as np
from ast import literal_eval
import faiss
from FaissWrapper import FaissWrapper
import utils
'''
Base for singleton class
'''


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


'''
This is the class for connecting with the filesystem where all the data
related to parsed video are stored. 
'''


class FileSystem:
    def __init__(self, fs_path="./data/"):
        self.fs_path = fs_path

    def get_video_embeddings(self, video_number):
        embeddings_path = self.fs_path + \
            str(video_number) + '/person_embeddings_mapping.csv'
        # read the file
        df = pd.read_csv(embeddings_path, sep='\t')
        # define a order to read the columns
        df = df[['Embeddings', 'person_label']]
        # convert str column type to list
        df['Embeddings'] = df['Embeddings'].apply(literal_eval)
        # return standard datatypes in array
        # TODO: optimize this??
        return df['person_label'].values, utils.conv_to_np_float32(df['Embeddings'])

    def get_video_bitmap(self, video_number):
        bitmap_path = self.fs_path + \
            str(video_number) + '/person_bitmap_vector.csv'
        df_person_bitmap = pd.read_csv(bitmap_path, sep='\t')
        df_person_bitmap = df_person_bitmap.loc[:, ~df_person_bitmap.columns.str.contains(
            '^Unnamed')]  # removing unnecessary column
        return df_person_bitmap

    def get_video_segments(self, video_number):
        segment_path = self.fs_path + str(video_number) + '/person_segment.csv'
        df_segment = pd.read_csv(segment_path, sep='\t')
        return df_segment['segment']

    def findVideos(self, query_object):
        data_matrix = self.fs_path + 'person_to_video_matrix.csv'
        df = pd.read_csv(data_matrix, sep='\t')

        # put 'person' column as first column
        cols = list(df)
        cols.insert(0, cols.pop(cols.index('person')))
        df = df.ix[:, cols]

        # convert str column type to np.array
        embeddings_arr = utils.conv_to_np_float32(df['person'])

        # searching section
        indexes = FaissWrapper().search(
            embeddings_arr, query_object)

        bitmaps = utils.gen_bitmaps(df, indexes)
        and_result = utils.list_anding(bitmaps)

        return and_result

    # TODO:refactor this to a proper design pattern
    def get_person_bitmap(self, df_person_bitmap, nearest_labels):
        person_bitmap = [0] * len(nearest_labels)  # initiliaze the array
        for i in range(len(nearest_labels)):
            x = df_person_bitmap.loc[
                df_person_bitmap['person_label'] == int(
                    nearest_labels[i]), df_person_bitmap.columns != 'person_label'].values[0]
            x[np.isnan(x)] = 0

            person_bitmap[i] = x.astype(int)
            person_bitmap[i] = "".join(map(str, person_bitmap[i]))
            person_bitmap[i] = int(person_bitmap[i], 2)
        return person_bitmap

# f = FileSystem("/home/buddha/thesis/cauli/data/")
# # print(f.get_video_embeddings(1))

# q1 = np.array([-0.09394396,-0.0325009,0.04033981,0.058163,-0.07830061,-0.00957991,0.00967481,-0.01413031,0.09056109,0.01226895,0.26494166,-0.02123076,-0.24777047,-0.01763329,-0.07384738,0.06715217,-0.12217229,-0.08997168,-0.07580122,-0.04950837,0.04795098,0.13332213,-0.01127091,0.02197756,-0.11553577,-0.27555162,-0.07677643,-0.13606225,0.04441723,-0.15569574,0.00895849,0.14969787,-0.11935396,-0.06838058,0.07786268,0.03718393,-0.06408153,-0.02089238,0.23585948,0.0805631,-0.15866175,0.01856041,0.00116307,0.41145056,0.16814063,0.02847197,-0.03902237,-0.05839182,0.07346018,-0.2937974,0.11478464,0.17386746,0.17906441,0.07801438,0.06278698,-0.19769681,-0.02882751,0.14530656,-0.1747098,0.18388774,0.02294914,-0.13189943,0.02977884,-0.12073777,0.12760715,0.01692488,-0.10679697,-0.05800077,0.18581267,-0.10150822,-0.05918678,0.1363336,-0.13339984,-0.26164994,-0.22236374,0.06707402,0.3977724,0.11571225,-0.23185259,0.0054906,-0.01040609,-0.03162199,0.04665727,0.08441081,-0.05217192,-0.03122335,-0.07699484,0.00657031,0.17925991,-0.00695905,-0.06547561,0.22437108,-0.0341662,-0.03394335,0.13902065,0.03442689,-0.00662828,-0.03761093,-0.1410363,-0.02334402,-0.00892982,-0.22861557,-0.01857813,0.06306355,-0.14851157,0.18434337,0.02413726,-0.01379691,-0.00887952,-0.06088896,-0.07728448,0.06671096,0.24838527,-0.2970275,0.25738838,0.14351685,0.08496048,0.20006748,-0.00258899,0.0801791,-0.06668909,-0.06449306,-0.18159339,-0.07939189,0.05472237,-0.00587219,0.02267785,0.03292688])

# q2 = np.array([-0.13995799,0.07357684,0.06817903,-0.03358017,-0.14017977,0.02780785,-0.00525929,-0.01784202,0.06178749,-0.10179561,0.18950391,-0.01475109,-0.27838451,-0.0266105,-0.08373968,0.09460582,-0.1788917,-0.17850886,-0.12365769,-0.14781366,0.03512874,0.00903247,-0.02803879,0.03783317,-0.16920707,-0.26304603,-0.06992229,-0.10061786,0.05101402,-0.16632022,0.11655138,0.08433184,-0.12079179,0.03596556,-0.0121152,0.12947829,0.04827371,-0.08188809,0.18349151,-0.00281307,-0.15602723,0.06405699,0.04490693,0.31951201,0.17494193,0.01719636,-0.0419702,-0.08516614,0.12460594,-0.21499792,0.06585026,0.19651721,0.09745228,0.03903785,0.10535515,-0.13724521,0.08984233,0.13477094,-0.21704961,0.05335507,0.0309218,-0.05781218,0.0019028,-0.09391791,0.0646577,0.06179845,-0.08008289,-0.14667159,0.20857613,-0.18729421,-0.06652444,0.14170116,-0.1157265,-0.1452689,-0.24668945,0.04148631,0.40796083,0.21065351,-0.11024678,0.02329403,-0.01277202,-0.04557295,0.07328524,0.03643153,-0.17143802,-0.08823329,-0.10314541,0.11035879,0.15545008,0.09105477,-0.0749627,0.19225572,0.03504357,-0.04319753,0.07131915,0.02174823,-0.08037397,0.02350202,-0.12592137,-0.0058411,0.06064202,-0.16452999,-0.01263573,0.0567708,-0.13627924,0.06755719,-0.01812054,-0.00742016,-0.0428388,0.01858514,-0.18413505,0.08322754,0.24775918,-0.29474801,0.18072528,0.21811365,0.03798047,0.06725502,0.17808905,0.0480503,-0.01554211,-0.12508231,-0.1766559,-0.07591728,0.04641546,-0.08250329,0.07233171,0.03478055])

# q = []

# q.append(q1)
# q.append(q2)

# q = np.array(q)
# q = q.astype('float32')

# print(f.findVideos(q))
