import numpy as np
from ast import literal_eval
import os
import errno


def check_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def conv_to_np_float32(data):
    data = data.apply(literal_eval)
    return np.array(
        [np.array(xi, dtype=np.float32) for xi in data.values])


def gen_bitmaps(df, indexes):
    p = [0] * len(indexes)
    # if face is not present: then add to the list
    if indexes == [[]]:
        print("Not found")
        return
    else:
        for i in range(len(indexes)):
            p[i] = list(df.iloc[indexes[i][0], 1:])
            p[i] = ''.join(str(int(x)) for x in p[i])
            p[i] = int(p[i], 2)
    return p


def list_anding(bitmap_list):
    and_result = bitmap_list[0]
    i = 1
    while True:
        and_result = and_result & bitmap_list[i]
        i = i + 1
        if i == len(bitmap_list):
            break
    max_len = max([len(bin(i)[2:]) for i in bitmap_list])
    # write comment for this theeesss logic////
    offsets = list(find_offsets(and_result))[::-1]
    offsets = [max_len - item for item in offsets]
    return offsets


def list_oring(bitmap_list):
    or_result = bitmap_list[0]
    i = 1
    while True:
        or_result = or_result | bitmap_list[i]
        i = i + 1
        if i == len(bitmap_list):
            break
    max_len = max([len(bin(i)[2:]) for i in bitmap_list])
    # write comment for this theeesss logic////
    offsets = list(find_offsets(or_result))[::-1]
    offsets = [max_len - item for item in offsets]
    return offsets


def find_offsets(haystack):
    """
    Find the start of all (possibly-overlapping) instances of needle in haystack
    """
    offs = -1
    while True:
        offs = offs + 1
        if not haystack:
            break
        if haystack & 1:
            yield offs
        haystack = haystack >> 1


def compute_start_end_pairs(segments):
    pair_list = []
    for each_i in segments:
        start = each_i.split(":")[0]
        end = each_i.split(":")[1]
        pair_list.append([start, end])
    return pair_list

# print(compute_and([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]))
