import numpy as np
import tqdm
import os
import re

TRAIN_DATA = "data/train-others-500"


def load_from_path(path):
    data = []
    for root, dirs, files in os.walk(TRAIN_DATA):
        for file in tqdm(files):
            if file.endswith('.wav'):
                pass
    return np.array(data)

def read_a_line(line):
    id, words, times = re.sub('["]','', line[:-3] ).split(' ')
    commas_index = np.where(np.array(list(words)) == ',')[0]
    times = times.split(',')


    word_dict = {}
    for i in range(len(commas_index) - 1):
        word = words[commas_index[i]+1: commas_index[i+1]]
        word_dict[word] = (times[i], times[i + 1])

    try:
        del word_dict['']
    except KeyError:
        pass
    to_return = {}
    to_return["id"] = id
    to_return['times'] = word_dict
    return id, to_return

models = {}
with open("test_seg.txt", 'r') as f:
    for line in f.readlines():

        id_temp, dict_temp = read_a_line(line)
        models[id_temp] = dict_temp

