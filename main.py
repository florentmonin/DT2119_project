import numpy as np
from tqdm import tqdm
import os
import re

TRAIN_DATA = "data/train-others-500"


def load_from_path(path):
    """Loads the forced alignment data
    :param path: The location of the TextGrid files
    :return: #TODO
    """
    data = []
    for root, dirs, files in os.walk(TRAIN_DATA):
        for file in tqdm(files):
            if file.endswith('.wav'):
                pass
                # TODO
    return np.array(data)


def read_a_line(line):
    """Reads a line from a TextGrid file and transforms it in a usable dict of timestamps
    :param line: A String of the shape
    <id> ",<WORD1>,<WORD2>,...,<WORDn>," "<TIME1>,<TIME2>,...,<TIMEk>"
    where <id> is the id of the corresponding file, the <WORDi> are the words of the transcription of said file
    and <TIMEi> is the time corresponding to the i-th comma between the <WORD>
    :return: A dict associating words to a tuple of times: {<WORD>: (<TIME_start>, <TIME_end>)}
    with <TIME_start>  and <TIME_end> the timestamps corresponding to the pronunciation of <WORD>
    """
    id, words, times = re.sub('["]', '', line[:-3]).split(' ')
    commas_index = np.where(np.array(list(words)) == ',')[0]
    times = times.split(',')

    word_dict = {}
    for i in range(len(commas_index) - 1):
        word = words[commas_index[i] + 1: commas_index[i + 1]]
        word_dict[word] = (times[i], times[i + 1])

    try:
        del word_dict['']
    except KeyError:
        pass
    return id, word_dict


models = {}
with open("test_seg.txt", 'r') as f:
    for line in f.readlines():
        id_temp, dict_temp = read_a_line(line)
        models[id_temp] = dict_temp
