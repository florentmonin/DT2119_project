import os
from tqdm import tqdm
import re
import numpy as np


def read_a_line(line):
    """Reads a line from a TextGrid file and transforms it in a usable dict of timestamps
    :param line: A String of the shape
    <id> ",<WORD1>,<WORD2>,...,<WORDn>," "<TIME1>,<TIME2>,...,<TIMEk>"
    where <id> is the id of the corresponding file, the <WORDi> are the words of the transcription of said file
    and <TIMEi> is the time corresponding to the i-th comma between the <WORD>
    :return: A list of the used words in the line, and a list of tuple (<TIME_start>, <TIME_end>)
    with <TIME_start>  and <TIME_end> the timestamps corresponding to the pronunciation of the corresponding word
    """
    id, words, times = re.sub('["]', '', line[:-3]).split(' ')
    commas_index = np.where(np.array(list(words)) == ',')[0]
    times = times.split(',')
    words_list = []
    times_list = []
    for i in range(len(commas_index) - 1):
        word = words[commas_index[i] + 1: commas_index[i + 1]]
        if word != '':
            words_list.append(word)
            times_list.append((times[i], times[i + 1]))
    return id, words_list, times_list


class DataProcessor:

    def __init__(self, text_path, audio_path):
        self.text_path = text_path
        # The file containing all the forced alignment TextGrids
        self.audio_path = audio_path
        # The file containing all the audio samples
        self.models = {}
        # A dictionary that maps id to the forced alignment for corresponding file
        self.audio = []
        # A list of lists of 2D numpy arrays, where each 2D array corresponds to the mspec of a word
        # Each element of self.audio is a list of words corresponding to one file (.flac)
        self.AUDIO_MAX_SIZE = 0
        # The max size of a word (for padding purposes)

        self.load_alignment()
        self.load_audio()

    def load_alignment(self):
        """Loads the forced alignment data
        Populates the attribute self.models
        """
        for root, dirs, files in os.walk(self.text_path):
            for file in tqdm(files):
                if file.endswith('.txt'):
                    name_file = os.path.join(root, file)
                    with open(name_file, 'r') as f:
                        for line in f.readlines():
                            id_temp, words_temp, times_temp = read_a_line(line)
                            self.models[id_temp] = (words_temp, times_temp)

    def load_audio(self):
        for speaker in os.scandir(self.audio_path):
            for dir in os.scandir(speaker):
                for audio_file in os.listdir(dir):
                    if audio_file.endswith('.flac'):
                        name_file = os.path.join(dir.path, audio_file)
                        id = audio_file[:-5]  # 19-198-0000
                        self.process_audio(id, name_file)

    def process_audio(self, id, name_file):
        """
        :param id:
        :param name_file:
        :return: A list of 2D arrays corresponding to the words of the file name_file
        """

        pass
