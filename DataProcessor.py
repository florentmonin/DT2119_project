import os
from tqdm import tqdm
import re
import numpy as np
from collections import defaultdict
from lab1_proto import *
import soundfile


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

    def __init__(self, text_path, audio_path, preprocessed_data):
        self.text_path = text_path
        # The file containing all the forced alignment TextGrids
        self.audio_path = audio_path
        # The file containing all the audio samples
        self.models = {}
        # A dictionary that maps id to the forced alignment for corresponding file
        self.audio = []
        # A 3D numpy arrays, where each 2D array corresponds to the MSpec of a word
        # Each element of self.audio is a list of words corresponding to one file (.flac)
        self.targets = []
        # Contains a list of context matrices
        # where a context matrix is the concatenation of context words feature matrices
        self.AUDIO_MAX_SIZE = 0
        # The max size of a word (for padding purposes)
        self.a2w = {}
        # A dictionary indicating for each 2D array in self.audio the corresponding word
        self.w2a = defaultdict(list)
        # A dictionary indicating for each word a list of audio positions where this word in uttered
        self.feature_dim = 40
        # Trust us, see implementation of mspec

        self.load_alignment()
        if preprocessed_data in os.listdir():
            self.load_from_file(preprocessed_data)
        else:
            self.load_audio()
            self.pad()
            self.compute_targets()
            self.flatten()
            self.save_to_file(preprocessed_data)

    def load_alignment(self):
        """Loads the forced alignment data
        Populates the attribute self.models
        """
        for root, dirs, files in os.walk(self.text_path):
            for file in files:
                if file.endswith('.txt'):
                    name_file = os.path.join(root, file)
                    with open(name_file, 'r') as f:
                        for line in f.readlines():
                            id_temp, words_temp, times_temp = read_a_line(line)
                            self.models[id_temp] = (words_temp, times_temp)

    def load_audio(self):
        """Loads the audio files in self.audio
        Transforms them in a feature matrix for each word
        Word matrices are stored by file (one file is made of a collection of audio files all read by the same speaker,
        and within the same context
        """
        for speaker in os.scandir(self.audio_path):
            for dir in os.scandir(speaker):
                tmp = []
                for audio_file in os.listdir(dir):
                    if audio_file.endswith('.flac'):
                        name_file = os.path.join(dir, audio_file)
                        id = audio_file[:-5]  # 19-198-0000
                        tmp += self.process_audio(id, name_file)
                self.audio.append(tmp)

    def process_audio(self, id, name_file):
        """Processes one audio file
        Cuts it into words, and populates self.audio, self.a2w and self.w2a
        :param id: The id of the current audio file we're in
        Is used to get the forced alignment data
        :param name_file: The path to the file to process
        :return: A list of 2D arrays corresponding to the words of the file name_file
        If there is no alignment data for this file, returns an empty list
        """
        x, y = len(self.audio), len(self.audio[-1])
        data, sample_rate = soundfile.read(name_file)
        try:
            words, times = self.models[id]
            for index, timestamp in enumerate(times):
                start, end = timestamp
                word_audio = data[int(start*sample_rate):int(end*sample_rate)]
                feature_matrix = mspec(word_audio).astype('float32')
                self.audio[-1].append(feature_matrix)
                self.AUDIO_MAX_SIZE = max(self.AUDIO_MAX_SIZE, feature_matrix.shape[0])
                self.a2w[(x, y)] = words[index]
                self.w2a[words[index]] += [(x, y)]
        except KeyError:
            return []

    def pad(self):
        """Adds 0s at the end of the 2D matrices for each word, so that each matrix has the same shape
        """
        for i, file in enumerate(self.audio):
            for j, word in enumerate(file):
                padded_audio = np.zeros((self.AUDIO_MAX_SIZE, self.feature_dim), dtype='float32')
                padded_audio[:word.shape[0]] = word
                self.audio[i][j] = padded_audio
            self.audio[i] = np.array(self.audio[i])

    def load_from_file(self, preprocessed_data):
        """
        TODO
        :param preprocessed_data:
        :return:
        """
        self.audio = np.load(preprocessed_data+".npz", allow_pickle=True)['audio_data']

    def save_to_file(self, preprocessed_data):
        """
        TODO
        :param preprocessed_data:
        :return:
        """
        np.savez(preprocessed_data, audio_data=self.audio)

    def compute_targets(self):
        """
        TODO
        :return:
        """
        pass

    def flatten(self):
        """Flattens self.audio to a 3D numpy array
        TODO
        :return:
        """
        pass
