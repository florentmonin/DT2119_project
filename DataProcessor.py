import os
from tqdm import tqdm
import re
import numpy as np
from collections import defaultdict
from lab1_proto import *
import soundfile
import time


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
            times_list.append((float(times[i]), float(times[i + 1])))
    return id, words_list, times_list


def log(message, file="log.log", reinit=False):
    """ Writes message in the file file
    If reinit, reinitialise the file
    :param message: Message to be printed in the log file
    :param file: File where the message is printed
    :param reinit: reinitialise the file if True
    :return:
    """
    if reinit:
        with open(file, "w+") as f:
            f.write(f"{time.asctime()} : {message} \n")
    else:
        with open(file, "a") as f:
            f.write(f"{time.asctime()} : {message} \n")


# noinspection SpellCheckingInspection
class DataProcessor:

    def __init__(self, text_path, audio_path, preprocessed_data):
        self.text_path = text_path
        # The file containing all the forced alignment TextGrids
        self.audio_path = audio_path
        # The file containing all the audio samples
        self.MEMORY_FEATURES = "memory/features/"
        # The paths where the data is saved
        self.models = {}
        # A dictionary that maps id to the forced alignment for corresponding file
        self.AUDIO_MAX_SIZE = 0
        # The max size of a word (for padding purposes)
        self.a2w = {}
        # A dictionary indicating for each 2D array in self.audio the corresponding word
        self.w2a = defaultdict(list)
        # A dictionary indicating for each word a list of audio positions where this word in uttered
        self.feature_dim = 40
        # Trust us, see implementation of mspec
        self.context_size = 3
        # The right and left context size (So the total window length is 2*self.context_size)
        self.number_processed_word = 0
        self.ids = []
        # The number of processed word we are in
        self.mean = np.zeros(self.feature_dim)
        self.std = np.zeros(self.feature_dim)

        log("Loading the alignment...", reinit=True)
        self.load_alignment()
        log("Alignment loaded")
        if preprocessed_data + ".npz" in os.listdir("."):
            log("Loading preprocessed data from file...")
            self.load_from_file(preprocessed_data)
            log("Preprocessed data loaded from file")
        else:
            log("Preprocessing data: loading audio...")
            os.mkdir(self.MEMORY_FEATURES)
            self.load_audio()
            log("Preprocessing data: audio loaded")
            log("Preprocessing data: padding audio...")
            self.pad()
            log("Preprocessing data: audio padded")
            log("Preprocessing data: normalizing data...")
            self.normalize()
            log("Preprocessing data: data normalized")
            log("Preprocessing data: saving data to file...")
            self.save_to_file(preprocessed_data)
            log("Preprocessing data: data saved to file")

    def load_alignment(self):
        """Loads the forced alignment data
        Populates the attribute self.models
        """
        for root, dirs, files in os.walk(self.text_path):
            for file in files:
                if file.endswith('.txt'):
                    name_file = os.path.join(root, file)
                    with open(name_file, 'r') as f:
                        for line in f:
                            try:
                                id_temp, words_temp, times_temp = read_a_line(line)
                                self.models[id_temp] = (words_temp, times_temp)
                            except IndexError as e:
                                log(f"Error while loading line {line}")
                                log(e)

    def load_audio(self):
        """Saves the features in a file
        Transforms them in a feature matrix for each word
        Word matrices are stored by file (one file is made of a collection of audio files all read by the same speaker,
        and within the same context
        """
        for speaker in os.scandir(self.audio_path):
            for dir in os.scandir(speaker):
                for audio_file in os.listdir(dir):
                    if audio_file.endswith('.flac'):
                        name_file = os.path.join(dir, audio_file)
                        id = audio_file[:-5]  # 19-198-0000-1.npz
                        self.process_audio(id, name_file)

    def process_audio(self, id, name_file):
        """Processes one audio file
        Cuts it into words, and populates self.a2w and self.w2a
        :param id: The id of the current audio file we're in
        Is used to get the forced alignment data
        :param name_file: The path to the file to process
        :return: A list of 2D arrays corresponding to the words of the file name_file
        If there is no alignment data for this file, returns an empty list
        """
        data, sample_rate = soundfile.read(name_file)
        i = 0
        try:
            words, times = self.models[id]
            for index, timestamp in enumerate(times):
                start, end = timestamp
                word_audio = data[int(start * sample_rate):int(end * sample_rate)]
                feature_matrix = mspec(word_audio).astype('float32')
                self.mean += np.sum(feature_matrix, axis=0)
                self.std += np.sum(feature_matrix ** 2, axis=0)
                np.savez(f"{self.MEMORY_FEATURES}{id}-{i}", word=feature_matrix)
                self.ids.append(f"{id}-{i}")
                self.AUDIO_MAX_SIZE = max(self.AUDIO_MAX_SIZE, feature_matrix.shape[0])
                self.a2w[self.number_processed_word] = words[index]
                self.w2a[words[index]] += [self.number_processed_word]
                self.number_processed_word += 1
                i += 1
        except KeyError:
            pass

    def pad(self):
        """Adds 0s at the end of the 2D matrices for each word, so that each matrix has the same shape
        """
        for file in os.listdir(self.MEMORY_FEATURES):
            word = np.load(f"{self.MEMORY_FEATURES}{file}")
            padded_audio = np.zeros((self.AUDIO_MAX_SIZE, self.feature_dim), dtype='float32')
            padded_audio[:word.shape[0]] = word
            np.savez(f"{self.MEMORY_FEATURES}{file}", word=padded_audio)

    def load_from_file(self, preprocessed_data):
        """Loads the information about the features.
        :param preprocessed_data: The name of the .npz file where the data is stored
        """
        tmp = np.load(preprocessed_data + ".npz", allow_pickle=True)
        self.w2a = tmp['w2a'].item()
        self.a2w = tmp['a2w'].item()
        self.ids = tmp['ids']

    def save_to_file(self, preprocessed_data):
        """Saves to the disk the the information about the preprocessed data
        :param preprocessed_data: The name of the .npz where the data will be stored
        """
        np.savez_compressed(preprocessed_data, a2w=self.a2w, w2a=self.w2a, ids=self.ids)

    def normalize(self):
        """Normalizes the mspec data
        """
        self.mean /= self.number_processed_word
        self.std = np.sqrt(abs(self.std - self.mean ** 2))
        for file in os.listdir(self.MEMORY_FEATURES):
            tmp = np.load(f"{self.MEMORY_FEATURES}{file}" + ".npz", allow_pickle=True)['word']
            tmp = (tmp - self.mean) / self.std
            np.savez(f"{self.MEMORY_FEATURES}{file}", word=tmp)
