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
        self.context_size = 3
        # The right and left context size (So the total window length is 2*self.context_size)
        self.number_processed_word = 0
        # The number of processed word we are in

        log("Loading the alignment...", reinit=True)
        self.load_alignment()
        log("Alignment loaded")
        if preprocessed_data + ".npz" in os.listdir():
            log("Loading preprocessed data from file...")
            self.load_from_file(preprocessed_data)
            log("Preprocessed data loaded from file")
        else:
            log("Preprocessing data: loading audio...")
            self.load_audio()
            log("Preprocessing data: audio loaded")
            log("Preprocessing data: padding audio...")
            self.pad()
            log("Preprocessing data: audio padded")
            log("Preprocessing data: computing targets...")
            self.compute_targets()
            log("Preprocessing data: targets computed")
            log("Preprocessing data: flattening the data...")
            self.flatten()
            log("Preprocessing data: data flattened")
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
        Cuts it into words, and populates self.a2w and self.w2a
        :param id: The id of the current audio file we're in
        Is used to get the forced alignment data
        :param name_file: The path to the file to process
        :return: A list of 2D arrays corresponding to the words of the file name_file
        If there is no alignment data for this file, returns an empty list
        """
        data, sample_rate = soundfile.read(name_file)
        try:
            tmp = []
            words, times = self.models[id]
            for index, timestamp in enumerate(times):
                start, end = timestamp
                word_audio = data[int(start * sample_rate):int(end * sample_rate)]
                feature_matrix = mspec(word_audio).astype('float32')
                tmp.append(feature_matrix)
                self.AUDIO_MAX_SIZE = max(self.AUDIO_MAX_SIZE, feature_matrix.shape[0])
                self.a2w[self.number_processed_word] = words[index]
                self.w2a[words[index]] += [self.number_processed_word]
                self.number_processed_word += 1
            return tmp
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
        """Loads the features and the targets in self.audio and self.targets respectively
        :param preprocessed_data: The name of the .npz file where the data is stored
        """
        tmp = np.load(preprocessed_data + ".npz", allow_pickle=True)
        self.audio = tmp['audio_features']
        self.targets = tmp['audio_targets']
        self.w2a = tmp['w2a'].item()
        self.a2w = tmp['a2w'].item()

    def save_to_file(self, preprocessed_data):
        """Saves to the disk the features and the targets
        :param preprocessed_data: The name of thr .npz where the data will be stored
        """
        np.savez_compressed(preprocessed_data, audio_features=self.audio, audio_targets=self.targets,
                            a2w = self.a2w, w2a = self.w2a)

    def compute_targets(self):
        """Computes the target contexts for each word in self.audio
        Populates self.targets with a list of 2D arrays of shape
            (self.context_size*2*self.AUDIO_MAX_SIZE, self.feature_dim)
        The context is constituted of the surrounding words (self.context_size words to the left, the same to the right)
        When at the boundary, fill with zeros
        """
        for file in self.audio:
            n = len(file)
            for i in range(n):
                context = []
                # Left context
                while len(context) + i < self.context_size:
                    context.append(np.zeros((self.AUDIO_MAX_SIZE, self.feature_dim)))
                for t in range(self.context_size - len(context), 0, -1):
                    context.append(file[i - t])
                # Right context
                for t in range(1, min(self.context_size, n - i)):
                    context.append(file[i + t])
                while len(context) < 2 * self.context_size:
                    context.append(np.zeros((self.AUDIO_MAX_SIZE, self.feature_dim)))
                self.targets.append(np.concatenate(context))

    def flatten(self):
        """Flattens self.audio and self.targets to 3D numpy arrays
        """
        self.targets = np.array(self.targets).astype('float32')
        self.audio = np.concatenate(self.audio).astype('float32')