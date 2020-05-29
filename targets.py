import os
import numpy as np
from DataProcessor import *
from tqdm import tqdm

data_processor = DataProcessor(r"light_segmentation", r"light_data", "train_data")
SHAPE = (data_processor.AUDIO_MAX_SIZE, data_processor.feature_dim)
ids = data_processor.ids

print("Starting to compute targets")

def context(id):
    """Computes the target contexts for the word stored in the file id
    :param id: The id of the word whose context is to be computed, of the shape xxx-xxxx-xxxx-x
    For instance, 19-198-0000-5 corresponds to the word uttered by speaker 19,
    reading excerpt 198, in the sentence 0000, and is the sixth word in the sentence
    The context is constituted of the surrounding words (self.context_size words to the left, the same to the right)
    When at the boundary, fill with zeros
    """
    ctxt = []
    last_id = id
    for i in range(3):
        ctxt.append(previous_word(last_id, ids))
        last_id = ctxt[-1]
    ctxt.reverse()

    last_id = id
    for i in range(3):
        ctxt.append(next_word(last_id, ids))
        last_id = ctxt[-1]
    return ctxt


def next_word(id, ids):
    if id is None:
        return None
    else:
        speaker, excerpt, sentence, nb_word = id.split("-")
        i = np.where(ids == id)[0][0]
        if i == len(ids) - 1:
            # last word
            return None
        else:
            next_id = ids[i + 1]
            next_speaker, next_excerpt, _, _ = next_id.split("-")
            if speaker != next_speaker or excerpt != next_excerpt:
                return None
            else:
                return next_id


def previous_word(id, ids):
    if id is None:
        return None
    else:
        speaker, excerpt, sentence, nb_word = id.split("-")
        i = np.where(ids == id)[0][0]
        if i == 0:
            # first word
            return None
        else:
            previous_id = ids[i - 1]
            previous_speaker, previous_excerpt, _, _ = previous_id.split("-")
            if speaker != previous_speaker or excerpt != previous_excerpt:
                return None
            else:
                return previous_id


for file in tqdm(os.listdir("memory/features")):
    y = []
    for id in context(file[:-4]):
        if id is not None:
            y.append(np.load(f'memory/features/{id}.npy'))
        else:
            y.append(np.zeros(SHAPE))
    np.save("memory/targets/" + file, np.concatenate(y))
