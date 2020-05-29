from torch.utils.data import Dataset
import numpy as np
import os


class AudioDataset(Dataset):
    def __init__(self, ids, context_size=3, path="memory/features/", shape=(127, 40)):
        self.path = path
        self.index2id = ids
        self.context_size = context_size
        self.shape = shape

    def __getitem__(self, index):
        file = self.index2id[index]
        x = np.load(f'{self.path}{file}.npy')
        y = []
        for id in self.context(self.index2id[index]):
            if id is not None:
                y.append(np.load(f'{self.path}{id}.npy'))
            else:
                y.append(np.zeros(self.shape))
        return x.astype('float32'), np.concatenate(y).astype('float32')

    def __len__(self):
        return len(self.index2id)

    def context(self, id):
        """Computes the target contexts for the word stored in the file id
        :param id: The id of the word whose context is to be computed, of the shape xxx-xxxx-xxxx-x
        For instance, 19-198-0000-5 corresponds to the word uttered by speaker 19,
        reading excerpt 198, in the sentence 0000, and is the sixth word in the sentence
        The context is constituted of the surrounding words (self.context_size words to the left, the same to the right)
        When at the boundary, fill with zeros
        """
        context = []
        last_id = id
        for i in range(self.context_size):
            context.append(previous_word(last_id, self.index2id))
            last_id = context[-1]
        context.reverse()

        last_id = id
        for i in range(self.context_size):
            context.append(next_word(last_id, self.index2id))
            last_id = context[-1]
        return context


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
            next_id = ids[i+1]
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
