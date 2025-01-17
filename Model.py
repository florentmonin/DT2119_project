from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from DataProcessor import log
import numpy as np


class Speech2Vec(nn.Module):
    def __init__(self, nb_frame, nb_features, embedding_size):
        """ Constructs a new instance of the Speech2Vec model
        :param nb_frame:        corresponds to AUDIO_MAX_SIZE in DataProcessor: max size of the padded audios
        :type nb_frame:         int
        :param embedding_size:  Size of the embedding
        :type embedding_size:   int
        """

        super(Speech2Vec, self).__init__()
        self.SEQ_LENGTH = nb_frame
        self.EMBEDDING_SIZE = embedding_size
        self.NB_FEATURES = nb_features
        self.encoder = nn.LSTM(input_size=self.NB_FEATURES,
                               hidden_size=int(self.EMBEDDING_SIZE / 2),
                               bidirectional=True,
                               batch_first=True)

        self.decoder1 = nn.LSTM(input_size=self.EMBEDDING_SIZE,
                                hidden_size=self.NB_FEATURES,
                                batch_first=True)
        self.decoder2 = nn.LSTM(input_size=self.EMBEDDING_SIZE,
                                hidden_size=self.NB_FEATURES,
                                batch_first=True)
        self.decoder3 = nn.LSTM(input_size=self.EMBEDDING_SIZE,
                                hidden_size=self.NB_FEATURES,
                                batch_first=True)
        self.decoder4 = nn.LSTM(input_size=self.EMBEDDING_SIZE,
                                hidden_size=self.NB_FEATURES,
                                batch_first=True)
        self.decoder5 = nn.LSTM(input_size=self.EMBEDDING_SIZE,
                                hidden_size=self.NB_FEATURES,
                                batch_first=True)
        self.decoder6 = nn.LSTM(input_size=self.EMBEDDING_SIZE,
                                hidden_size=self.NB_FEATURES,
                                batch_first=True)
        self.decoders = [self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5, self.decoder6]

    def forward(self, X):
        """ Performs a forward pass inside the Speech2Vec
        :param X:   2D list of dimensionality (Batch_Size, self.INPUT_SIZE, N_features), already on CUDA
        :return     The current prediction of the model
        """

        _, h_n_c_n = self.encoder(X)
        enc_last_state, _ = h_n_c_n
        enc_last_state = enc_last_state.reshape(-1, 2 * int(self.EMBEDDING_SIZE / 2))

        all_outputs = []
        input_decoder = enc_last_state.repeat(self.SEQ_LENGTH, 1, 1).transpose(0, 1)

        for decoder in self.decoders:
            output_sequence, _ = decoder(input_decoder)
            all_outputs.append(output_sequence)

        return torch.cat(all_outputs, dim=1)

    def save(self, SAVE_PATH):
        """
        Saves the model
        :param SAVE_PATH:   model saving path
        """
        torch.save(self.state_dict(), SAVE_PATH + '.pyt')
        print(f"Model saved to {SAVE_PATH}.pyt.")

    def load(self, SAVE_PATH):
        """
        Loads the model
        :param SAVE_PATH:   model saving path
        """
        self.load_state_dict(torch.load(SAVE_PATH + '.pyt'))

    def generate(self, word):
        """ Generates the embedding of a word
        :param word:        Word to embed, 2D np.array of shape (self.SEQ_LENGTH, self.NB_FEATURES)
        :return:            Embedding of the word of the shape(self.EMBEDDING_SIZE,)
        """
        # noinspection PyArgumentList
        tens_word = torch.Tensor(word.reshape((1, self.SEQ_LENGTH, self.NB_FEATURES)))
        _, h_n_c_n = self.encoder(tens_word)
        enc_last_state, _ = h_n_c_n
        enc_last_state = enc_last_state.reshape(-1, 2 * int(self.EMBEDDING_SIZE / 2))
        return enc_last_state.detach().numpy()[0]


def stream(string, variables):
    """ Streams string to std.out
    :param string:          String to stream
    :param variables:       Variables to format in the string above
    """
    log(f'\r{string}' % variables, file="train.log")


def train(model, dataset, batch_size, learning_rate=0.001, nb_epochs=500):
    """ Trains the given model with the given dataset
    :param learning_rate:   Learning rate
    :param model:           Torch Speech2Vec model to be trained
    :param dataset:         Dataset to provide for the training
    :param batch_size:      Size of the batch
    :param nb_epochs:       Number of epochs
    :return:                Validation loss
    """
    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    step = 0

    val_losses = []
    train_losses = []
    avg_loss = 0

    N = len(dataset)
    train_size = int(N * 0.9)
    train_set, val_set = random_split(dataset, [train_size, N - train_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2, pin_memory=True)

    for epoch in range(nb_epochs):

        model.train(True)

        running_loss = 0.
        start = time.time()
        iters = len(train_loader)

        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()

            y_hat = model(X)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            speed = (i + 1) / (time.time() - start)
            avg_loss = running_loss / (i + 1)
            step += 1
            k = step // 1000
            stream('Epoch: %i/%i -- Batch: %i/%i -- Loss: %.6f -- %.2f steps/sec -- Step: %ik ',
                   (epoch + 1, nb_epochs, i + 1, iters, avg_loss, speed, k))

        model.train(False)

        val_loss = 0.
        avg_val_loss = 0.
        for i, (X, y) in enumerate(val_loader):
            X_val, y_val = X.cuda(), y.cuda()
            y_val_hat = model(X_val)
            val_loss += criterion(y_val_hat, y_val).item()
            avg_val_loss = val_loss / (i + 1)
        val_losses.append(avg_val_loss)
        train_losses.append(avg_loss)
        model.save('memory/models/model' + str(epoch + 1))
    model.save('memory/models/model')
    return val_losses, train_losses


def generate_embedding(model, w2a, ids, DATA_PATH, SAVE_PATH):
    """ Generates the embedding after training the model
    :param DATA_PATH:   Path where the features are saved
    :param model:       Instance of Speech2Vec
    :param w2a:         Word to list of utterances of this word mapping
    :param ids:         Maps ids to file names
    :param SAVE_PATH:   Path to save the embedding txt file
    """
    with open(SAVE_PATH, "w+") as saving_file:
        for word in tqdm(w2a.keys()):
            temp = []
            for id in w2a[word]:
                filename = ids[id]
                corr_audio = np.load(DATA_PATH + filename + '.npy', allow_pickle=True)
                corr_emb = model.generate(corr_audio)
                temp.append(corr_emb)
            embedding_word = np.mean(temp, axis=0)
            saving_file.write(word)
            for emb_value in embedding_word:
                saving_file.write(f" {emb_value}")
            saving_file.write('\n')
