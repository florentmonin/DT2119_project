import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


class Speech2Vec(nn.Module):
    def __init__(self, nb_frame, nb_features, embedding_size, context_size=3):
        """
        Constructs a new instance of the Speech2Vec model
        :param nb_frame:        corresponds to AUDIO_MAX_SIZE in DataProcessor: max size of the padded audios
        :type nb_frame:         int
        :param embedding_size:  Size of the embedding
        :type embedding_size:   int
        :param context_size:    Size of the context
        :type context_size:     int
        """

        super(Speech2Vec, self).__init__()
        self.CONTEXT_SIZE = context_size
        self.SEQ_LENGTH = nb_frame
        self.EMBEDDING_SIZE = embedding_size
        self.NB_FEATURES = nb_features
        self.encoder = nn.LSTM(input_size=self.NB_FEATURES,
                               hidden_size=int(self.EMBEDDING_SIZE / 2),
                               bidirectional=True,
                               batch_first=True)
        self.decoders = []
        for _ in range(2 * self.CONTEXT_SIZE):
            self.decoders.append(nn.LSTM(input_size=self.EMBEDDING_SIZE,
                                         hidden_size=self.NB_FEATURES,
                                         batch_first=True))

    def forward(self, X):
        """
        Performs a forward pass inside the Speech2Vec
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

        return torch.cat(all_outputs)

    def save(self, SAVE_PATH):
        torch.save(self.state_dict(), SAVE_PATH)
        print(f"Model saved to {SAVE_PATH}.")


def stream(string, variables):
    sys.stdout.write(f'\r{string}' % variables)


def train(model, dataset, batch_size, learning_rate=0.001, nb_epochs=500):
    """
    TODO
    :param model:
    :param dataset:
    :param batch_size:
    :param nb_epochs:
    :return:
    """
    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    step = 0

    val_losses = []

    N = len(dataset)
    train_size = int(N * 0.9)
    train_set, val_set = random_split(dataset, [train_size, N - train_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=N - train_size, pin_memory=True)

    X_val, y_val = torch.Tensor(), torch.Tensor()
    for X, y in val_loader:
        X_val, y_val = X.cuda(), y.cuda()

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
            stream('Epoch: %i/%i -- Batch: %i/%i -- Loss: %.3f -- %.2f steps/sec -- Step: %ik ',
                   (epoch + 1, nb_epochs, i + 1, iters, avg_loss, speed, k))

        model.train(False)
        y_val_hat = model(X_val)
        val_loss = criterion(y_val_hat, y_val).item()
        val_losses.append(val_loss)
    model.save()
