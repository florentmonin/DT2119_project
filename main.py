from DataProcessor import *
import numpy as np
from Model import *
from AudioDataset import AudioDataset

EMBEDDING_SIZE = 50
FEATURE_DIM = 40

# data_processor = DataProcessor(r"segmentation", r"data\train-clean-500", "train_data")

data_processor = DataProcessor(r"light_segmentation", r"data/light_data", "train_data")
dataset = AudioDataset(data_processor.ids, shape=(data_processor.AUDIO_MAX_SIZE, FEATURE_DIM))
log(data_processor.AUDIO_MAX_SIZE)
log("Loading the model...")
model = Speech2Vec(data_processor.AUDIO_MAX_SIZE, data_processor.feature_dim, EMBEDDING_SIZE).cuda()
log("Training the model...")
val_losses, train_losses = train(model, dataset, batch_size=512, nb_epochs=500)
log("Saving the losses")
np.save('val_loss', val_losses)
np.save('train_loss', train_losses)
log("Done !")
