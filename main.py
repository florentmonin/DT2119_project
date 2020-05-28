from DataProcessor import *
from Model import *
from AudioDataset import AudioDataset

EMBEDDING_SIZE = 50
FEATURE_DIM = 40

# data_processor = DataProcessor(r"segmentation", r"data\train-clean-500", "train_data")

data_processor = DataProcessor(r"light_segmentation", r"data/light_data", "train_data")
dataset = AudioDataset(data_processor.ids)

model = Speech2Vec(data_processor.AUDIO_MAX_SIZE, data_processor.feature_dim, EMBEDDING_SIZE).cuda()
val_losses, train_losses = train(model, dataset, batch_size=10, nb_epochs=1)
