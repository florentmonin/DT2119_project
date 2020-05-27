from DataProcessor import *
from Model import *
from AudioDataset import AudioDataset

EMBEDDING_SIZE = 50
FEATURE_DIM = 40

# data_processor = DataProcessor(r"light_segmentation", r"data/ligh_data", "train_data")
data_processor = DataProcessor(r"segmentation", r"data\train-clean-500", "train_data")
dataset = AudioDataset(data_processor.ids)

model = Speech2Vec(, FEATURE_DIM, EMBEDDING_SIZE).cuda()
train(model, dataset, batch_size=10)