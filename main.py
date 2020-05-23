import tensorflow_datasets as tfds
from tensorflow_datasets.core import download
import numpy as np

librispeech = tfds.load("librispeech", download_and_prepare_kwargs={"force_download": True} )

download_config = tfds.download.DownloadConfig(download_mode=download.GenerateMode.FORCE_REDOWNLOAD)