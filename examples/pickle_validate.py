import unittest
import os
import time
from phaino.data_acquisition.inference import InferenceDataAcquisition
from phaino.data_acquisition.training import TrainingDataAcquisition
from phaino.models.gaussian import Gaussian
from phaino.models.lstm_autoencoder import LSTMAutoEncoder

from phaino.streams.producers import VideoProducer
from phaino.utils.commons import frame_from_bytes_str
from phaino.config.config import PhainoConfiguration
import pickle


config = PhainoConfiguration().get_config()
profile = config['general']['profile']
ADOC_DATASET_LOCATION = config[profile]['adoc_dataset_location']

home = os.getenv("HOME")


lstm = LSTMAutoEncoder(model_name='lstm_1', epochs=1)


lstm.load_last_model()



with open(os.path.join(home, "lstm_class.pkl"),"wb") as handle:
    pickle.dump(lstm, handle, protocol=pickle.HIGHEST_PROTOCOL)

