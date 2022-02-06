import time
from tqdm import tqdm
import logging
from multiprocessing import Value

logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)



class InsufficientComputingCapacityException(Exception):
    pass

class MockModel():

    def __init__(self, epochs=60, epoch_time_seconds=1, insufficient_computing=False, model_name='default'):
        self.epochs = epochs
        self.epoch_time_seconds = epoch_time_seconds
        self.insufficient_computing = insufficient_computing
        self.model_name = model_name
    

    def fit(self, training_data=None, training_data_name=None):
        if not self.insufficient_computing:
            for i in tqdm(range(self.epochs), desc=self.model_name):
                time.sleep(self.epoch_time_seconds)
        else:
            raise InsufficientComputingCapacityException("Not enough cpu capacity")
            

    def predict(self, examples):
        return 1