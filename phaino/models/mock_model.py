import time
from tqdm import tqdm
from multiprocessing import Value


class InsufficientComputingCapacityException(Exception):
    pass

class MockModel():

    def __init__(self, epochs=60, epoch_time_seconds=1, insufficient_computing=False):
        self.epochs = epochs
        self.epoch_time_seconds = epoch_time_seconds
        self.insufficient_computing = insufficient_computing
    

    def fit(self):
        if not self.insufficient_computing:
            for i in tqdm(range(self.epochs)):
                time.sleep(self.epoch_time_seconds)
        else:
            raise InsufficientComputingCapacityException("Not enough cpu capacity")
            

    def predict(self):
        pass