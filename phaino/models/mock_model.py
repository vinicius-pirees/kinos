import time
from tqdm import tqdm


class InsufficientComputingCapacityException(Exception):
    pass

class MockModel():

    def __init__(self, epochs=60, epoch_time_seconds=1, insufficient_computing_retries=0):
        self.epochs = epochs
        self.epoch_time_seconds = epoch_time_seconds
        self.insufficient_computing_retries = insufficient_computing_retries
        self.insufficient_count = 0

    def fit(self):
        if self.insufficient_count >= self.insufficient_computing_retries:
            for i in tqdm(range(self.epochs)):
                time.sleep(self.epoch_time_seconds)
        else:
            self.insufficient_count+=1
            raise InsufficientComputingCapacityException()
            

    def predict(self):
        pass