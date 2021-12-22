import time
from tqdm import tqdm

class MockModel():

    def __init__(self, epochs=60, epoch_time_seconds=1):
        self.epochs = epochs
        self.epoch_time_seconds = epoch_time_seconds

    def fit(self):
        for i in tqdm(range(self.epochs)):
            time.sleep(self.epoch_time_seconds)

    def predict(self):
        pass