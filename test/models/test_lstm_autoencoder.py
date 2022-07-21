import unittest
from kinos.models.lstm_autoencoder import LSTMAutoEncoder
from sklearn.datasets import load_sample_images



class TestLSTMAutoEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        dataset = load_sample_images() 
        sequence_1 = [dataset.images[0] for x in range(20)]
        sequence_2 = [dataset.images[1] for x in range(20)]
        self.sequences = [sequence_1,sequence_2]


    def test_lstm(self):
        lstm = LSTMAutoEncoder(model_name='lstm_1', epochs=1)
        lstm.fit(self.sequences)
        lstm.save_model()


    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()