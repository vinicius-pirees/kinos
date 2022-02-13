import os
import cv2
import unittest
from river.drift import PageHinkley
from phaino.deploy.handler import Handler
from phaino.deploy.main_handler import MainHandler
from phaino.drift.dimensionality_reduction.pca import PCA
from phaino.models.gaussian import Gaussian
from phaino.models.lstm_autoencoder import LSTMAutoEncoder
from phaino.models.mock_model import MockModel
from phaino.streams.producers import VideoProducer
from sklearn.datasets import load_sample_images
from phaino.config.config import PhainoConfiguration


config = PhainoConfiguration().get_config()
profile = config['general']['profile']
ADOC_DATASET_LOCATION = config[profile]['adoc_dataset_location']



class TestHandlerReal(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.is_initial_training_from_topic = True   
        self.initially_load_models  = False
        self.initial_training_data  = None


        self.detect_drift = True
        self.adapt_after_drift = False

        self.inference_data_topic = 'inference_5'
        self.prediction_result_topic = 'prediction'
        self.training_data_topic = 'training_3'

        
        # # Send training data
        if not self.initially_load_models:
            
            adoc_dataset_location = ADOC_DATASET_LOCATION
            video_files = os.listdir(adoc_dataset_location)
            train_video_files = [x for x in video_files if x[0:5] == 'train']
            train_video_files.sort()
            train_video_files = train_video_files[1:2] # not all videos for test
            for video in train_video_files:
                video_producer = VideoProducer("localhost:29092", self.training_data_topic, os.path.join(adoc_dataset_location, video), debug=True, resize_to_dimension=(256,256))
                video_producer.send_video(extra_fields={"sequence_name": video})




        self.user_constraints = {
            "is_real_time": False,
            "minimum_efectiveness": None
        }
        
        self.models = [
            {
                "name": "gaussian_1",
                "training_rate": 200,
                "efectiveness": 30,
                "inference_rate": 10,
                "model":  Gaussian(model_name='gaussian_1', pca=True, pca_n_components=.95)
            },
            {
                "name": "gaussian_2",
                "training_rate": 250,
                "efectiveness": 25,
                "inference_rate": 10,
                "model":  Gaussian(model_name='gaussian_2', pca=True, pca_n_components=.90)
            },
            {
                "name": "lstm_1",
                "training_rate": 30,
                "efectiveness": 60,
                "inference_rate": 3,
                "model":  LSTMAutoEncoder(model_name='lstm_1', epochs=1)
            },
            
            
           
        ]
        self.drift_algorithm = PageHinkley(min_instances=20, delta=0.005, threshold=10, alpha=1 - 0.01)
        self.dimensionality_reduction = PCA()
        self.number_training_frames_after_drift = 10
        


        self.handler = MainHandler(
            models=self.models,
            user_constraints=self.user_constraints,
            number_training_frames_after_drift=self.number_training_frames_after_drift,
            drift_algorithm=self.drift_algorithm,
            dimensionality_reduction=self.dimensionality_reduction,
            training_data_topic=self.training_data_topic,
            is_initial_training_from_topic=self.is_initial_training_from_topic,
            initial_training_data=self.initial_training_data,
            prediction_result_topic=self.prediction_result_topic,
            inference_data_topic=self.inference_data_topic,
            initially_load_models=self.initially_load_models,
            detect_drift=self.detect_drift,
            adapt_after_drift=self.adapt_after_drift
            )


    def test_main_handler(self):
        self.handler.start()
        

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()