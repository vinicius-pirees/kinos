import os
import cv2
import unittest
from river.drift import PageHinkley
from kinos.deploy.handler import Handler
from kinos.deploy.main_handler import MainHandler
from kinos.drift.dimensionality_reduction.pca import PCA
from kinos.models.mock_model import MockModel
from kinos.streams.producers import VideoProducer
from sklearn.datasets import load_sample_images
from kinos.config.config import KinosConfiguration


config = KinosConfiguration().get_config()
profile = config['general']['profile']
ADOC_DATASET_LOCATION = config[profile]['adoc_dataset_location']
KAFKA_BROKER_LIST = config[profile]['kafka_broker_list']




class TestHandler(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.is_initial_training_from_topic = False    

        self.inference_data_topic = 'inference'
        self.prediction_result_topic = 'prediction'


        # Mock training data
        self.training_data_topic = None
        dataset = load_sample_images() 
        sequence_1 = [dataset.images[0] for x in range(20)]
        sequence_2 = [dataset.images[1] for x in range(20)]
        self.initial_training_data = sequence_1 + sequence_2

        for i in range(0, len(self.initial_training_data)):
            self.initial_training_data[i] = cv2.resize(self.initial_training_data[i], (256,256))
        

        
        # # Send training data
        self.training_data_topic = 'training'

        # adoc_dataset_location = ADOC_DATASET_LOCATION
        # video_files = os.listdir(adoc_dataset_location)
        # train_video_files = [x for x in video_files if x[0:5] == 'train']
        # train_video_files.sort()
        # train_video_files = train_video_files[1:2] # not all videos for test
        # for video in train_video_files:
        #     video_producer = VideoProducer("localhost:29092", self.training_data_topic, os.path.join(adoc_dataset_location, video), debug=True, resize_to_dimension=(256,256))
        #     video_producer.send_video(extra_fields={"sequence_name": video})




        self.user_constraints = {
            "is_real_time": False,
            "minimum_efectiveness": None
        }
        
        self.models = [
            {
                "name": "model_1",
                "training_rate": 200,
                "efectiveness": 30,
                "inference_rate": 10,
                "model":  MockModel(40, model_name= "model_1")
            },
            {
                "name": "model_2",
                "training_rate": 300,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel(30, model_name= "model_2")
            },
            {
                "name": "model_3",
                "training_rate": 400,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel(10, model_name= "model_3")
            }
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
            inference_data_topic=self.inference_data_topic
            )


    def test_main_handler(self):
        self.handler.start()


    def send_training_data(self):

        
        adoc_dataset_location = ADOC_DATASET_LOCATION
        video_files = os.listdir(adoc_dataset_location)

        train_video_files = [x for x in video_files if x[0:5] == 'train']
        train_video_files.sort()
        train_video_files = train_video_files[1:2] # not all videos for test

        for video in train_video_files:
            video_producer = VideoProducer(KAFKA_BROKER_LIST, self.training_data_topic, os.path.join(adoc_dataset_location, video), debug=True, resize_to_dimension=(256,256))
            video_producer.send_video(extra_fields={"sequence_name": video})

        

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()