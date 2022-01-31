import os
import cv2
import unittest
from river.drift import PageHinkley
from phaino.deploy.handler import Handler
from phaino.drift.dimensionality_reduction.pca import PCA
from phaino.models.mock_model import MockModel
from phaino.streams.producers import VideoProducer
from sklearn.datasets import load_sample_images



class TestHandler(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.is_initial_training_from_topic = False    

        self.inference_data_topic = 'inference_5'


        # Mock training data
        self.training_data_topic = None
        dataset = load_sample_images() 
        sequence_1 = [dataset.images[0] for x in range(20)]
        sequence_2 = [dataset.images[1] for x in range(20)]
        self.initial_training_data = sequence_1 + sequence_2

        for i in range(0, len(self.initial_training_data)):
            self.initial_training_data[i] = cv2.resize(self.initial_training_data[i], (256,256))
        

        
        # # Send training data
        self.training_data_topic = 'training_2'

        # home_dir = '/home/viniciusgoncalves'
        # dataset_location = os.path.join(home_dir,'toy_dataset/adoc/')
        # video_files = os.listdir(dataset_location)
        # train_video_files = [x for x in video_files if x[0:5] == 'train']
        # train_video_files.sort()
        # train_video_files = train_video_files[1:2] # not all videos for test
        # for video in train_video_files:
        #     video_producer = VideoProducer("localhost:29092", self.training_data_topic, os.path.join(dataset_location, video), debug=True, resize_to_dimension=(256,256))
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
                "model":  MockModel(40)
            },
            {
                "name": "model_2",
                "training_rate": 300,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel(30)
            },
            {
                "name": "model_3",
                "training_rate": 400,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel(10)
            }
        ]
        self.drift_algorithm = PageHinkley(min_instances=20, delta=0.005, threshold=10, alpha=1 - 0.01)
        self.dimensionality_reduction = PCA()
        self.number_training_frames_after_drift = 10
        


        self.handler = Handler(
            models=self.models,
            user_constraints=self.user_constraints,
            number_training_frames_after_drift=self.number_training_frames_after_drift,
            drift_algorithm=self.drift_algorithm,
            dimensionality_reduction=self.dimensionality_reduction,
            training_data_topic=self.training_data_topic,
            is_initial_training_from_topic=self.is_initial_training_from_topic,
            initial_training_data=self.initial_training_data,
            inference_data_topic=self.inference_data_topic
            )


    def test_main_handler(self):
        self.handler.start()


    def send_training_data(self):

        home_dir = '/home/viniciusgoncalves'
        dataset_location = os.path.join(home_dir,'toy_dataset/adoc/')
        video_files = os.listdir(dataset_location)

        train_video_files = [x for x in video_files if x[0:5] == 'train']
        train_video_files.sort()
        train_video_files = train_video_files[1:2] # not all videos for test

        for video in train_video_files:
            video_producer = VideoProducer("localhost:29092", self.training_data_topic, os.path.join(dataset_location, video), debug=True, resize_to_dimension=(256,256))
            video_producer.send_video(extra_fields={"sequence_name": video})

        

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()