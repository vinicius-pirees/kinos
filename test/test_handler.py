import os
import unittest
from river.drift import PageHinkley
from phaino.deploy.handler import Handler
from phaino.drift.dimensionality_reduction.pca import PCA
from phaino.models.mock_model import MockModel
from phaino.streams.producers import VideoProducer


class TestHandler(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.training_data_topic = 'training'
        self.inference_data_topic = 'inference'

        # Send training data
        home_dir = '/home/viniciusgoncalves'
        dataset_location = os.path.join(home_dir,'toy_dataset/adoc/')
        video_files = os.listdir(dataset_location)
        train_video_files = [x for x in video_files if x[0:5] == 'train']
        train_video_files.sort()
        train_video_files = train_video_files[1:2] # not all videos for test
        for video in train_video_files:
            video_producer = VideoProducer("localhost:29092", self.training_data_topic, os.path.join(dataset_location, video), debug=True, resize_to_dimension=(256,256))
            video_producer.send_video(extra_fields={"sequence_name": video})




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
                "model":  MockModel()
            },
            {
                "name": "model_2",
                "training_rate": 300,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel()
            },
            {
                "name": "model_3",
                "training_rate": 400,
                "efectiveness": 20,
                "inference_rate": 20,
                "model":  MockModel()
            }
        ]
        self.drift_algorithm = PageHinkley(min_instances=30, delta=0.005, threshold=80, alpha=1 - 0.01)
        self.dimensionality_reduction = PCA()
        self.number_training_frames_after_drift = 200
        


        self.handler = Handler(
            models=self.models,
            user_constraints=self.user_constraints,
            number_training_frames_after_drift=self.number_training_frames_after_drift,
            drift_algorithm=self.drift_algorithm,
            dimensionality_reduction=self.dimensionality_reduction,
            training_data_topic=self.training_data_topic,
            inference_data_topic=self.inference_data_topic
            )


    def test_main_handler(self):
        self.handler.start()

    def send_train_data(self):

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