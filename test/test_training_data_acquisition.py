import unittest
import os
import time

from kinos.streams.producers import VideoProducer
from kinos.data_acquisition.training import TrainingDataAcquisition
from kinos.config.config import KinosConfiguration


config = KinosConfiguration().get_config()
profile = config['general']['profile']
# MODELS_DIRECTORY = config[profile]['directory']
# PROJECT_NAME = config['general']['project_name']
ADOC_DATASET_LOCATION = config[profile]['adoc_dataset_location']




class TestDriftDetection(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.broker = "localhost:29092"
        


    def test_dataacquisition_kafka(self):

        adoc_dataset_location = ADOC_DATASET_LOCATION

        video_files = os.listdir(adoc_dataset_location)

        train_video_files = [x for x in video_files if x[0:5] == 'train']
        train_video_files.sort()
        train_video_files = train_video_files[1:2] #not all videos for test

        for video in train_video_files:
            video_producer = VideoProducer(self.broker, "training", os.path.join(adoc_dataset_location, video), debug=True, resize_to_dimension=(256,256))
            video_producer.send_video(extra_fields={"sequence_name": video})

        time.sleep(5)    

        data_acquirer = TrainingDataAcquisition(topic='training')
        data_acquirer.load()

        self.assertEqual(len(data_acquirer.data), 1)

        self.assertEqual(len(data_acquirer.data['train_2.avi']), 200)

        print(data_acquirer.train_name)

        # Train more



    def test_dataacquisition_input(self):
        pass

        # Test with input

    @classmethod
    def tearDownClass(cls):
        pass

if __name__ == '__main__':
    unittest.main()