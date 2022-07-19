import unittest
import os
import time
from kinos.data_acquisition.inference import InferenceDataAcquisition
from kinos.data_acquisition.training import TrainingDataAcquisition
from kinos.models.gaussian import Gaussian

from kinos.streams.producers import VideoProducer
from kinos.utils.commons import frame_from_bytes_str
from kinos.config.config import PhainoConfiguration


config = PhainoConfiguration().get_config()
profile = config['general']['profile']
ADOC_DATASET_LOCATION = config[profile]['adoc_dataset_location']



class TestGaussianPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        

        self.is_initial_training_from_topic = True   
        self.initial_training_data  = None

        self.inference_data_topic = 'inference'
        self.inference_data_acquisition = InferenceDataAcquisition(topic=self.inference_data_topic)

        
        # # Send training data
        self.training_data_topic = 'training'

        adoc_dataset_location = ADOC_DATASET_LOCATION
        video_files = os.listdir(adoc_dataset_location)
        train_video_files = [x for x in video_files if x[0:5] == 'train']
        train_video_files.sort()
        train_video_files = train_video_files[1:2] # not all videos for test
        for video in train_video_files:
            video_producer = VideoProducer("localhost:29092", self.training_data_topic, os.path.join(adoc_dataset_location, video), debug=True, resize_to_dimension=(256,256))
            video_producer.send_video(extra_fields={"sequence_name": video})


        self.training_data_acquirer = TrainingDataAcquisition(topic=self.training_data_topic, group_id_suffix="training")
        self.training_data_acquirer.load()


        self.training_data_acquirer.data
        self.training_data_acquirer.train_name


    def test_gaussian_with_pca(self):
        gaussian = Gaussian(model_name='gaussian_1', pca=True, pca_n_components=.95)
        gaussian.fit(self.training_data_acquirer.data)
        gaussian.save_model()
        #gaussian.load_last_model()

        sequence_size = 5
        sequence_counter = 0
        sequence= []

        for msg in self.inference_data_acquisition.consumer.consumer:
            data = frame_from_bytes_str(msg.value['data'])
            sequence.append(data)
            sequence_counter+=1
            
            if sequence_counter == sequence_size:
                # TODO send to prediction topic?
                prediciton = gaussian.predict(sequence)
                print(prediciton)
                sequence_counter = 0
                sequence = []
                


    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()