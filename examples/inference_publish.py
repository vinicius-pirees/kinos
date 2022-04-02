import os
import cv2
from phaino.streams.producers import VideoProducer, ImageProducer
from phaino.streams.consumers import ImageFiniteConsumer
from phaino.utils.commons import frame_from_bytes_str

from phaino.models.lstm_autoencoder import LSTMAutoEncoder
from phaino.models.gaussian import Gaussian
from sklearn.datasets import load_sample_images
from sklearn.datasets import load_digits
from phaino.config.config import PhainoConfiguration


config = PhainoConfiguration().get_config()
profile = config['general']['profile']
ADOC_DATASET_LOCATION = config[profile]['adoc_dataset_location']
KAFKA_BROKER_LIST = config[profile]['kafka_broker_list']


inference_data_topic = 'inference'



#mode = 'mock'
mode = 'real'
mock_repeat = True


if mode == 'mock':

    num_frames = 1000

    # Mock data
    dataset = load_sample_images() 
    
    
    sequence_1 = [dataset.images[0] for x in range(num_frames)]
    sequence_2 = [dataset.images[1] for x in range(num_frames)]
    
    if mock_repeat:
        initial_training_data = sequence_1 + sequence_2
    else:
        initial_training_data = sequence_1


    image_producer = ImageProducer(KAFKA_BROKER_LIST, inference_data_topic, max_message_size_mb=8, debug=True, resize_to_dimension=(256,256))

    for i, frame in enumerate(initial_training_data):
        image_producer.send_frame(frame, extra_fields={"sequence_name": 'mock', "frame_number": i})



if mode == 'real':

    # Real data, two videos
    image_producer = ImageProducer(KAFKA_BROKER_LIST, inference_data_topic, max_message_size_mb=8, debug=True, resize_to_dimension=(256,256))

    #num_frames = 15
    #num_frames = 30
    num_frames = 500


    # video_files = ['test_drift_two_rainn_1.avi','train_1.avi']
    video_files = ['train_1.avi', 'test_drift_two_rainn_1.avi']

    video_files = [os.path.join(ADOC_DATASET_LOCATION, x) for x in video_files]

    for video in video_files:
        print("Sending", video)
        counter = 0
        cap = cv2.VideoCapture(video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True and counter<num_frames:
                image_producer.send_frame(frame, extra_fields={"sequence_name": video, "frame_number": counter})    
                counter+=1
            else:
                cap.release()
                break
            

    





