import os
from phaino.streams.producers import VideoProducer, ImageProducer
from phaino.streams.consumers import ImageFiniteConsumer
from phaino.utils.commons import frame_from_bytes_str

from phaino.models.lstm_autoencoder import LSTMAutoEncoder
from phaino.models.gaussian import Gaussian
from sklearn.datasets import load_sample_images
from sklearn.datasets import load_digits


inference_data_topic = 'inference_3'

# # Real data
home_dir = '/home/viniciusgoncalves'
dataset_location = os.path.join(home_dir,'toy_dataset/adoc/')
video_files = os.listdir(dataset_location)
train_video_files = [x for x in video_files if x[0:5] == 'train']
train_video_files.sort()
train_video_files = train_video_files[2:3] # not all videos for test
for video in train_video_files:
    video_producer = VideoProducer("localhost:29092", inference_data_topic, os.path.join(dataset_location, video), debug=True, resize_to_dimension=(256,256))
    video_producer.send_video(extra_fields={"sequence_name": video})


# Mock data
# dataset = load_sample_images() 
# sequence_1 = [dataset.images[0] for x in range(20)]
# sequence_2 = [dataset.images[1] for x in range(20)]
# initial_training_data = sequence_1

# initial_training_data = load_digits()['data']





# image_producer = ImageProducer("localhost:29092", inference_data_topic, max_message_size_mb=8, debug=True, resize_to_dimension=(256,256))

# for frame in initial_training_data:
#     image_producer.send_frame(frame)


