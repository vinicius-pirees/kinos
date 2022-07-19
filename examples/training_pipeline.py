import os
from kinos.streams.producers import VideoProducer
from kinos.streams.consumers import ImageFiniteConsumer
from kinos.utils.commons import frame_from_bytes_str

from kinos.models.lstm_autoencoder import LSTMAutoEncoder
from kinos.models.gaussian import Gaussian
from kinos.config.config import PhainoConfiguration


config = PhainoConfiguration().get_config()
profile = config['general']['profile']
ADOC_DATASET_LOCATION = config[profile]['adoc_dataset_location']
KAFKA_BROKER_LIST = config[profile]['kafka_broker_list']


topic="training"
home_dir = '/home/viniciusgoncalves'
temp_dir =  os.path.join(home_dir,'temp/')
dataset_location = os.path.join(home_dir,'toy_dataset/adoc/')

video_files = os.listdir(dataset_location)


train_video_files = [x for x in video_files if x[0:5] == 'train']
train_video_files.sort()

#train_video_files = [train_video_files[1]] # Only one video
train_video_files = train_video_files[1:] # More than one video


for video in train_video_files:
    print(f"Publishing video {video}")
    video_producer = VideoProducer("localhost:29092", "training", os.path.join(dataset_location, video), debug=True, resize_to_dimension=(256,256))
    video_producer.send_video(extra_fields={"sequence_name": video})




# consumer = ImageFiniteConsumer(topic="inference_5", bootstrap_servers="localhost:29092")

# videos = {}
# for msg in consumer.consumer:
#     val = msg.value['data']
#     sequence_name = msg.value['sequence_name']
#     if videos.get(sequence_name) is None:
#         videos[sequence_name] = []

    
    
#     videos[sequence_name].append(frame_from_bytes_str(msg.value['data']))




#consumer = ImageFiniteConsumer(KAFKA_BROKER_LIST, "training_3")
#videos = {}
#for msg in consumer.consumer:
#    val = msg.value['data']



print('done')

# gaussian = Gaussian()
# lstm_autoencoder = LSTMAutoEncoder()



# project_name = 'experiment_1'




# models_list = [lstm_autoencoder, gaussian]