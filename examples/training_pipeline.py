import os
from phaino.streams.producers import VideoProducer
from phaino.streams.consumers import ImageFiniteConsumer
from phaino.utils.commons import frame_from_bytes_str

from phaino.models.lstm_autoencoder import LSTMAutoEncoder
from phaino.models.gaussian import Gaussian



# home_dir = '/home/viniciusgoncalves'
# temp_dir =  os.path.join(home_dir,'temp/')
# dataset_location = os.path.join(home_dir,'toy_dataset/adoc/')

# video_files = os.listdir(dataset_location)

# train_video_files = [x for x in video_files if x[0:5] == 'train']
# train_video_files.sort()

# for video in train_video_files:
#     video_producer = VideoProducer("localhost:29092", "training", os.path.join(dataset_location, video), debug=True, resize_to_dimension=(256,256))
#     video_producer.send_video(extra_fields={"sequence_name": video})

#topic="training_3"


# consumer = ImageFiniteConsumer(topic="inference_5", bootstrap_servers="localhost:29092")

# videos = {}
# for msg in consumer.consumer:
#     val = msg.value['data']
#     sequence_name = msg.value['sequence_name']
#     if videos.get(sequence_name) is None:
#         videos[sequence_name] = []

    
    
#     videos[sequence_name].append(frame_from_bytes_str(msg.value['data']))




consumer = ImageFiniteConsumer("localhost:29092", "training_3")
videos = {}
for msg in consumer.consumer:
    val = msg.value['data']



print('done')

# gaussian = Gaussian()
# lstm_autoencoder = LSTMAutoEncoder()



# project_name = 'experiment_1'




# models_list = [lstm_autoencoder, gaussian]