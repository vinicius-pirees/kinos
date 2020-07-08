import sys
sys.path.append('/home/vinicius/git-projects/phaino')


from producers import ImageProducer
from consumers import ImageConsumer
from phaino.utils.spatio_temporal_gradient_features import generate_features
from phaino.utils.commons import frame_to_gray, reduce_frame



class SpatioTemporalFeaturesProducer:
    def __init__(self, bootstrap_servers, source_topic, target_topic, cube_depth=5, tile_size=10):
        self.consumer = ImageConsumer(bootstrap_servers, source_topic)
        self.producer = ImageProducer(bootstrap_servers, target_topic)
        self.cube_depth = cube_depth
        self.tile_size = tile_size
          
    def send_frames(self):       
        counter = 1
        frame_sequence = []
        
        for msg in self.consumer.get_consumer():
            load_bytes = BytesIO(msg.value)
            frame = np.load(load_bytes, allow_pickle=True)
              
            if counter%self.cube_depth == 0:
                features = generate_features(frame_sequence, self.cube_depth, self.tile_size)
                result = features.reshape(1, features.shape[0]*features.shape[1])
                self.producer.send_frame(result) 
                # Reset frame sequence
                frame_sequence = []
                counter+=1
            else:
                # To grayscale
                frame = frame_to_gray(frame)
                # Divide pixels by 255
                reduced_frame = reduce_frame(frame)
                frame_sequence.append(reduced_frame)
                counter+=1
            

      
   
    




