import sys
import numpy as np
import cv2

from kinos.streams.producers import ImageProducer
from kinos.streams.consumers import ImageConsumer
from kinos.utils.spatio_temporal_gradient_features import generate_features
from kinos.utils.commons import frame_to_gray, reduce_frame

from kinos.utils.commons import frame_to_bytes_str, frame_from_bytes_str



class SpatioTemporalFeaturesProducer:
    def __init__(self, bootstrap_servers, source_topic, target_topic, cube_depth=5, tile_size=10):
        self.consumer = ImageConsumer(bootstrap_servers, source_topic)
        self.producer = ImageProducer(bootstrap_servers, target_topic)
        self.cube_depth = cube_depth
        self.tile_size = tile_size
        self.frame_counter = 0
          
    def send_frames(self):       
        counter = 0
        frame_sequence = []
        original_frame_sequence = []
      
        for msg in self.consumer.get_consumer():
            
            frame = frame_from_bytes_str(msg.value['data'])
            original_frame_sequence.append(frame)
            
              
            if counter%self.cube_depth == 0 and counter!=0:
                # To grayscale
                frame = frame_to_gray(frame)
                # Divide pixels by 255
                reduced_frame = reduce_frame(frame)
                frame_sequence.append(reduced_frame)
                
                
                features = generate_features(frame_sequence, self.cube_depth, self.tile_size)
                result = features.reshape(1, features.shape[0]*features.shape[1])
                
                
                ##Keep original frames
                #jpeg encode
                jpeg_frames = np.array([cv2.imencode('.jpg', x)[1] for x in original_frame_sequence])
                origin_frames = frame_to_bytes_str(jpeg_frames)
                
                
                end_frame_number = self.frame_counter
                source_end_timestamp = msg.value['timestamp']
                
                extra_fields = {'origin_frames':origin_frames, 
                                'start_frame_number': start_frame_number,
                                'end_frame_number': end_frame_number,
                                'source_end_timestamp': source_end_timestamp, 
                                'source_start_timestamp': source_start_timestamp}
                
                self.producer.send_frame(result, extra_fields=extra_fields) 
                
                start_frame_number = end_frame_number
                source_start_timestamp = source_end_timestamp
                # Reset frame sequences
                original_frame_sequence = []
                frame_sequence = []
                counter+=1
                self.frame_counter+=1
            else:
                if counter == 0:
                    start_frame_number = self.frame_counter
                    source_start_timestamp = msg.value['timestamp']
                    counter+=1
                    self.frame_counter+=1
                    continue
            
                # To grayscale
                frame = frame_to_gray(frame)
                # Divide pixels by 255
                reduced_frame = reduce_frame(frame)
                frame_sequence.append(reduced_frame)
                counter+=1
                self.frame_counter+=1
            

      
   
    




