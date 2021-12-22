import sys
import numpy as np
import cv2

from phaino.streams.producers import ImageProducer
from phaino.streams.consumers import ImageConsumer
from phaino.utils.commons import frame_to_gray, reduce_frame

from phaino.utils.commons import frame_to_bytes_str, frame_from_bytes_str



class SequenceFramesFeaturesProducer:
    def __init__(self, bootstrap_servers, source_topic, target_topic, sequence_size, output_frame_size, to_grayscale=True, debug=False):
        self.consumer = ImageConsumer(bootstrap_servers, source_topic)
        self.producer = ImageProducer(bootstrap_servers, target_topic, debug=debug)
        self.sequence_size = sequence_size
        self.output_frame_size = output_frame_size
        self.to_grayscale = to_grayscale
        self.frame_counter = 0
      
                 
    def send_frames(self):       
        counter = 0
        frame_sequence = []
        original_frame_sequence = []
        frame_height, frame_width = self.output_frame_size
        
        
        for msg in self.consumer.get_consumer():
            frame = frame_from_bytes_str(msg.value['data'])
            original_frame_sequence.append(frame)
            
            
            if counter%self.sequence_size == 0 and counter!=0:
                # Resize
                frame = cv2.resize(frame, self.output_frame_size)
                # To grayscale
                if self.to_grayscale:
                    frame = frame_to_gray(frame)
                # Divide pixels by 255
                reduced_frame = reduce_frame(frame)
                
                frame_sequence.append(reduced_frame)
                
                
                #Todo: Think when image is RGB
                #Todo: Maybe possible with resize, reshape?
                clip = np.zeros(shape=(self.sequence_size, frame_height, frame_width, 1))
                clip[:, :, :, 0] = frame_sequence
                
                
                #Keep original frames
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
                
                
                #send to producer
                self.producer.send_frame(clip, extra_fields=extra_fields)
                
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

                # Resize
                frame = cv2.resize(frame, self.output_frame_size)
                # To grayscale
                if self.to_grayscale:
                    frame = frame_to_gray(frame)
                # Divide pixels by 255
                reduced_frame = reduce_frame(frame)
                
                frame_sequence.append(reduced_frame)
                counter+=1
                self.frame_counter+=1
                 
            