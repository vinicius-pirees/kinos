import os
import sys
sys.path.append('/home/vinicius/git-projects/phaino')



from kafka import KafkaProducer
from kafka.errors import KafkaError
import cv2
import numpy as np
from io import BytesIO
import json
from datetime import datetime

from phaino.utils.commons import frame_to_bytes_str, frame_from_bytes_str



class ImageProducer:
    def __init__(self, bootstrap_servers, topic):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers, 
                                      value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                                      compression_type='gzip',
                                      batch_size=20000000,
                                      max_request_size=5048576)

    def send_frame(self, frame, extra_fields={}):
        dict_object = {}
        dict_object['timestamp'] = datetime.timestamp(datetime.now())
        dict_object['data'] =  frame_to_bytes_str(frame)
        
        for key, value in extra_fields.items():
            dict_object[key] = value

        self.producer.send(self.topic, dict_object)
        

class VideoProducer(ImageProducer):
    def __init__(self, bootstrap_servers, topic, video_path):
        super().__init__(bootstrap_servers, topic)
        self.video_path = video_path

    def send_video(self):
        cap = cv2.VideoCapture(self.video_path)

        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret==True:
                super().send_frame(frame)    
            else:
                break

        cap.release()

                
class WebCamProducer(ImageProducer):
    def __init__(self, bootstrap_servers, topic, camera_number):
        super().__init__(bootstrap_servers, topic)
        self.camera_number = camera_number

    def send_video(self):
        cap = cap = cv2.VideoCapture(self.camera_number)

        while(True):
            ret, frame = cap.read()

            if ret==True:
                super().send_frame(frame) 
            else:
                break   
                
        cap.release()
