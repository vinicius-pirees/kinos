import os
import sys

from kafka import KafkaProducer
from kafka.errors import KafkaError
import cv2
import numpy as np
from io import BytesIO
import json
from datetime import datetime

from kinos.utils.commons import frame_to_bytes_str, frame_from_bytes_str


#Todo: use GenericProducer as base class
class GenericProducer:
    def __init__(self, bootstrap_servers, topic, max_message_size_mb=8, debug=False):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.debug = debug
        max_message_size_bytes = max_message_size_mb * 1024 * 1024
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers, 
                                      value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                                      compression_type='gzip',
                                      batch_size=20000000,
                                      max_request_size=max_message_size_bytes)

    def send(self, dict_object):
        if self.debug:
             # Asynchronous by default
            future = self.producer.send(self.topic, dict_object)
            # Block for 'synchronous' sends
            record_metadata = future.get(timeout=10)   
        else:
            self.producer.send(self.topic, dict_object)


class ImageProducer:
    def __init__(self, bootstrap_servers, topic, max_message_size_mb=8,debug=False, resize_to_dimension=None):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.debug = debug
        self.resize_to_dimension = resize_to_dimension
        
        max_message_size_bytes = max_message_size_mb * 1024 * 1024
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers, 
                                      value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                                      compression_type='gzip',
                                      batch_size=20000000,
                                      max_request_size=max_message_size_bytes)

    def send_frame(self, frame, extra_fields={}):
        dict_object = {}
        dict_object['timestamp'] = datetime.timestamp(datetime.now())
        if self.resize_to_dimension is not None:
            frame = cv2.resize(frame, self.resize_to_dimension)
        dict_object['data'] =  frame_to_bytes_str(frame)
        
        for key, value in extra_fields.items():
            dict_object[key] = value
            
            
        if self.debug:
             # Asynchronous by default
            future = self.producer.send(self.topic, dict_object)
            # Block for 'synchronous' sends
            record_metadata = future.get(timeout=10)   
        else:
            self.producer.send(self.topic, dict_object)

        

class VideoProducer(ImageProducer):
    def __init__(self, bootstrap_servers, topic, video_path, debug=False, resize_to_dimension=None):
        super().__init__(bootstrap_servers, topic, debug=debug, resize_to_dimension=resize_to_dimension)
        self.video_path = video_path

    def send_video(self, extra_fields={}):
        cap = cv2.VideoCapture(self.video_path)

        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret==True:
                super().send_frame(frame, extra_fields=extra_fields)    
            else:
                break

        cap.release()

                
class CameraProducer(ImageProducer):
    def __init__(self, bootstrap_servers, topic, camera_number, debug=False, resize_to_dimension=None):
        super().__init__(bootstrap_servers, topic, debug=debug, resize_to_dimension=resize_to_dimension)
        self.camera_number = camera_number

    def send_video(self,  extra_fields={}):
        cap = cap = cv2.VideoCapture(self.camera_number)

        while(True):
            ret, frame = cap.read()

            if ret==True:
                super().send_frame(frame, extra_fields=extra_fields) 
            else:
                break   
                
        cap.release()
