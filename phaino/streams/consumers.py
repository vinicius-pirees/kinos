import os
import sys
sys.path.append('/home/vinicius/git-projects/phaino')

from kafka import KafkaConsumer
from kafka.errors import KafkaError
import cv2
import numpy as np
from io import BytesIO
import json



from phaino.utils.commons import frame_to_bytes_str, frame_from_bytes_str

#Todo: use GenericConsumer as base class
class GenericConsumer:
    def __init__(self, bootstrap_servers, topic, finite=False):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        
        if finite:
            consumer_timeout_ms=1000
        else:
            consumer_timeout_ms=None
            
        self.consumer = KafkaConsumer(topic,
                                      bootstrap_servers=bootstrap_servers,
                                      auto_offset_reset='earliest',
                                      consumer_timeout_ms=consumer_timeout_ms,
                                      enable_auto_commit=True,
                                      value_deserializer=lambda x: json.loads(x.decode('utf-8')))

        
    def get_consumer(self):
        return self.consumer
    
    

    
#Todo: Inherit from GenericConsumer
class ImageConsumer:
    def __init__(self, bootstrap_servers, topic, finite=False):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        
        if finite:
            consumer_timeout_ms=1000
        else:
            consumer_timeout_ms=None
            
        self.consumer = KafkaConsumer(topic,
                                      bootstrap_servers=bootstrap_servers,
                                      auto_offset_reset='earliest',
                                      consumer_timeout_ms=consumer_timeout_ms,
                                      enable_auto_commit=True,
                                      value_deserializer=lambda x: json.loads(x.decode('utf-8')))

        
    def get_consumer(self):
        return self.consumer
    
    def get_one_image(self):
        for msg in self.consumer:
            return frame_from_bytes_str(msg['data'])
    
    
    
class ImageFiniteConsumer:

    def __init__(self, bootstrap_servers, topic):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer = KafkaConsumer(topic,
                                      bootstrap_servers=bootstrap_servers,
                                      consumer_timeout_ms=1000,
                                      auto_offset_reset='earliest',
                                      enable_auto_commit=True,
                                      value_deserializer=lambda x: json.loads(x.decode('utf-8')))
        
        
    def get_consumer(self):
        return self.consumer
    
    
    def get_one_image(self):
        for msg in self.consumer:
            return frame_from_bytes_str(msg['data'])
        
    def get_all_images(self):
        images = []
        for msg in self.consumer:
            image = frame_from_bytes_str(msg['data'])

            images.append(image)
        return images
        