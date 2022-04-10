import os
import sys

from kafka import KafkaConsumer
from kafka.errors import KafkaError
import cv2
import numpy as np
from io import BytesIO
import json
from phaino.config.config import PhainoConfiguration

config = PhainoConfiguration().get_config()
project_name  = config['general']['project_name']


from phaino.utils.commons import frame_to_bytes_str, frame_from_bytes_str

#Todo: use GenericConsumer as base class
class GenericConsumer:
    def __init__(self, bootstrap_servers, topic, finite=False, set_consumer_timeout_ms=None, group_id_suffix=None, enable_auto_commit=True, max_poll_records=50, max_poll_interval_ms=3000000):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        
        if finite:
            consumer_timeout_ms=1000
        else:
            if set_consumer_timeout_ms is None:
              consumer_timeout_ms=float('inf')
            else:
              consumer_timeout_ms=set_consumer_timeout_ms

        if group_id_suffix is None:
            group_id = project_name
        else:
            group_id = project_name + '-' + group_id_suffix

        self.consumer = KafkaConsumer(topic,
                                      bootstrap_servers=bootstrap_servers,
                                      auto_offset_reset='earliest',
                                      group_id=group_id,
                                      connections_max_idle_ms=12000,
                                      request_timeout_ms=11000,
                                      max_poll_records=max_poll_records,
                                      max_poll_interval_ms=max_poll_interval_ms,
                                      consumer_timeout_ms=consumer_timeout_ms,
                                      enable_auto_commit=enable_auto_commit,
                                      value_deserializer=lambda x: json.loads(x.decode('utf-8')))

        
    def get_consumer(self):
        return self.consumer
    
    

    
#Todo: Inherit from GenericConsumer
class ImageConsumer:
    def __init__(self, bootstrap_servers, topic, finite=False, group_id_suffix=None, enable_auto_commit=True,  max_poll_records=50, max_poll_interval_ms=3000000):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        
        if finite:
            consumer_timeout_ms=1000
        else:
            consumer_timeout_ms=float('inf')


        if group_id_suffix is None:
            group_id = project_name
        else:
            group_id = project_name + '-' + group_id_suffix
            
        self.consumer = KafkaConsumer(topic,
                                      bootstrap_servers=bootstrap_servers,
                                      auto_offset_reset='earliest',
                                      connections_max_idle_ms=12000,
                                      request_timeout_ms=11000,
                                      group_id=group_id,
                                      max_poll_records=max_poll_records,
                                      consumer_timeout_ms=consumer_timeout_ms,
                                      max_poll_interval_ms=max_poll_interval_ms,
                                      enable_auto_commit=enable_auto_commit,
                                      value_deserializer=lambda x: json.loads(x.decode('utf-8')))

        
    def get_consumer(self):
        return self.consumer
    
    def get_one_image(self):
        for msg in self.consumer:
            return frame_from_bytes_str(msg['data'])
    
    
    
class ImageFiniteConsumer:

    def __init__(self, bootstrap_servers, topic,  group_id_suffix=None, consumer_timeout_ms=5000, enable_auto_commit=True):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer_timeout_ms = consumer_timeout_ms

        if group_id_suffix is None:
            group_id = project_name
        else:
            group_id = project_name + '-' + group_id_suffix
        
        self.consumer = KafkaConsumer(topic,
                                      bootstrap_servers=bootstrap_servers,
                                      consumer_timeout_ms=consumer_timeout_ms,
                                      auto_offset_reset='earliest',
                                      enable_auto_commit=enable_auto_commit,
                                      group_id=group_id,
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
        