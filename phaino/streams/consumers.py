from kafka import KafkaConsumer
from kafka.errors import KafkaError
import cv2
import numpy as np
from io import BytesIO



class ImageConsumer:
    def __init__(self, bootstrap_servers, topic):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer = KafkaConsumer(topic,
                                      bootstrap_servers=bootstrap_servers,
                                      auto_offset_reset='earliest',
                                      enable_auto_commit=True)

        
    def get_consumer(self):
        return self.consumer
    
    def get_one_image(self):
        for msg in self.consumer:
            load_bytes = BytesIO(msg.value)
            loaded_np = np.load(load_bytes, allow_pickle=True)
            return loaded_np
    
    
    
class ImageFiniteConsumer:

    def __init__(self, bootstrap_servers, topic):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer = KafkaConsumer(topic,
                                      bootstrap_servers=bootstrap_servers,
                                      consumer_timeout_ms=10000,
                                      auto_offset_reset='earliest',
                                      enable_auto_commit=True)
        
        
    def get_consumer(self):
        return self.consumer
    
    
    def get_one_image(self):
        for msg in self.consumer:
            load_bytes = BytesIO(msg.value)
            loaded_np = np.load(load_bytes, allow_pickle=True)
            return loaded_np
        
    def get_all_images(self):
        images = []
        for msg in self.consumer:
            load_bytes = BytesIO(msg.value)
            loaded_np = np.load(load_bytes, allow_pickle=True)

            images.append(loaded_np)
        return images
        