from kafka import KafkaProducer
from kafka.errors import KafkaError
import cv2
import numpy as np
from io import BytesIO


class ImageProducer:
    def __init__(self, bootstrap_servers, topic):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

    def send_frame(self, frame):
        np_bytes = BytesIO()
        np.save(np_bytes, frame, allow_pickle=True)
        np_bytes = np_bytes.getvalue()
        self.producer.send(self.topic, np_bytes)
        

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
