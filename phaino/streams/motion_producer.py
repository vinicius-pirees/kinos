from phaino.utils.motion_detection import MotionDetection
from producers import ImageProducer



class MotionProducer:
    def __init__(self, bootstrap_servers, topic, min_movement_area=40, weight=0.6):
        self.motion_detector = MotionDetection(min_movement_area, weight)
        self.producer = ImageProducer(bootstrap_servers, topic)
        
    
    def send_frame(self, frame):
        if self.motion_detector.is_current_frame_moving(frame):
            self.producer.send_frame(frame)




