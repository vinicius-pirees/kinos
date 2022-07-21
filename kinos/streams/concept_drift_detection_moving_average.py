import sys


from kinos.models.gaussian import Gaussian
from kinos.streams.producers import ImageProducer
from kinos.streams.consumers import ImageConsumer
import numpy as np


class ConceptDriftMovingAvg:
    def __init__(self, bootstrap_servers, source_topic, target_topic):
        self.consumer = ImageConsumer(bootstrap_servers, source_topic)
        self.producer = ImageProducer(bootstrap_servers, target_topic)
        self.current_mean = np.array([])
        self.current_variance = np.array([])
        self.current_std = np.array([])
        self.gaussian = Gaussian()
        self.max_cost = None
        self.min_cost = None
        
    def update_moving_measures(self, x, beta=0.998):
        if self.current_mean.size==0:
            self.current_mean = np.zeros(x.shape)
            self.current_variance = np.zeros(x.shape)
            self.current_std = np.zeros(x.shape)
            
        mean = (beta *  self.current_mean) +  ((1-beta) * x)
        self.current_mean = mean
        variance = (1 - beta) * (self.current_variance + np.power(beta * (x - mean), 2))
        self.current_variance = variance
        self.current_std = np.sqrt(variance)
        
    def get_measures(self):
        return self.current_mean, self.current_variance, self.current_std
    
    def evaluate_and_update(self, x, beta=0.998):
        result = self.gaussian.evaluate(x, self.current_mean, self.current_std)
        self.update_moving_measures(x, beta)
        
        return result
    
    
    def consume_features_and_predict(self, treshold=0):
        for msg in self.consumer.get_consumer():
            load_bytes = BytesIO(msg.value)
            feature = np.load(load_bytes, allow_pickle=True)
            
            result = self.evaluate_and_update(feature)
            
            if result > treshold:
                pass
                #Anomaly, if more than 2 minutes, then concept has drifted
                #producer.send(source_image)
                
    def update_max_min(self, value):
        if self.max_cost == None:
            self.max_cost = value
            self.min_cost = value
        else:
            if value < self.min_cost:
                self.min_cost = value
            
            if value > self.max_cost:
                self.max_cost = value
                
    def get_min_cost(self):
        return self.min_cost
    
    def get_max_cost(self):
        return self.max_cost
    
    def get_current_reg_score(self, value):
        sa = (value - self.min_cost) / (self.max_cost - self.min_cost)
        regularity_score = 1.0 - sa
        return regularity_score
    
    def reset_cost(self):
        self.max_cost = None
        self.min_cost = None
            