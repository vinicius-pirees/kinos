import configparser
import os



class PhainoConfiguration:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.path.dirname(__file__), 'phaino.conf'))
        
    def get_config(self):
        return self.config
        