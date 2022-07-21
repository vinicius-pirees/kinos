import configparser
import os



class KinosConfiguration:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.path.dirname(__file__), 'kinos.conf'))
        
    def get_config(self):
        return self.config
        