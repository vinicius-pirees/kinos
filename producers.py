from kafka import KafkaProducer
from kafka.errors import KafkaError



class BaseProducer:
	def __init__(self, bootstrap_servers, topic):
	    self.bootstrap_servers = bootstrap_servers
	    self.topic = topic



class VideoProducer(BaseProducer):
	def __init__(self, bootstrap_servers, topic, year):
	    super().__init__(bootstrap_servers, topic)
	    self.graduationyear = year