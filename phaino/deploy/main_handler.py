import json
from multiprocessing import Process, Queue
from threading import Thread
import sys
import time
from kafka import KafkaConsumer
from tqdm import tqdm
from phaino.config.config import PhainoConfiguration
from phaino.deploy.handler import Handler
from phaino.data_acquisition.training import TrainingDataAcquisition
from phaino.data_acquisition.training import DataNotFoundException
from phaino.streams.consumers import ImageFiniteConsumer
import logging
from phaino.streams.producers import GenericProducer


logger = logging.getLogger('kafka')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

config = PhainoConfiguration().get_config()
profile = config['general']['profile']
KAFKA_BROKER_LIST = config[profile]['kafka_broker_list']


from phaino.utils.commons import frame_from_bytes_str


class MainHandler():
    def __init__(
            self, 
            models, 
            user_constraints={},
            number_training_frames_after_drift=200,
            drift_algorithm=None,
            dimensionality_reduction=None,
            training_data_topic=None,
            is_initial_training_from_topic=True,
            initial_training_data=None,
            inference_data_topic=None,
            prediction_result_topic=None,
            frame_dimension=(256,256),
            read_retries=2,
            initially_load_models=False,
            detect_drift=True,
            adapt_after_drift=True
            ):


        #TODO: Validate parameters

        self.number_training_frames_after_drift = number_training_frames_after_drift
        self.read_retries = read_retries
        self.initially_load_models = initially_load_models
        self.detect_drift = detect_drift
        self.adapt_after_drift = adapt_after_drift
        self.prediction_result_topic = prediction_result_topic
        self.prediction_queue = Queue()
        self.handler = Handler(
            models=models,
            user_constraints=user_constraints,
            number_training_frames_after_drift=number_training_frames_after_drift,
            drift_algorithm=drift_algorithm,
            dimensionality_reduction=dimensionality_reduction,
            training_data_topic=training_data_topic,
            is_initial_training_from_topic=is_initial_training_from_topic,
            initial_training_data=initial_training_data,
            inference_data_topic=inference_data_topic,
            prediction_result_topic=prediction_result_topic,
            frame_dimension=frame_dimension,
            initially_load_models=initially_load_models,
            detect_drift=self.detect_drift,
            read_retries=read_retries,
            prediction_queue=self.prediction_queue,
            adapt_after_drift=self.adapt_after_drift
            )


    def start(self):
        p_prediction = Process(target=self.prediction_process)
        p_prediction.start()

        while True:
            p = Process(target=self.handler.start)
            p.start()
            p.join()

            if self.adapt_after_drift:
                self.handler.on_drift()
                p.terminate()



    def prediction_process(self):
        prediction_result_producer = GenericProducer(KAFKA_BROKER_LIST, self.prediction_result_topic, debug=True)
        while True:
            prediction_dict = self.prediction_queue.get()
            prediction_result_producer.send(prediction_dict)

    
    


