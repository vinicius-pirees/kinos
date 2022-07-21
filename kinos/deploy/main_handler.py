import json
from multiprocessing import Process, Queue
from threading import Thread
import sys
import time
from kafka import KafkaConsumer
from tqdm import tqdm
from kinos.config.config import KinosConfiguration
from kinos.deploy.handler import Handler
from kinos.data_acquisition.training import TrainingDataAcquisition
from kinos.data_acquisition.training import DataNotFoundException
from kinos.streams.consumers import ImageFiniteConsumer
import logging
from kinos.streams.producers import GenericProducer


logger = logging.getLogger('kafka')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

config = KinosConfiguration().get_config()
profile = config['general']['profile']
KAFKA_BROKER_LIST = config[profile]['kafka_broker_list']


from kinos.utils.commons import frame_from_bytes_str


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
            adapt_after_drift=True,
            provide_training_data_after_drift=False,
            drift_alert_topic='drift'
            ):


        #TODO: Validate parameters

        self.number_training_frames_after_drift = number_training_frames_after_drift
        self.read_retries = read_retries
        self.initially_load_models = initially_load_models
        self.detect_drift = detect_drift
        self.adapt_after_drift = adapt_after_drift
        self.prediction_result_topic = prediction_result_topic
        self.drift_alert_topic = drift_alert_topic
        self.prediction_queue = Queue()
        self.drift_alert_queue = Queue()
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
            drift_alert_queue=self.drift_alert_queue,
            adapt_after_drift=self.adapt_after_drift,
            provide_training_data_after_drift=provide_training_data_after_drift
            )


    def start(self):
        p_prediction = Process(target=self.prediction_process)
        p_prediction.start()

        p_drift_alert = Process(target=self.drift_alert_process)
        p_drift_alert.start()

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

    def drift_alert_process(self):
        drift_alert_producer = GenericProducer(KAFKA_BROKER_LIST, self.drift_alert_topic, debug=True)
        while True:
            drift_message = self.drift_alert_queue.get()
            drift_alert_producer.send(drift_message)

    
    


