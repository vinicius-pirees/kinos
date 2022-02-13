import time
import sys
import logging
import os
import psutil
from tqdm import tqdm
from multiprocessing import Process, Manager, Queue
from multiprocessing.managers import BaseManager
from phaino.deploy.model_training.training_manager import TrainingManager
from phaino.drift.detector import DriftDetector
from phaino.data_acquisition.training import TrainingDataAcquisition
from phaino.data_acquisition.inference import InferenceDataAcquisition
from phaino.streams.producers import ImageProducer, GenericProducer
from phaino.utils.commons import frame_from_bytes_str
from phaino.config.config import PhainoConfiguration
from kafka.errors import CommitFailedError



config = PhainoConfiguration().get_config()
profile = config['general']['profile']
KAFKA_BROKER_LIST = config[profile]['kafka_broker_list']


logging.basicConfig(level = logging.INFO, format='%(filename)s - %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class Handler():
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
            initially_load_models=False,
            detect_drift=True,
            adapt_after_drift=True
            ):
        self.models= models
        self.user_constraints = user_constraints
        self.sequence_size = 1
        self.training_data_topic = training_data_topic
        self.prediction_result_topic = prediction_result_topic
        self.number_training_frames_after_drift = number_training_frames_after_drift
        self.initially_load_models = initially_load_models
        self.detect_drift = detect_drift
        self.adapt_after_drift = adapt_after_drift
        self.inference_data_acquisition = InferenceDataAcquisition(topic=inference_data_topic, enable_auto_commit=False)
        self.drift_detector = DriftDetector(dimensionality_reduction=dimensionality_reduction,
                                            drift_algorithm=drift_algorithm)
        self.model_queue = Queue()


        if is_initial_training_from_topic:
            self.training_data_acquirer = TrainingDataAcquisition(topic=self.training_data_topic, group_id_suffix='training')
            self.training_data_acquirer.load()
        else:
            self.training_data_acquirer = TrainingDataAcquisition()
            self.training_data_acquirer.load(input_data=initial_training_data, load_last_saved=initially_load_models)


        self.training_after_drift_producer = ImageProducer(KAFKA_BROKER_LIST, self.training_data_topic, resize_to_dimension=frame_dimension, debug=True)
        self.prediction_result_producer = GenericProducer(KAFKA_BROKER_LIST, self.prediction_result_topic, debug=True)

                                            
        self.reset()


    def kill_child_proc(self, ppid):
        for process in psutil.process_iter():
            _ppid = process.ppid()
            if _ppid == ppid:
                _pid = process.pid
                if sys.platform == 'win32':
                    process.terminate()
                else:
                    os.system('kill -9 {0}'.format(_pid))
        

    def reset(self):
        self.training_manager = TrainingManager(self.models, self.user_constraints, self.model_queue)

        training_sequence = []
        if isinstance(self.training_data_acquirer.data, dict):
            for sequence in self.training_data_acquirer.data.values():
                training_sequence += sequence
        else:
            training_sequence = self.training_data_acquirer.data
       
        if self.detect_drift:
            self.drift_detector.update_base_data(training_sequence)



    def start(self):

        with Manager() as manager:
            model_list = manager.list()

            p = Process(target=self.training_manager.adapt, args=(self.training_data_acquirer.data, 
                                                                    self.training_data_acquirer.train_name,
                                                                    model_list, self.initially_load_models))
            p.start()

            self.initially_load_models = False

            logger.info("Waiting for an available model")

            sequence_counter = 0
            sequence = []
            model_name = None
            training_data_name = None
            sequence_name = None

            while True:
                try:
                    model = model_list[-1]
                    model.load_last_model()
                    model_list.pop()

                    if hasattr(model, 'model_name'):
                        model_name = model.model_name
                    if hasattr(model, 'training_data_name'):
                        training_data_name = model.training_data_name
                    if hasattr(model, 'sequence_size'):
                        self.sequence_size = model.sequence_size
                except IndexError as e:
                    sleep_seconds = 5
                    logger.info(f"No model is available yet, checking again in {sleep_seconds} seconds")
                    time.sleep(sleep_seconds)
                    continue

                
                for msg in self.inference_data_acquisition.consumer.consumer:    
                    try:
                        model = model_list[-1]
                        model.load_last_model()
                        model_list.pop()
                        logger.info("Switching model")
                        if hasattr(model, 'model_name'):
                            model_name = model.model_name
                        if hasattr(model, 'training_data_name'):
                            training_data_name = model.training_data_name
                        if hasattr(model, 'sequence_size'):
                            self.sequence_size = model.sequence_size
                    except IndexError as e:
                        pass
                    except FileNotFoundError as e:
                        print(f"The model {model.model_name} does not seem to exist yet")

                    

                    data = frame_from_bytes_str(msg.value['data'])


                    current_sequence_name = msg.value.get('sequence_name')

                    if current_sequence_name != sequence_name: # sequence changed
                        sequence_counter = 0
                        sequence = []

                    if current_sequence_name:
                        sequence_name = current_sequence_name

                    
                    frame_number = msg.value.get('frame_number') 

                    if self.sequence_size > 1:
                        sequence.append(data)
                        sequence_counter+=1
                        if sequence_counter == self.sequence_size:
                            prediction_dict = {}
                            prediciton = model.predict(sequence)

                            end_frame = frame_number
                            start_frame = frame_number - self.sequence_size + 1

                            prediction_dict['prediction'] = prediciton
                            prediction_dict['model_name'] = model_name
                            prediction_dict['training_data_name'] = training_data_name
                            prediction_dict['start_frame'] = start_frame
                            prediction_dict['end_frame'] = end_frame
                            prediction_dict['sequence_name'] = sequence_name
                            sequence_counter = 0
                            sequence = []
                            self.prediction_result_producer.send(prediction_dict)
                            try:
                                self.inference_data_acquisition.consumer.consumer.commit()
                            except CommitFailedError as e:
                                print(e)
                                return
                    else:
                        prediction_dict = {}
                        prediciton = model.predict(data)
                        prediction_dict['prediction'] = prediciton
                        prediction_dict['model_name'] = model_name
                        prediction_dict['training_data_name'] = training_data_name


                        self.prediction_result_producer.send(prediction_dict)
                        try:
                            self.inference_data_acquisition.consumer.consumer.commit()
                        except CommitFailedError as e:
                            print(e)
                            return
                    
                    time.sleep(1)

                    if self.detect_drift:
                        in_drift, drift_index = self.drift_detector.drift_check([data])
                        if in_drift:
                            logger.info("Drift detected")
                            if self.adapt_after_drift:
                                self.kill_child_proc(p.pid)
                                p.terminate()
                                os.system('kill {0}'.format(p.pid))
                                return True


                        


                        
          
