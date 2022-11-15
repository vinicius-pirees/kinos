import time
import sys
import logging
import os
import psutil
from tqdm import tqdm
from multiprocessing import Process, Manager, Queue
from multiprocessing.managers import BaseManager
from kinos.deploy.model_training.training_manager import TrainingManager
from kinos.drift.detector import DriftDetector
from kinos.data_acquisition.training import TrainingDataAcquisition
from kinos.data_acquisition.inference import InferenceDataAcquisition
from kinos.streams.producers import ImageProducer, GenericProducer
from kinos.utils.commons import frame_from_bytes_str
from kinos.config.config import KinosConfiguration
from kinos.data_acquisition.training import DataNotFoundException
from kafka.errors import CommitFailedError
from numba import cuda


config = KinosConfiguration().get_config()
profile = config['general']['profile']
KAFKA_BROKER_LIST = config[profile]['kafka_broker_list']


logging.basicConfig(level = logging.INFO, format='%(filename)s - %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


prediction_queue = Queue()







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
            prediction_queue=None,
            drift_alert_queue=None,
            read_retries=2,
            adapt_after_drift=True,
            provide_training_data_after_drift=False            
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
        self.provide_training_data_after_drift = provide_training_data_after_drift
        self.read_retries = read_retries
        self.inference_data_topic = inference_data_topic
        self.inference_data_acquisition = InferenceDataAcquisition(topic=inference_data_topic, enable_auto_commit=False)
        self.drift_detector = DriftDetector(dimensionality_reduction=dimensionality_reduction,
                                            drift_algorithm=drift_algorithm)
        self.model_queue = Queue()
        self.prediction_queue = prediction_queue
        self.drift_alert_queue = drift_alert_queue


        if is_initial_training_from_topic:
            self.training_data_acquirer = TrainingDataAcquisition(topic=self.training_data_topic, group_id_suffix='training')
            self.training_data_acquirer.load()
        else:
            self.training_data_acquirer = TrainingDataAcquisition()
            self.training_data_acquirer.load(input_data=initial_training_data, load_last_saved=initially_load_models)


        self.training_after_drift_producer = ImageProducer(KAFKA_BROKER_LIST, self.training_data_topic, resize_to_dimension=frame_dimension, debug=True)
                                            
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


    def on_drift(self):
        # Inject new data at training topic

        if not hasattr(self.training_data_acquirer, 'consumer'):
            self.training_data_acquirer = TrainingDataAcquisition(topic=self.training_data_topic, group_id_suffix='training', consumer_timeout_ms=5000)

        if self.provide_training_data_after_drift:
            print("Waiting for training data")
            input("Press Enter when the data is finally loaded to the topic...")
            print("The process will resume when no new data is added after 5 seconds")
        else:
            print("Acquiring new training data")
            training_frames_counter = 0
            self.initially_load_models = False
            with tqdm(total=self.number_training_frames_after_drift) as pbar:
                for msg in self.inference_data_acquisition.consumer.consumer:
                    if training_frames_counter >= self.number_training_frames_after_drift:
                        break
                    self.training_after_drift_producer.send_frame(frame_from_bytes_str(msg.value['data']))   
                    self.training_after_drift_producer.producer.flush()
                    training_frames_counter+=1
                    pbar.update(1)
            self.inference_data_acquisition.consumer.consumer.commit()

                
            time.sleep(10)
            
        
        sucess_read = False
        for i in range(0, self.read_retries):
            try:
                #Load the new training data
                print("Loading new training data")
                self.training_data_acquirer.load(load_last_saved=self.initially_load_models)
                sucess_read = True
                self.initially_load_models = False
                break
            except DataNotFoundException as e:
                print(e)
                print(f"Try {i} of {self.read_retries}")
                
        if not sucess_read:
            raise Exception("The data could not be loaded!")
            
        print("New training data loaded")

        self.reset()





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
            adaptation = ""

            while True:

                try:
                    candidate_model = model_list[-1]

                    try:
                        if candidate_model.priority < model.priority:
                            model = candidate_model
                            model.load_last_model()
                            
                    except Exception as e: # model not defined yet
                        model = candidate_model

                    model_list.pop()

                    if hasattr(model, 'model_name'):
                        model_name = model.model_name
                        logger.info("Current model: " + model_name)
                    if hasattr(model, 'training_data_name'):
                        training_data_name = model.training_data_name
                    if hasattr(model, 'sequence_size'):
                        self.sequence_size = model.sequence_size


                    model_user_configuration = [x for x in self.models if  model_name == x.get("name")]
                    if len(model_user_configuration) != 0:
                        adaptation = model_user_configuration[0].get("adaptation")
                    else:
                        adaptation = ""


                except IndexError as e:
                    sleep_seconds = 5
                    logger.info(f"No model is available yet, checking again in {sleep_seconds} seconds")
                    time.sleep(sleep_seconds)
                    continue

                
                for msg in self.inference_data_acquisition.consumer.consumer:    
                    try:
                        candidate_model = model_list[-1]

                        try:
                            if candidate_model.priority < model.priority:
                                model = candidate_model
                                model.load_last_model()
                        except Exception as e: # model not defined yet
                            print(e)
                            model = candidate_model

                        model_list.pop()

                        logger.info("Switching model")
                        if hasattr(model, 'model_name'):
                            model_name = model.model_name
                            logger.info("Current model: " + model_name)
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

                            self.prediction_queue.put(prediction_dict)
                            
                            try:
                                self.inference_data_acquisition.consumer.consumer.commit()
                            except CommitFailedError as e:
                                print(e)
                                time.sleep(1)
                                continue
                    else:
                        prediction_dict = {}
                        prediciton = model.predict(data)
                        prediction_dict['prediction'] = prediciton
                        prediction_dict['model_name'] = model_name
                        prediction_dict['training_data_name'] = training_data_name

                        self.prediction_queue.put(prediction_dict)
                        try:
                            self.inference_data_acquisition.consumer.consumer.commit()
                        except CommitFailedError as e:
                            print(e)
                            return
                    
                    #time.sleep(1)

                    if self.detect_drift:
                        in_drift, drift_index = self.drift_detector.drift_check([data])
                        if in_drift:
                            logger.info("Drift detected")
                            if current_sequence_name is not None:
                                print(f"Drift at {current_sequence_name} frame {frame_number}")
                                self.drift_alert_queue.put({"at": current_sequence_name, "frame": frame_number})
                            
                            if adaptation == 'continuous':
                                print("Adaptation of the current model is continuous. Thus, retraining is not needed")
                                continue

                            if self.adapt_after_drift:
                                self.kill_child_proc(p.pid)
                                p.terminate()
                                os.system('kill {0}'.format(p.pid))

                                # Making sure to release the gpus, if it's the case
                                #if cuda.is_available():
                                    #for device in cuda.gpus:
                                        #device.reset()
                                        #cuda.select_device(device.id)
                                        #cuda.close()
                                return True


                        


                        
          
