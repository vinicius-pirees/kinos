import time
import sys
import logging
from tqdm import tqdm
from multiprocessing import Process, Manager, Queue
from multiprocessing.managers import BaseManager
from phaino.deploy.model_training.training_manager import TrainingManager
from phaino.drift.detector import DriftDetector
from phaino.data_acquisition.training import TrainingDataAcquisition
from phaino.data_acquisition.inference import InferenceDataAcquisition
from phaino.streams.producers import ImageProducer
from phaino.utils.commons import frame_from_bytes_str


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
            frame_dimension=(256,256)
            ):
        self.models= models
        self.user_constraints = user_constraints
        self.sequence_size = 1
        self.training_data_topic = training_data_topic
        self.number_training_frames_after_drift = number_training_frames_after_drift
        self.inference_data_acquisition = InferenceDataAcquisition(topic=inference_data_topic)
        self.drift_detector = DriftDetector(dimensionality_reduction=dimensionality_reduction,
                                            drift_algorithm=drift_algorithm)
        self.model_queue = Queue()


        if is_initial_training_from_topic:
            self.training_data_acquirer = TrainingDataAcquisition(topic=self.training_data_topic, group_id_suffix='training')
            self.training_data_acquirer.load()
        else:
            self.training_data_acquirer = TrainingDataAcquisition()
            self.training_data_acquirer.load(input_data=initial_training_data)


        self.training_after_drift_producer = ImageProducer("localhost:29092", self.training_data_topic, resize_to_dimension=frame_dimension, debug=True)
                                            
        self.reset()



        

    def reset(self):
        self.training_manager = TrainingManager(self.models, self.user_constraints, self.model_queue)

        training_sequence = []
        if isinstance(self.training_data_acquirer.data, dict):
            for sequence in self.training_data_acquirer.data.values():
                training_sequence += sequence
        else:
            training_sequence = self.training_data_acquirer.data
       
        self.drift_detector.update_base_data(training_sequence)



    def start(self):

        with Manager() as manager:
            model_list = manager.list()

            p = Process(target=self.training_manager.adapt, args=(self.training_data_acquirer.data, 
                                                                    self.training_data_acquirer.train_name,
                                                                    model_list))
            p.start()

            logger.info("Waiting for an available model")

            sequence_counter = 0
            sequence = []

            while True:
                try:
                    model = model_list[-1]
                    model_list.pop()
                    
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
                        model_list.pop()
                        logger.info("Switching model")
                        if hasattr(model, 'sequence_size'):
                            self.sequence_size = model.sequence_size
                    except IndexError as e:
                        pass


                    data = frame_from_bytes_str(msg.value['data'])

                    if self.sequence_size > 1:
                        sequence.append(data)
                        sequence_counter+=1
                        if sequence_counter == self.sequence_size:
                            prediciton = model.predict(sequence) # TODO send to prediction topic?
                            print(prediciton)
                            sequence_counter = 0
                            sequence = []
                    else:
                        prediciton = model.predict(data)  # TODO send to prediction topic?

                    
                    time.sleep(1)

                    in_drift, drift_index = self.drift_detector.drift_check([data])
                    if in_drift:
                        logger.info("Drift detected")
                        #p.join()
                        p.terminate()
                        return True


                        


                        
          
