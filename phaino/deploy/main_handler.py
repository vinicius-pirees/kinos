import json
import sys
import time
from kafka import KafkaConsumer
from tqdm import tqdm
from phaino.deploy.handler import Handler
from phaino.data_acquisition.training import TrainingDataAcquisition
from phaino.data_acquisition.training import DataNotFoundException
from phaino.streams.consumers import ImageFiniteConsumer
import logging


logger = logging.getLogger('kafka')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


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
            initially_load_models=False
            ):


        #TODO: Validate parameters

        self.number_training_frames_after_drift = number_training_frames_after_drift
        self.read_retries = read_retries
        self.initially_load_models = initially_load_models
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
            initially_load_models=initially_load_models
            )


    def start(self):
        while True:
            drift = self.handler.start()
            if drift:
                self.on_drift()

    
    def on_drift(self):
        #inject new data at training topic
        print("Acquiring new training data")
        training_frames_counter = 0
        self.initially_load_models = False
        with tqdm(total=self.number_training_frames_after_drift) as pbar:
            for msg in self.handler.inference_data_acquisition.consumer.consumer:
                if training_frames_counter >= self.number_training_frames_after_drift:
                    break
                self.handler.training_after_drift_producer.send_frame(frame_from_bytes_str(msg.value['data']))   
                self.handler.training_after_drift_producer.producer.flush()
                training_frames_counter+=1
                pbar.update(1)
        self.handler.inference_data_acquisition.consumer.consumer.commit()

                
        time.sleep(10)
        #Load the new training data
        print("Loading new training data")
        self.handler.training_data_acquirer = TrainingDataAcquisition(topic=self.handler.training_data_topic, group_id_suffix='training')
        
        sucess_read = False
        for i in range(0, self.read_retries):
            try:
                self.handler.training_data_acquirer.load(load_last_saved=self.initially_load_models)
                sucess_read = True
                self.initially_load_models = False
                break
            except DataNotFoundException as e:
                print(e)
                print(f"Try {i} of {self.read_retries}")
                
        if not sucess_read:
            raise Exception("The data could not be loaded!")
            
        print("New training data loaded")

        self.handler.reset()


