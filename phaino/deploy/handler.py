import time
import sys
from tqdm import tqdm
from multiprocessing import Process, Manager, Queue
from multiprocessing.managers import BaseManager
from phaino.deploy.model_training.training_manager import TrainingManager
from phaino.drift.detector import DriftDetector
from phaino.data_acquisition.training import TrainingDataAcquisition
from phaino.data_acquisition.inference import InferenceDataAcquisition
from phaino.streams.producers import ImageProducer
from phaino.utils.commons import frame_from_bytes_str



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
        self.training_data_topic = training_data_topic
        self.number_training_frames_after_drift = number_training_frames_after_drift
        self.inference_data_acquisition = InferenceDataAcquisition(topic=inference_data_topic)
        self.drift_detector = DriftDetector(dimensionality_reduction=dimensionality_reduction,
                                            drift_algorithm=drift_algorithm)
        self.model_queue = Queue()


        if is_initial_training_from_topic:
            self.training_data_acquirer = TrainingDataAcquisition(topic=self.training_data_topic)
            self.training_data_acquirer.load()
        else:
            self.training_data_acquirer = TrainingDataAcquisition()
            self.training_data_acquirer.load(input_data=initial_training_data)


        self.training_after_drift_producer = ImageProducer("localhost:29092", self.training_data_topic, resize_to_dimension=frame_dimension)

                                            
        self.reset()



        

    def reset(self):



        self.training_manager = TrainingManager(self.models, self.user_constraints, self.model_queue)

        # BaseManager.register('TrainingManager', TrainingManager)
        # manager = BaseManager()
        # manager.start()
        # self.training_manager = manager.TrainingManager(self.models, self.user_constraints)

        

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

            print("Waiting for an available model")

            while True:
                # model = self.model_queue.get()
                # print("Model ready")

                # if self.training_manager.get_current_model() is None:
                #     time.sleep(10)
                #     print("No model is available yet!")
                #     continue

                try:
                    model = model_list[-1]
                    model_list.pop()
                    #model_list = manager.list()
                except IndexError as e:
                    sleep_seconds = 5
                    print(f"No model is available yet, checking again in {sleep_seconds} seconds")
                    time.sleep(sleep_seconds)
                    continue

                

                for msg in self.inference_data_acquisition.consumer.consumer:

                    # new_model = self.model_queue.get_nowait()
                    # if new_model is not None:
                    #     model = new_model
                    #     print("Switching model")

                    # if not self.model_queue.empty:
                    #     model = self.model_queue.get()
                    #     print("Switching model")

                    try:
                        model = model_list[-1]
                        model_list.pop()
                        print("Switching model")
                    except IndexError as e:
                        pass



                    data = frame_from_bytes_str(msg.value['data'])
                    # TODO send to prediction topic?
                    prediciton = model.predict(data)
                    time.sleep(1)

                    in_drift, drift_index = self.drift_detector.drift_check([data])
                    if in_drift:
                        print("Drift detected")


                        #inject new data at training topic
                        print("Acquiring new training data")
                        training_frames_counter = 0
                        with tqdm(total=self.number_training_frames_after_drift) as pbar:
                            for msg in self.inference_data_acquisition.consumer.consumer:
                                if training_frames_counter >= self.number_training_frames_after_drift:
                                    break
                                #self.training_after_drift_producer.send_frame(frame_from_bytes_str(msg.value['data']))   
                                self.training_after_drift_producer.producer.send(self.training_after_drift_producer.topic ,msg.value)     
                                training_frames_counter+=1
                                pbar.update(1)


                        #Load the new training data
                        self.training_data_acquirer = TrainingDataAcquisition(topic=self.training_data_topic)
                        self.training_data_acquirer.load()
                        print("New training data loaded")

                        self.reset()
                        sys.exit(0)
