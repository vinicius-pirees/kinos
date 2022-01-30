import time
import sys
from multiprocessing import Process, Manager, Queue
from multiprocessing.managers import BaseManager
from phaino.deploy.model_training.training_manager import TrainingManager
from phaino.drift.detector import DriftDetector
from phaino.data_acquisition.training import TrainingDataAcquisition
from phaino.data_acquisition.inference import InferenceDataAcquisition
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
            initial_training_data=None,
            inference_data_topic=None
            ):
        self.models= models
        self.user_constraints = user_constraints
        self.training_data_topic = training_data_topic
        self.number_training_frames_after_drift = number_training_frames_after_drift
        self.inference_data_acquisition = InferenceDataAcquisition(topic=inference_data_topic)
        self.drift_detector = DriftDetector(dimensionality_reduction=dimensionality_reduction,
                                            drift_algorithm=drift_algorithm)
        self.model_queue = Queue()
                                            
        self.reset(initial_training_data)



        

    def reset(self, training_data=None):



        self.training_manager = TrainingManager(self.models, self.user_constraints, self.model_queue)

        # BaseManager.register('TrainingManager', TrainingManager)
        # manager = BaseManager()
        # manager.start()
        # self.training_manager = manager.TrainingManager(self.models, self.user_constraints)

        if self.training_data_topic is not None:
            self.training_data_acquirer = TrainingDataAcquisition(topic=self.training_data_topic)
            self.training_data_acquirer.load()
        else:
            self.training_data_acquirer = TrainingDataAcquisition()
            self.training_data_acquirer.load(input_data=training_data)

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


            # Needs to run detached
            # self.training_manager.adapt(training_data=self.training_data_acquirer.data, 
            #                             training_data_name=self.training_data_acquirer.train_name)

            p = Process(target=self.training_manager.adapt, args=(self.training_data_acquirer.data, 
                                                                    self.training_data_acquirer.train_name,
                                                                    model_list))
            p.start()
        

            ## TODO: Issue self.training_manager.current_model is alwasy null. Probabily due to threading issues
            while True:

                print("Waiting for an available model")


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
                except Exception as e:
                    print("No model is available yet!")
                    time.sleep(10)
                    print(e)
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
                    except:
                        pass



                    data = frame_from_bytes_str(msg.value['data'])
                    # TODO send to prediction topic?
                    #prediciton = self.training_manager.get_current_model().predict(data)
                    prediciton = model.predict(data)

                    in_drift, drift_index = self.drift_detector.drift_check([data])
                    if in_drift:
                        print("Drift detected")
                        #inject new data at training topic
                        self.reset()
                        sys.exit(0)






        

        


