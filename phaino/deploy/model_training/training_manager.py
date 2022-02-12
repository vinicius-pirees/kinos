import sys
import time
import logging
from multiprocessing import Process, Queue, Array, Manager
from phaino.deploy.model_training.model_selection import assign_models_priority



logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingManager():
    def __init__(self, models, user_constraints={}, model_queue=None):
        self.models = assign_models_priority(user_constraints, models)
        self.current_model = None
        self.model_queue = model_queue

        self.models_indexed_by_priority = {}
        for model in models:
            self.models_indexed_by_priority[model["priority"]] = model

        self.process_map = {}
        self.message_queue = Queue()
        self.n_models = len(models)


    def get_current_model(self):
        return self.current_model
        

    def handle_training(self, model, priority, message_queue, insufficient_capacity_list, training_data=None, training_data_name=None, model_list=None, load_models=False):
        
        logger.info(f"priority {priority}")

        if load_models:
            logger.info("Loading previously trained model")
        else:
            logger.info(f"priority {priority}")
            try:
                model.fit(training_data, training_data_name)
            except Exception as e: #TODO Consider only exceptions related to lack of computing capacity
                logger.info(e)
                if priority not in insufficient_capacity_list:
                    # Append to array
                    insufficient_capacity_list.append(priority)

                # Stop process and remove it from the process map, if map is filled
                if  self.process_map.get(priority) is not None:
                    self.process_map[priority].terminate()
                    self.process_map.pop(priority)
                return


        message_queue.put(priority) # Notify

        if model_list is not None:
            if hasattr(model, 'model'):
                model.model = None # Must set model as None due to multiprocessing (in order to pickle and unpickle later). Some objects are no pickable
            model_list.append(model)

        self.current_model = model # Switch to model

        logger.debug("insufficient_capacity_list " +  str(insufficient_capacity_list))
        if priority in insufficient_capacity_list:
                self.__remove_from_insufficient_list(priority, insufficient_capacity_list) 

        self.__attempt_train_insufficient(insufficient_capacity_list)


    def __remove_from_insufficient_list(self, priority, insufficient_capacity_list):
        insufficient_capacity_list.remove(priority)

    def __attempt_train_insufficient(self, insufficient_capacity_list):
        priorities = insufficient_capacity_list
        for priority in priorities:
            self.handle_training(self.models_indexed_by_priority[priority]['model'], priority, self.message_queue, insufficient_capacity_list)


    def __stop_lower_priority(self, priority):
            stop_priorities = list(range(priority+1, self.n_models))

            for priority in stop_priorities:
                process = self.process_map.get(priority)
                if process is not None:
                    model_info = self.models_indexed_by_priority[priority]
                    if process.is_alive():
                        logger.info(f'Stopping process {process.name}, model_info: {model_info}')
                        process.terminate()

    def stop_all_processes(self):
        for priority, process in self.process_map.items():
            if process is not None:
                model_info = self.models_indexed_by_priority[priority]
                if process.is_alive():
                    logger.info(f'Stopping process {process.name}, model_info: {model_info}')
                    process.terminate()


    def adapt(self, training_data=None, training_data_name=None, model_list=None, load_models=False):
        priorities = list(range(0, self.n_models))

        with Manager() as manager:
            insufficient_capacity_list = manager.list()

            for priority in priorities:
                model = self.models_indexed_by_priority[priority]
                model_class = model['model']
                p = Process(target=self.handle_training, args=(model_class, priority, self.message_queue,insufficient_capacity_list,training_data, training_data_name, model_list, load_models))
                p.start()
                self.process_map[priority] = p
                time.sleep(0.05) # Avoid processes to be launched out of order

        
            while True:
                priority = self.message_queue.get()
                time.sleep(1)

                logger.info(f"Finished priority {priority}")

                if priority == 0:
                    logger.info("Finishing training since the model with highest priority is ready")
                    #sys.exit(0)
                    self.__stop_lower_priority(0)
                    return 0
                else:
                    self.__stop_lower_priority(priority)


    
    def predict(self, example):
        """
        Uses the current model to make the inference
        """
        if self.current_model is None:
            logger.info('The training is not yet finished!')
        else:
            self.current_model.predict(example)


    def switch_to_model(self, model_name):
        for model_info in self.models:
            if model_info['name'] == model_name:
                self.current_model = model_info['model']
                
