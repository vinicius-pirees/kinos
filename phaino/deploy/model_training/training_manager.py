import sys
import time
from multiprocessing import Process, Queue, Array
from phaino.deploy.model_training.model_selection import assign_models_priority



class TrainingManager():
    def __init__(self, models, user_constraints={}):
        self.models = assign_models_priority(user_constraints, models)
        self.current_model = None

        self.models_indexed_by_priority = {}
        for model in models:
            self.models_indexed_by_priority[model["priority"]] = model

        self.process_map = {}
        self.message_queue = Queue()
        self.n_models = len(models)
        self.insufficient_capacity_list = Array('i', [])


    def handle_training(self, model, priority, message_queue, training_data=None):
        print("priority", priority)
        try:
            model.fit()
        except Exception as e: #TODO Consider only exceptions related to lack of computing capacity
            print(e)
            if priority not in self.insufficient_capacity_list:
                # Append to array
                self.insufficient_capacity_list = Array('i', self.insufficient_capacity_list[:] + [priority])
                print(self.insufficient_capacity_list[:])

                # Stop process and remove it from the process map, if map is filled
                if  self.process_map.get(priority) is not None:
                    self.process_map[priority].terminate()
                    self.process_map.pop(priority)
                return

        message_queue.put(priority) # Notify
        self.current_model = model # Switch to model

        if priority in self.insufficient_capacity_list[:]:
                self.__remove_from_insufficient_list(priority) 

        self.__attempt_train_insufficient()


    def __remove_from_insufficient_list(self, priority):
        self.insufficient_capacity_list =  Array('i', [x for x in self.insufficient_capacity_list[:] if x != priority])

    def __attempt_train_insufficient(self):
        priorities = self.insufficient_capacity_list[:]
        for priority in priorities:
            print(priority)
            self.handle_training(self.models_indexed_by_priority[priority]['model'], priority, self.message_queue)


    def __stop_lower_priority(self, priority):
            stop_priorities = list(range(priority+1, self.n_models))

            for priority in stop_priorities:
                process = self.process_map.get(priority)
                if process is not None:
                    model_info = self.models_indexed_by_priority[priority]
                    if process.is_alive():
                        print(f'Stopping process {process.name}, model_info: {model_info}')
                        process.terminate()

    def adapt(self):
        priorities = list(range(0, self.n_models))

        for priority in priorities:
            model = self.models_indexed_by_priority[priority]
            model_class = model['model']
            p = Process(target=self.handle_training, args=(model_class, priority, self.message_queue,))
            p.start()
            self.process_map[priority] = p
            time.sleep(0.05) # Avoid processes to be launched out of order

    
        while True:
            priority = self.message_queue.get()
            time.sleep(1)

            print("Finished priority", priority)

            if priority == 0:
                print("Finishing training since the model with highest priority is ready")
                sys.exit(0)
            else:
                self.__stop_lower_priority(priority)


    
    def predict(self, example):
        """
        Uses the current model to make the inference
        """
        if self.current_model is None:
            print('The training is not yet finished!')
        else:
            self.current_model.predict(example)


    def switch_to_model(self, model_name):
        for model_info in self.models:
            if model_info['name'] == model_name:
                self.current_model = model_info['model']
                
