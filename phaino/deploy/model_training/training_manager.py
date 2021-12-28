from multiprocessing import Process, Event, Queue
import sys
import time


class TrainingManager():
    def __init__(self, models):
        self.models = models

        self.models_indexed_by_priority = {}
        for model in models:
            self.models_indexed_by_priority[model["priority"]] = model

        self.process_map = {}
        self.message_queue = Queue()
        self.n_models = len(models)


    def handle_training(self, model, priority, message_queue, training_data=None):
        print("priority", priority)
        model.fit()
        message_queue.put(priority)
        #TODO Switch to use the new model for inference at the main class


    def __stop_lower_priority(self, priority):
            stop_priorities = list(range(priority+1, self.n_models))

            for priority in stop_priorities:
                process = self.process_map[priority]
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
                
