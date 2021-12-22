from multiprocessing import Process


class TrainingManager():
    def __init__(self, models):
        self.models = models


    def handle_training(self, model):
        model.fit()
        ##notify

    def adapt(self):
        process_list = []
        for model in self.models:
            model_class = model['model']
            #p = Process(target=model_class.fit, args=('bob',))
            p = Process(target=self.handle_training, args=(model_class, ))
            p.start()
            process_list.append(p)
            #p.join()

        for process in process_list:
            process.join()
