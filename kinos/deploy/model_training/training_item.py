class TrainingItem():
    def __init__(self, model):
        self.model = model 


    def notify(self):
        pass

    def update(self):
        pass

    def train(self):
        self.model.fit()
