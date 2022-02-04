from distutils.log import debug
import os
import pickle
from phaino.streams.consumers import ImageFiniteConsumer
from phaino.utils.commons import frame_from_bytes_str
from phaino.config.config import PhainoConfiguration


config = PhainoConfiguration().get_config()
profile = config['general']['profile']
project_name  = config['general']['project_name']


class TrainingDataAcquisition():
    """
    If input_data is defined, then the topic data is ignored

    input_data should be a dict of of sequence frames, where each key is the sequence name
        Ex: input_data = {
                'sequence_1': [[0, 2, 145, 200], [100, 200, 145, 200]],
                'sequence_2'  [[0, 0, 0, 200], [0, 0, 0, 200]]
            }
    """
    def __init__(self, topic=None, group_id_suffix=None):
        if topic is not None:
            self.consumer = ImageFiniteConsumer("localhost:29092", topic, group_id_suffix=group_id_suffix)

        self.set_training_count()
        self.data = {}

    def set_training_count(self):
        training_data_dir =  os.path.join(config[profile]['directory'], project_name, "training_data")

        if not os.path.isdir(training_data_dir):
            self.training_count = 0
            return
            
        previous_trainings = os.listdir(training_data_dir)
        if len(previous_trainings) == 0:
            self.training_count = 0
        else:
            train_numbers = [int(x.split('training_data_')[1]) for x in previous_trainings if 'training_data' in x]
            train_numbers.sort()
            self.training_count = train_numbers[-1] + 1


    def load(self, input_data=None):
        if input_data is None:
            self.data = {}
            for msg in self.consumer.consumer:
                sequence_name = msg.value.get('sequence_name')

                if sequence_name is None:
                    sequence_name = 'sequence'

                if self.data.get(sequence_name) is None:
                    self.data[sequence_name] = []
                
                self.data[sequence_name].append(frame_from_bytes_str(msg.value['data']))
        else:
            self.data = input_data

        # Save the training data
        self.train_name = 'training_data_' + str(self.training_count)
        path = os.path.join(config[profile]['directory'], project_name, "training_data", self.train_name)
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, "data.pkl"),"wb") as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.training_count+=1

