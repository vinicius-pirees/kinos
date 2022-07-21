from distutils.log import debug
import os
import pickle
from kinos.streams.consumers import ImageFiniteConsumer
from kinos.utils.commons import frame_from_bytes_str
from kinos.config.config import KinosConfiguration


config = KinosConfiguration().get_config()
profile = config['general']['profile']
project_name  = config['general']['project_name']
KAFKA_BROKER_LIST = config[profile]['kafka_broker_list']



class DataNotFoundException(Exception):
    pass
 

class TrainingDataAcquisition():
    """
    If input_data is defined, then the topic data is ignored

    input_data should be a dict of of sequence frames, where each key is the sequence name
        Ex: input_data = {
                'sequence_1': [[0, 2, 145, 200], [100, 200, 145, 200]],
                'sequence_2'  [[0, 0, 0, 200], [0, 0, 0, 200]]
            }
    """
    def __init__(self, topic=None, group_id_suffix=None, consumer_timeout_ms=5000):
        if topic is not None:
            self.consumer = ImageFiniteConsumer(KAFKA_BROKER_LIST, topic, group_id_suffix=group_id_suffix, consumer_timeout_ms=consumer_timeout_ms)

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


    def load(self, input_data=None, load_last_saved=False):
        if load_last_saved:
            last_saved_training_count = self.training_count - 1
            self.train_name = 'training_data_' + str(last_saved_training_count)
            path = os.path.join(config[profile]['directory'], project_name, "training_data", self.train_name)
            with open(os.path.join(path, "data.pkl"),"rb") as handle:
                self.data = pickle.load(handle)
        else:
            if input_data is None:
                sequences_data = {}
                for msg in self.consumer.consumer:
                    sequence_name = msg.value.get('sequence_name')

                    if sequence_name is None:
                        sequence_name = 'sequence'

                    if sequences_data.get(sequence_name) is None:
                        sequences_data[sequence_name] = []
                    
                    sequences_data[sequence_name].append(frame_from_bytes_str(msg.value['data']))
            else:
                sequences_data = input_data


            if isinstance(sequences_data, dict):
                self.data = []
                number_training_examples = 0
                for seq, value in sequences_data.items():
                    self.data.append(value)
                    number_training_examples += len(sequences_data[seq])

                num_training_sequences = len(self.data)
                
                    
            else:
                self.data = sequences_data
                num_training_sequences = 1
                number_training_examples = len(self.data)
                
            if num_training_sequences == 0:
                raise DataNotFoundException("There is not data to proceed with the training")
            else:
                print("Number of training examples:", number_training_examples)

            # Save the training data
            self.train_name = 'training_data_' + str(self.training_count)
            path = os.path.join(config[profile]['directory'], project_name, "training_data", self.train_name)
            os.makedirs(path, exist_ok=True)
            
            with open(os.path.join(path, "data.pkl"),"wb") as handle:
                pickle.dump(sequences_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.training_count+=1

