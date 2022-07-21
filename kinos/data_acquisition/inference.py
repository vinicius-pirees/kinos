import os
import pickle
from kinos.streams.consumers import ImageConsumer
from kinos.utils.commons import frame_from_bytes_str
from kinos.config.config import KinosConfiguration

config = KinosConfiguration().get_config()
profile = config['general']['profile']
KAFKA_BROKER_LIST = config[profile]['kafka_broker_list']


class InferenceDataAcquisition():
    def __init__(self, topic, enable_auto_commit=True, group_id_suffix=None):
        self.consumer = ImageConsumer(KAFKA_BROKER_LIST, topic, enable_auto_commit=enable_auto_commit, group_id_suffix=group_id_suffix)


    # TODO: consume in batches
    def infer(self, model, transform_function=None, **kwargs):
        for msg in self.consumer.consumer:

            data = frame_from_bytes_str(msg.value['data'])
            data = transform_function(data, **kwargs)
            return model.predict(data)


            # TODO: Get info of sequence name?
            # sequence_name = msg.value['sequence_name']
            # if self.data.get(sequence_name) is None:
            #     self.data[sequence_name] = []
            
            # self.data[sequence_name].append(frame_from_bytes_str(msg.value['data']))

    def raw_data(self):
        for msg in self.consumer.consumer:
            return msg
    

