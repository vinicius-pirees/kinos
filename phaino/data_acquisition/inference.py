import os
import pickle
from phaino.streams.consumers import ImageConsumer
from phaino.utils.commons import frame_from_bytes_str


class InferenceDataAcquisition():
    def __init__(self, topic):
        self.consumer = ImageConsumer("localhost:29092", topic)


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
    

