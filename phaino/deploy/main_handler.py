import json
from kafka import KafkaConsumer
from tqdm import tqdm
from phaino.deploy.handler import Handler
from phaino.data_acquisition.training import TrainingDataAcquisition
from phaino.streams.consumers import ImageFiniteConsumer


from phaino.utils.commons import frame_from_bytes_str


class MainHandler():
    def __init__(
            self, 
            models, 
            user_constraints={},
            number_training_frames_after_drift=200,
            drift_algorithm=None,
            dimensionality_reduction=None,
            training_data_topic=None,
            is_initial_training_from_topic=True,
            initial_training_data=None,
            inference_data_topic=None,
            frame_dimension=(256,256)
            ):

        self.number_training_frames_after_drift = number_training_frames_after_drift
        self.handler = Handler(
            models=models,
            user_constraints=user_constraints,
            number_training_frames_after_drift=number_training_frames_after_drift,
            drift_algorithm=drift_algorithm,
            dimensionality_reduction=dimensionality_reduction,
            training_data_topic=training_data_topic,
            is_initial_training_from_topic=is_initial_training_from_topic,
            initial_training_data=initial_training_data,
            inference_data_topic=inference_data_topic,
            frame_dimension=frame_dimension
            )


    def start(self):
        while True:
            drift = self.handler.start()
            if drift:
                self.on_drift()

    


    def on_drift(self):
        #inject new data at training topic
        print("Acquiring new training data")
        training_frames_counter = 0
        with tqdm(total=self.number_training_frames_after_drift) as pbar:
            for msg in self.handler.inference_data_acquisition.consumer.consumer:
                if training_frames_counter >= self.number_training_frames_after_drift:
                    break
                self.handler.training_after_drift_producer.send_frame(frame_from_bytes_str(msg.value['data']))   
                #self.training_after_drift_producer.producer.send(self.training_after_drift_producer.topic,msg.value['data'])     
                self.handler.training_after_drift_producer.producer.flush()
                training_frames_counter+=1
                pbar.update(1)



        #Load the new training data
        print("Loading new training data")
        self.handler.training_data_acquirer = TrainingDataAcquisition(topic=self.handler.training_data_topic, group_id_suffix='training')
        self.handler.training_data_acquirer.load()

        #consumer = ImageFiniteConsumer("localhost:29092", "training_3")

        # consumer = KafkaConsumer("training_3",
        #                               bootstrap_servers="localhost:29092",
        #                               consumer_timeout_ms=5000,
        #                               auto_offset_reset='earliest',
        #                               #max_poll_interval_ms=5000,
        #                               enable_auto_commit=True,
        #                               group_id="test-train",
        #                               value_deserializer=lambda x: json.loads(x.decode('utf-8')))
        # videos = {}
        # vals=[]
        #for msg in consumer.consumer:
        #for msg in consumer:
        #    vals.append(msg.value['data'])


        print("New training data loaded")

        self.handler.reset()


