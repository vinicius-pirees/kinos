import os
import cv2
import numpy as np
from io import BytesIO
import base64
from datetime import datetime
from confluent_kafka.admin import AdminClient, NewTopic
from phaino.config.config import PhainoConfiguration


config = PhainoConfiguration().get_config()
profile = config['general']['profile']
models_directory = config[profile]['directory']


def frame_to_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def reduce_frame(frame):
    frame = frame/255
    return frame


def frame_to_bytes_str(frame):
	np_bytes = BytesIO()
	np.save(np_bytes, frame, allow_pickle=True)
	np_bytes = np_bytes.getvalue()
	np_string = base64.b64encode(np_bytes).decode("utf-8")
	return np_string


def frame_from_bytes_str(frame_bytes_str):
	load_value = base64.b64decode(frame_bytes_str.encode("utf-8"))
	load_bytes = BytesIO(load_value)
	loaded_np = np.load(load_bytes, allow_pickle=True)
	return loaded_np 


def frame_to_bytes_str_simple(frame):
	return base64.b64encode(frame).decode('utf-8')

def frame_from_bytes_str_simple(frame_bytes_str, dtype_str, shape):
	return np.frombuffer(base64.b64decode(frame_bytes_str), dtype=dtype_str).reshape(shape)



def create_topic(bootstrap_servers, topic_name, retention_ms=604800000, segment_bytes=1073741824, max_message_bytes=50048576):
    a = AdminClient({'bootstrap.servers': ','.join(bootstrap_servers)})
    
    topics = []
    t = NewTopic(topic_name, num_partitions=1, replication_factor=1, config={'retention.ms': str(retention_ms),
                                                                             'segment.bytes': str(segment_bytes),
                                                                            'max.message.bytes': str(max_message_bytes)})
    topics.append(t)

    # Call create_topics to asynchronously create topics, a dict
    # of <topic,future> is returned.
    fs = a.create_topics(topics, request_timeout=10)

    # Wait for operation to finish.
    # Timeouts are preferably controlled by passing request_timeout=15.0
    # to the create_topics() call.
    # All futures will finish at the same time.
    for topic, f in fs.items():
        try:
            f.result()  # The result itself is None
            print("Topic {} created".format(topic))
        except Exception as e:
            print("Failed to create topic {}: {}".format(topic, e))


def init_model_dir(model_dir):
    os.makedirs(model_dir, exist_ok=True) 


def resolve_model_path(project_name, model_type, model_name):
    return os.path.join(models_directory, project_name, model_type, model_name, datetime.now().strftime("%Y%m%d_%H%M%S")) 


def get_list_of_models_in_path(project_name, model_type, model_name):
    models_path = os.path.join(models_directory, project_name, model_type, model_name)
    model_list = os.listdir(models_path) 
    if len(model_list) == 0:
        print('No model yet')
        return None
    else:
        model_list.sort()
        return model_list

def get_last_model_path(project_name, model_type, model_name):
        models_path = os.path.join(models_directory, project_name, model_type, model_name)
        model_list = get_list_of_models_in_path(project_name, model_type, model_name)
        if model_list is not None:
            last_model = model_list[-1]
            return os.path.join(models_path, last_model)


def get_first_model_path(project_name, model_type, model_name):
        models_path = os.path.join(models_directory, project_name, model_type, model_name)
        model_list = get_list_of_models_in_path(project_name, model_type, model_name)
        if model_list is not None:
            last_model = model_list[0]
            return os.path.join(models_path, last_model)




