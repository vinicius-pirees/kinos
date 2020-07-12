import cv2
import numpy as np
from io import BytesIO
import base64


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
