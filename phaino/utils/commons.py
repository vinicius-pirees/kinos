import cv2
import numpy as np


def frame_to_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def reduce_frame(frame):
    frame = frame/255
    return frame