import cv2
import os
import sys
import numpy as np


dataset_dir = '/home/vinicius/anomaly_datasets'



def ucsdped1_dataset():
	train_path = os.path.join(dataset_dir, 'UCSD_Anomaly_Dataset.v1p2','UCSDped1','Train')

	train_dirs = [x for x in os.listdir(train_path) if '.' not in x]
	train_dirs.sort()


	images = []
	for train_dir in train_dirs:
	    cur_train_dir = os.path.join(train_path,'Train001')
	    files =  [x for x in os.listdir(cur_train_dir) if 'DS' not in x]
	    files.sort()
	    
	    for file in files:
	        img = cv2.imread(os.path.join(cur_train_dir,file))
	        images.append(img)

	return np.array(images)

