import cv2
import os
import sys
import numpy as np


def ucsdped1_train_dataset(dataset_dir, ucsd_version):
	train_path = os.path.join(dataset_dir, 'UCSD_Anomaly_Dataset.v1p2','UCSDped' + str(ucsd_version),'Train')

	train_dirs = [x for x in os.listdir(train_path) if '.' not in x]
	train_dirs.sort()


	images = []
	for train_dir in train_dirs:
	    cur_train_dir = os.path.join(train_path,train_dir)
	    files =  [x for x in os.listdir(cur_train_dir) if 'DS' not in x]
	    files.sort()
	    
	    for file in files:
	        img = cv2.imread(os.path.join(cur_train_dir,file))
	        images.append(img)

	return np.array(images)


def ucsdped1_test_dataset(dataset_dir, ucsd_version):
	test_path = os.path.join(dataset_dir, 'UCSD_Anomaly_Dataset.v1p2','UCSDped' + str(ucsd_version),'Test')

	test_dirs = [x for x in os.listdir(test_path) if '.' not in x and '_gt' not in x]
	test_dirs.sort()


	images = []
	for test_dir in test_dirs:
	    cur_test_dir = os.path.join(test_path,test_dir)
	    files =  [x for x in os.listdir(cur_test_dir) if 'DS' not in x]
	    files.sort()
	    
	    for file in files:
	        img = cv2.imread(os.path.join(cur_test_dir,file))
	        images.append(img)

	return np.array(images)

def ucsdped_ground_truths(gt_file, ucsd_version):
    train_path = os.path.join(dataset_dir, 'UCSD_Anomaly_Dataset.v1p2','UCSDped' + str(ucsd_version),'Test')
    gt_file =  os.path.join(train_path2, 'UCSDped'+ str(ucsd_version) + '.m')
    gt_dict = {}
    index=1
    f = open(gt_file, "r")
    for line in f:
        frames = re.search("\[(.*?)\]", line)
        if frames:
            intervals = frames.group(1).split(',')
            gt_dict[index] = []
            for interval in intervals:
                start_end = interval.strip().split(':')

                gt_dict[index].append({'start':start_end[0], 'end': start_end[1]})

            index+=1
    return gt_dict

