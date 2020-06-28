import cv2
import os
import sys
import numpy as np


def ucsdped_train_dataset(dataset_dir, ucsd_version):
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


def ucsdped_test_dataset(dataset_dir, ucsd_version):
    test_path = os.path.join(dataset_dir, 'UCSD_Anomaly_Dataset.v1p2','UCSDped' + str(ucsd_version),'Test')

    test_dirs = [x for x in os.listdir(test_path) if '.' not in x and '_gt' not in x]
    test_dirs.sort()


    dataset_dict = {}
    for test_dir in test_dirs:
        images = []
        cur_test_dir = os.path.join(test_path,test_dir)
        files =  [x for x in os.listdir(cur_test_dir) if 'DS' not in x]
        files.sort()

        for file in files:
            img = cv2.imread(os.path.join(cur_test_dir,file))
            images.append(img)

        result = np.array(images)
        dataset_dict[test_dir] = result
        
    return dataset_dict

def ucsdped_test_video(dataset_dir, ucsd_version, video_number):
    test_path = os.path.join(dataset_dir, 'UCSD_Anomaly_Dataset.v1p2','UCSDped' + str(ucsd_version),'Test')
    video_number_str = str(video_number).zfill(3)
    
    video_name = 'Test' + video_number_str

    images = []

    test_dir = os.path.join(test_path,video_name)
    files =  [x for x in os.listdir(test_dir) if 'DS' not in x]
    files.sort()

    for file in files:
        img = cv2.imread(os.path.join(test_dir,file))
        images.append(img)

    return np.array(images)

def ucsdped_ground_truths(ucsd_version):
    train_path = os.path.join(dataset_dir, 'UCSD_Anomaly_Dataset.v1p2','UCSDped' + str(ucsd_version),'Test')
    gt_file =  os.path.join(train_path, 'UCSDped'+ str(ucsd_version) + '.m')
    gt_dict = {}
    index=1
    f = open(gt_file, "r")
    for line in f:
        frames = re.search("\[(.*?)\]", line)
        if frames:
            
            video_number_str = str(index).zfill(3)
            video_name = 'Test' + video_number_str
            intervals = frames.group(1).split(',')
            gt_dict[video_name] = []
            for interval in intervals:
                start_end = interval.strip().split(':')
                
                gt_dict[video_name].append({'start':start_end[0], 'end': start_end[1]})

            index+=1
    return gt_dict

