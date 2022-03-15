import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics



#Ground Truth
#{"VIDEO": {"total_frames": total_frames_number, "anomalies": [{"start": start_frame_number}, {"end", end_frame_number}]}}

# Ex:
# {
# 	"VIDEO_1": {"total_frames": 2000, "anomalies" : [{"start": 10, "end": 15}, {"start": 250, "end": 256}]},
# 	"VIDEO_5": {"total_frames": 3000, "anomalies" : [{"start": 300, "end": 304}]}
#   "VIDEO_6": {"total_frames": 1000, "anomalies" : [{"start": 4, "end": 15}]}
# }

# Prediction
#{"VIDEO": {frame_number:confidence, frame_number:confidence}}

# Ex:
# {
# 	"VIDEO_1": {10:0.9, 11:0.9, 12:0.8, 13:0.54, 14:0.3, 250:0.98, 251:0.4, 258:0.3, 259:0.3},
# 	"VIDEO_5": {300:0.6, 301:0.55, 302:0.7, 303:0.4, 304:0.3}

# }




# {"start": 10, "end": 15}
def validate_gt_interval(interval):
    if interval['start'] >= interval['end']:
        raise ValueError("Start frame number can't be greater than the End one." + 
                         "Start: {}, End {}".format(interval['start'], interval['end']))



def get_model_scores_map(pred):
    """Creates a dictionary of from model_scores to frames numbers.
    Args:
        pred_boxes (dict): dict of frame and score
    Returns:
        dict: keys are model_scores and values
    """
    model_scores_map = {}

    for frame, score in pred.items():
        score = np.round(score, 3)
        if score not in model_scores_map.keys():
            model_scores_map[score] = [frame]
        else:
            model_scores_map[score].append(frame)
    return model_scores_map


def get_pred_frames_above_threshold(scores_map, chosen_threshold):
    pred_frames_above_threshold  = []
    for threshold, frames in scores_map.items():
        if threshold >= chosen_threshold:
            pred_frames_above_threshold = pred_frames_above_threshold + frames
    return pred_frames_above_threshold

def get_video_anomaly_frames(gts_video):
    anomaly_frames = set()

    for interval in gts_video['anomalies']:
        for frame_number in range(int(interval['start']), int(interval['end']) + 1):
            anomaly_frames.add(frame_number)
    return anomaly_frames


def get_video_normal_frames(gts_video, anomaly_frames):
    normal_frames = set()
    for frame_number in range(1, int(gts_video['total_frames']) + 1):
        if frame_number not in anomaly_frames:
            normal_frames.add(frame_number)
    return normal_frames




def get_video_results(preds, anomaly_frames, normal_frames):
    """Returns confusion matrix values for predictions of one video
    Args:
        preds (set): set of frame number predictions for a threshold
        anomaly_frames (set) set of frames numbers that are anomalies
        normal_frames (set) set of normal frame numbers
    Returns:
        dict: true positives (int), false positives (int), false negatives (int), true negatives (int)
    """
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for pred_frame in preds:
        if pred_frame in anomaly_frames:
            tp += 1
        else:
            fp += 1     
    
    for frame_number in anomaly_frames:
        if frame_number not in preds:
            fn += 1
            
    for frame_number in normal_frames:
        if frame_number not in preds:
            tn += 1
            
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}




def get_overall_result(gts, preds):
    results_dict = {}


    all_scores = []
    for video, pred_scores in preds.items():
        for frame, score in pred_scores.items():
            all_scores.append(score)


    for score in all_scores:
        score = np.round(score, 3)
        if results_dict.get(score) is None:
            results_dict[score] = {"tp":0,"fp":0,"fn":0,"tn":0}


    all_anormal_frames = []
    all_normal_frames = []
    all_scores_map = {}
    
    for video in gts.keys():
        anomaly_frames = get_video_anomaly_frames(gts[video])
        normal_frames = get_video_normal_frames(gts[video], anomaly_frames)


        all_anormal_frames += [video + '_' + str(x) for x in anomaly_frames]
        all_normal_frames += [video + '_' + str(x) for x in normal_frames]

        video_predictions = {}

        for key, value in preds.get(video).items():
            video_predictions[video + '_' + str(key)] = value

        scores_map = {}
        if video_predictions is not None:
            scores_map = get_model_scores_map(video_predictions)
            all_scores_map.update(scores_map)

    for score in results_dict.keys():
        if all_scores_map.get(score) is None:
            results_dict[score]['fn'] = results_dict[score]['fn']  + len(anomaly_frames)
            results_dict[score]['tn'] = results_dict[score]['tn'] + len(normal_frames)
        else:
            pred_frames  =  get_pred_frames_above_threshold(all_scores_map, score)
            video_results = get_video_results(pred_frames, all_anormal_frames, all_normal_frames)
            results_dict[score]['tp'] = results_dict[score]['tp'] + video_results['tp']
            results_dict[score]['fp'] = results_dict[score]['fp'] + video_results['fp']
            results_dict[score]['fn'] = results_dict[score]['fn'] + video_results['fn']
            results_dict[score]['tn'] = results_dict[score]['tn'] + video_results['tn']

    return results_dict



def tpr_fpr_points(results_dict):
    tpr_list = []
    fpr_list = []
    
    thresholds = sorted(results_dict.keys())

    for threshold in thresholds:
        tpr = results_dict[threshold]['tp'] / (results_dict[threshold]['tp'] + results_dict[threshold]['fn'])
        fpr = results_dict[threshold]['fp'] / (results_dict[threshold]['fp'] + results_dict[threshold]['tn'])

        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


def precision_recall_points(results_dict):
    recall_list = []
    precision_list = []
    
    thresholds = sorted(results_dict.keys())

    for threshold in thresholds:
        recall = results_dict[threshold]['tp'] / (results_dict[threshold]['tp'] + results_dict[threshold]['fn'])
        precision = results_dict[threshold]['tp'] / (results_dict[threshold]['tp'] + results_dict[threshold]['fp'])

        recall_list.append(recall)
        precision_list.append(precision)
    return recall_list, precision_list



def plot_roc(fpr_list, tpr_list):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_list, tpr_list, 'b', label = 'AUC = %0.2f' % metrics.auc(fpr_list, tpr_list))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_pr(recall_list, precision_list):
    plt.title('Precision Recall')
    plt.plot(recall_list, precision_list, label = None)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()