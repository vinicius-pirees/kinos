
import numpy as np
import scipy
from numpy import inf
from phaino.utils.spatio_temporal_gradient_features import process_frames
from phaino.utils.commons import frame_to_gray, reduce_frame
import pickle

def measures(X):
        means = X.mean(axis=0)
        variances = np.mean((X - means) ** 2, axis=0 )
        stds = np.sqrt(variances)

        return means, variances, stds
    
    

class Gaussian:
    
    
    def __init__(self):
        self.means = None
        self.variances = None
        self.stds = None
    
    
    def update_model(self, X):
        self.means, self.variances, self.stds = measures(X)

    def get_measures(self):
        return self.means, self.variances, self.stds
    
    
    def evaluate(self, x):
        calc_gaussian = lambda x, mu, sigma: (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp( -(x - mu)**2 / (2 * sigma**2) )

        gaussian  = calc_gaussian(x, self.means, self.stds)
        #return np.sum(np.log(gaussian))

        # Test approach
        result = np.log(gaussian)
        
        
        result[result == -inf] = -10000000
        result[result == inf] = 10000000 
        result = np.nan_to_num(result)

        return np.sum(result) 
    
    
    def gaussian_model_frame(self, frame):
        frame = frame_to_gray(frame)
        reduced_frame = reduce_frame(frame)
        return reduced_frame
    
    def input_frames_transform(self, train_frames):
        input_frames = []
        for frame in range(0,train_frames.shape[0]):
            input_frames.append(gaussian_model_frame(train_frames[frame]))
        return np.array(input_frames)
    
    
    def frame_predictions(self, test_video, pred_list, clip_size=5, threshold=0):
        frame_predicitions_dict = {}
        video_len = test_video.shape[0]
        eval_index = 0
        cnt = 1
        for frame_num in range(0,video_len):   
            pred = pred_list[eval_index]


            if pred < threshold:
                detection = 1
            else:
                detection = 0

            frame_predicitions_dict[frame_num] = detection

            if cnt == clip_size:
                eval_index += 1
                cnt = 0
            cnt += 1

        return frame_predicitions_dict
    
    
    def frame_predictions_show(self, test_video, frame_predictions_dict): 
        pass

    
    def get_training_set(self, input_frames):
        return self.process_frames(input_frames)
    
    
    def use_last_model(self, path):
        model = self.load_model(path)
        self.means = model['means']
        self.variances = model['variances']
        self.stds = model['stds']
    
    def fit(self, training_set):
        self.update_model(training_set)
        self.save_model(self.means, self.variances, self.stds)
    
        
    def save_model(self, means, variances, stds):
        with open('gaussian_model_date.pickle', 'wb') as handle:
            pickle.dump({'means': means, 'variances': variances, 'stds': stds}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_model(self, path):
        with open('gaussian_model_date.pickle', 'rb') as handle:
            model = pickle.load(handle)
        return model
        
    