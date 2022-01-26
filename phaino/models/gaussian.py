import os
import numpy as np
import pickle
import logging
import json
from numpy import inf
from sklearn.decomposition import PCA

from phaino.utils.spatio_temporal_gradient_features import generate_features_frames
from phaino.utils.commons import frame_to_gray, reduce_frame, resolve_model_path, get_last_model_path


logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)



class Gaussian:
    def __init__(
                    self, 
                    model_name=None, 
                    pca=False, 
                    pca_n_components=150,
                    spatio_temporal_features=True,
                    spatio_temporal_depth=5,
                    spatio_temporal_sequence_size=10                  
                ):
        self.means = None
        self.variances = None
        self.stds = None
        if model_name is None:
            self.model_name = 'gaussian'
        else:
            self.model_name = model_name
        self.pca = pca
        self.pca_n_components = pca_n_components
        self.spatio_temporal_features = spatio_temporal_features
        self.spatio_temporal_depth = spatio_temporal_depth
        self.spatio_temporal_sequence_size = spatio_temporal_sequence_size
        self.pca_set = None
        self.training_data_name=None


    def measures(self, X):
        means = X.mean(axis=0)
        variances = np.mean((X - means) ** 2, axis=0 )
        stds = np.sqrt(variances)

        return means, variances, stds
    
    
    def update_model(self, X):
        self.means, self.variances, self.stds = self.measures(X)

    def get_measures(self):
        return self.means, self.variances, self.stds
    
    
    def predict(self, x):
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
            input_frames.append(self.gaussian_model_frame(train_frames[frame]))
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
    
    def fit(self, training_set, training_data_name=None):
        temp_training_set = []
        if training_data_name is not None:
            self.training_data_name = training_data_name

        if any(isinstance(el, list) for el in training_set): # if training set is a sequence of frames
            if self.spatio_temporal_features:
                logger.info('Generating spatio-temporal features')
                for sequence in training_set:
                    temp_training_set += generate_features_frames(sequence, cube_depth=self.spatio_temporal_depth, tile_size=self.spatio_temporal_sequence_size)
            else:
                for sequence in training_set:
                    temp_training_set += sequence

        else:
            if self.spatio_temporal_features:
                logger.info('Generating spatio-temporal features')
                temp_training_set = generate_features_frames(training_set, cube_depth=self.spatio_temporal_depth, tile_size=self.spatio_temporal_sequence_size)
            else:
                temp_training_set = training_set

        final_training_set = np.array(temp_training_set)
        final_training_set = final_training_set.reshape(final_training_set.shape[0], final_training_set.shape[2])


        if self.pca:
            logger.info('Generating PCA')
            self.pca_set = PCA(n_components=self.pca_n_components)
            self.pca_set.fit(final_training_set)
            final_training_set  = self.pca_set.transform(final_training_set)


        self.update_model(final_training_set)
         
    def save_model(self):
        logger.info(f"Saving the model {self.model_name}")
        path = resolve_model_path(self.model_name)
        with open(os.path.join(path, 'model.pkl'), 'wb') as handle:
            pickle.dump({'means': self.means, 'variances': self.variances, 'stds': self.stds}, handle, protocol=pickle.HIGHEST_PROTOCOL)  

        if self.pca:
            with open(os.path.join(path, "pca.pkl"),"wb") as handle:
                pickle.dump(self.pca_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #save metadata
        metadata = {
            "pca": self.pca, 
            "pca_n_components": self.pca_n_components,
            "spatio_temporal_features": self.spatio_temporal_features,
            "spatio_temporal_depth": self.spatio_temporal_depth,
            "spatio_temporal_sequence_size": self.spatio_temporal_sequence_size,
            "training_data_name": self.training_data_name
        }

        with open(os.path.join(path, "metadata.json"), "w") as outfile:
            json.dump(metadata, outfile)

        
    def load_model(self, path):
        with open(path, 'rb') as handle:
            model = pickle.load(handle)
        self.means = model['means']
        self.variances = model['variances']
        self.stds = model['stds']
        
    def load_last_model(self):
        path = get_last_model_path(self.model_name)
        self.pca_set = pickle.load(open(os.path.join(path, "pca.pkl"),'rb'))
        return self.load_model(os.path.join(path, 'model.pkl'))
    