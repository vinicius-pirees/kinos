
from sklearn import svm
import numpy as np
from sklearn.decomposition import PCA
import cv2
import os
import numpy as np
import pickle
import logging
import json
from numpy import inf
from sklearn.decomposition import PCA

from phaino.utils.spatio_temporal_gradient_features import generate_features_frames
from phaino.utils.commons import frame_to_gray, reduce_frame, resolve_model_path, get_last_model_path, get_clips

logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class OneClassSVM:
    def __init__(
            self, 
            kernel="rbf", 
            nu=0.5, 
            gamma='scale', 
            pca=False, 
            pca_n_components=150, 
            spatio_temporal_features=True,
            spatio_temporal_depth=5,
            spatio_temporal_sequence_size=10, 
            model_name=None):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model_name = model_name
        self.model = None
        self.pca = pca
        self.pca_n_components = pca_n_components
        self.spatio_temporal_features = spatio_temporal_features
        self.spatio_temporal_depth = spatio_temporal_depth
        self.sequence_size = spatio_temporal_depth
        self.spatio_temporal_sequence_size = spatio_temporal_sequence_size

        if model_name is None:
            self.model_name = 'oneclass_svm'
        else:
            self.model_name = model_name

        self.pca_set = None
        self.training_data_name=None


    def svm_model_frame(self, frame):
        frame = frame_to_gray(frame)
        reduced_frame = reduce_frame(frame)
        return reduced_frame
    
    def input_frames_transform(self, train_frames):
        input_frames = []
        for frame in range(0,train_frames.shape[0]):
            input_frames.append(self.svm_model_frame(train_frames[frame]))
        return np.array(input_frames)

    def fit(self, training_set, training_data_name=None):

        temp_training_set = []
        if training_data_name is not None:
            self.training_data_name = training_data_name

        if any(isinstance(el, list) for el in training_set): # if training set is a sequence of frames
            if self.spatio_temporal_features:
                logger.info('Generating spatio-temporal features')
                for sequence in training_set:
                    temp_training_set += generate_features_frames(sequence, cube_depth=self.spatio_temporal_depth, tile_size=self.spatio_temporal_sequence_size, description=self.model_name)
            else:
                for sequence in training_set:
                    temp_training_set += sequence

        else:
            if self.spatio_temporal_features:
                logger.info('Generating spatio-temporal features')
                temp_training_set = generate_features_frames(training_set, cube_depth=self.spatio_temporal_depth, tile_size=self.spatio_temporal_sequence_size, description=self.model_name)
            else:
                temp_training_set = training_set

        final_training_set = np.array(temp_training_set)
        final_training_set = final_training_set.reshape(final_training_set.shape[0], final_training_set.shape[2])


        if self.pca:
            logger.info('Generating PCA')
            self.pca_set = PCA(n_components=self.pca_n_components)
            self.pca_set.fit(final_training_set)
            final_training_set  = self.pca_set.transform(final_training_set)




        self.model = svm.OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.model.fit(final_training_set)
        self.save_model()

    def predict(self, sequence):

        if self.spatio_temporal_features:
            x = generate_features_frames(sequence, cube_depth=self.spatio_temporal_depth, tile_size=self.spatio_temporal_sequence_size)
            x = x[0]
        else:
            x = sequence

        if self.pca:
            x = self.pca_set.transform(x)


        predictions = self.model.score_samples(x)
        return predictions

    def save_model(self):
        logger.info(f"Saving the model {self.model_name}")
        path = resolve_model_path(self.model_name)
        with open(os.path.join(path, 'model.pkl'), 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)  

        if self.pca:
            with open(os.path.join(path, "pca.pkl"),"wb") as handle:
                pickle.dump(self.pca_set, handle, protocol=pickle.HIGHEST_PROTOCOL)


        #save metadata
        metadata = {
            "kernel": self.kernel,
            "nu": self.nu,
            "gamma": self.gamma,
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
            self.model = pickle.load(handle)
       
        with open(os.path.join(os.path.dirname(path), "metadata.json")) as infile:
            metadata = json.load(infile)
        
        self.training_data_name = metadata['training_data_name']
        
    def load_last_model(self):
        path = get_last_model_path(self.model_name)

        with open(os.path.join(path, "pca.pkl"),"rb") as handle:
            self.pca_set = pickle.load(handle)

        return self.load_model(os.path.join(path, 'model.pkl'))

