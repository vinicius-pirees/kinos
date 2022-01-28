import numpy as np
from sklearn.decomposition import PCA as Pca


class PCA():
    def __init__(self):
        self.algorithm = None

    def fit(self, training_data):
        temp_training_set = []
        
        if any(isinstance(el, list) for el in training_data): # if training set is a sequence of frames
            for sequence in training_data:
                temp_training_set += sequence
        else:
            temp_training_set = training_data

        final_training_set = np.array(temp_training_set)
        final_training_set = final_training_set/255 
        # Stack values for PCA
        final_training_set = final_training_set.reshape(final_training_set.shape[0], final_training_set.shape[1]* final_training_set.shape[2] * final_training_set.shape[3])


        self.algorithm = Pca(.95)
        self.algorithm.fit(final_training_set)


    def predict(self, examples):
        test_frames_data = np.array(examples)
        test_frames_data = test_frames_data/255
        test_frames_data = test_frames_data.reshape(test_frames_data.shape[0], test_frames_data.shape[1]* test_frames_data.shape[2] * test_frames_data.shape[3])
        frames_infer_transformed = self.algorithm.transform(test_frames_data)
        pca2_proj_back=self.algorithm.inverse_transform(frames_infer_transformed)
        reconstruction_errors = np.linalg.norm(np.subtract(test_frames_data,pca2_proj_back), axis=1)
        return reconstruction_errors
