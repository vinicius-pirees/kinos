import keras
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from keras.models import Sequential, load_model
from keras_layer_normalization import LayerNormalization
import os
import sys
import cv2
from matplotlib import pyplot as plt
import time
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import tensorflow as tf
from phaino.utils.commons import frame_to_gray, reduce_frame


from datetime import datetime

models_dir = '/home/vinicius/phaino_models'


class LSTMAutoEncoder:
    
    def __init__(self, batch_size=1, epochs=2, sequence_size=10):
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_size = sequence_size
        self.model_dir = os.path.join(models_dir, 'lstm_autoencoder')
        self.model = None
        self.max_cost = None
        self.min_cost = None
        
    def get_model_in_use(self):
        return self.model
        
        
    def lstm_autoencoder_frame(frame, input_size):
        frame = cv2.resize(frame, input_size)
        frame = frame_to_gray(frame)
        reduced_frame = reduce_frame(frame)
        return reduced_frame


    def input_frames_transform(train_frames):
        input_frames = []
        for frame in range(0,train_frames.shape[0]):
            input_frames.append(lstm_autoencoder_frame(train_frames[frame], (256,256)))
            
        return input_frames
    
    
    
    def get_clips_by_stride(stride, frames_list, sequence_size):
        """ For data augmenting purposes.
        Parameters
        ----------
        stride : int
            The distance between two consecutive frames
        frames_list : list
            A list of sorted frames of shape 256 X 256
        sequence_size: int
            The size of the lstm sequence
        Returns
        -------
        list
            A list of clips , sequence_size frames each
        """
        clips = []
        sz = len(frames_list)
        clip = np.zeros(shape=(sequence_size, 256, 256, 1))
        cnt = 0
        for start in range(0, stride):
            for i in range(start, sz, stride):
                clip[cnt, :, :, 0] = frames_list[i]
                cnt = cnt + 1
                if cnt == sequence_size:
                    clips.append(clip.copy())
                    cnt = 0
        return clips





    def get_clips(frames_list, sequence_size):
        """ 
        Parameters
        ----------
        frames_list : list
            A list of sorted frames of shape 256 X 256
        sequence_size: int
            The size of the lstm sequence
        Returns
        -------
        list
            A list of clips , sequence_size frames each
        """
        clips = []
        sz = len(frames_list)
        clip = np.zeros(shape=(sequence_size, 256, 256, 1))
        cnt = 0

        for i in range(0, sz):
            clip[cnt, :, :, 0] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(clip.copy())
                cnt = 0
        return clips
    
    
    
    
    
    def get_training_set(self, input_frames):
        return self.get_clips(frames_list=input_frames, sequence_size=self.sequence_size)
    
    
    
    
    
    
    def init_model_dir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
        
        
    def new_model_path(self, model_dir,name):
        return os.path.join(model_dir, name + '_' +  datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    def get_last_model_path(self, model_dir):
        model_list = os.listdir(model_dir) 
        if len(model_list) == 0:
            print('No model yet')
        else:
            model_list.sort()
            last_model = model_list[-1]
            return os.path.join(model_dir,last_model)

    def get_model(self, model_path):
        return load_model(model_path, custom_objects={'LayerNormalization': LayerNormalization})




    def fit_new_model(self, training_set, batch_size_, epochs_, model_dir):
        training_set = np.array(training_set)
        seq = Sequential()
        seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, 10, 256, 256, 1)))
        seq.add(LayerNormalization())
        seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same")))
        seq.add(LayerNormalization())
        # # # # #
        seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
        seq.add(LayerNormalization())
        seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
        seq.add(LayerNormalization())
        seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
        seq.add(LayerNormalization())
        # # # # #
        seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
        seq.add(LayerNormalization())
        seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
        seq.add(LayerNormalization())
        seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))
        print(seq.summary())
        seq.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))
        seq.fit(training_set, training_set,
                batch_size=batch_size_, epochs=epochs_, shuffle=False)
        seq.save(self.new_model_path(model_dir,'lstm'))
        
        
        
    def pick_threshold(threshold_list):
        pass
    
    
    def predict(self, frame_sequence):
        frame_sequence = np.reshape(frame_sequence, (1,) + frame_sequence.shape)

        reconstructed_sequence = self.model.predict(feat,batch_size=self.batch_size)
        reconstruction_cost = np.linalg.norm(np.subtract(frame_sequence,reconstructed_sequence))
        return reconstruction_cost
    
    
    
    def fit(self, training_set):
        self.fit_new_model(training_set, self.batch_size, self.epochs, self.model_dir)
        
        
    
        
        
        
    def get_last_model(self, model_dir):
        ##Todo if model is not present
        last_model_path = self.get_last_model_path(model_dir)
        model = self.get_model(last_model_path)
        return model
    
    def use_last_model(self):
        model = self.get_last_model(self.model_dir)
        self.model = model
        
        
        
        
    def update_max_min(self, value):
        if self.max_cost == None:
            self.max_cost = value
            self.min_cost = value
        else:
            if value < self.min_cost:
                self.min_cost = value
            
            if value > self.max_cost:
                self.max_cost = value
                
    def get_min_cost(self):
        return self.min_cost
    
    def get_max_cost(self):
        return self.max_cost
    
    def get_current_reg_score(self, value):
        sa = (value - self.min_cost) / self.max_cost
        regularity_score = 1.0 - sa
        return regularity_score
    
    def reset_cost(self):
        self.max_cost = None
        self.min_cost = None