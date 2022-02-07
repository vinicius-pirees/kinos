import keras
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from keras.models import Sequential, load_model
from keras_layer_normalization import LayerNormalization
import logging
import json
import os
import sys
import cv2
from matplotlib import pyplot as plt
import time
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import tensorflow as tf
from phaino.utils.commons import frame_to_gray, reduce_frame, resolve_model_path, get_last_model_path


logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


from datetime import datetime



class LSTMAutoEncoder:
    
    def __init__(self, batch_size=1, epochs=2, sequence_size=10, model_name=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_size = sequence_size
        self.model = None
        self.max_cost = None
        self.min_cost = None
        if model_name is None:
            self.model_name = 'lstm_autoencoder'
        else:
            self.model_name = model_name

        self.training_data_name=None
        
        
    def lstm_autoencoder_frame(self, frame, input_size):
        frame = cv2.resize(frame, input_size)
        frame = frame_to_gray(frame)
        reduced_frame = reduce_frame(frame)
        return reduced_frame


    def input_frames_transform(self, train_frames):
        input_frames = []
        for frame in range(0,train_frames.shape[0]):
            input_frames.append(self.lstm_autoencoder_frame(train_frames[frame], (256,256)))
            
        return input_frames
    
    
    def get_clips_by_stride(self, stride, frames_list, sequence_size):
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



    def get_clips(self, frames_list, sequence_size):
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
    

    def get_model(self, model_path):
        return load_model(model_path, custom_objects={'LayerNormalization': LayerNormalization})


    def fit(self, training_set, training_data_name=None):

        temp_training_set = []
        if training_data_name is not None:
            self.training_data_name = training_data_name

        if any(isinstance(el, list) for el in training_set): # if training set is a sequence of frames
            for sequence in training_set:
                transformed_frames = self.input_frames_transform(np.array(sequence))
                temp_training_set += self.get_training_set(transformed_frames)
        else:
            transformed_frames = self.input_frames_transform(np.array(training_set))
            temp_training_set = self.get_training_set(transformed_frames)

        final_training_set = np.array(temp_training_set)


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
        seq.fit(final_training_set, final_training_set,
                batch_size=self.batch_size, epochs=self.epochs, shuffle=False)

        self.model = seq
        print('veio')
        
        
    def save_model(self):
        logger.info(f"Saving the model {self.model_name}")
        base_path = resolve_model_path(self.model_name)
        path = os.path.join(base_path, 'model')
        self.model.save(path)


        #save metadata
        metadata = {
            "batch_size": self.batch_size, 
            "epochs": self.epochs,
            "sequence_size": self.sequence_size,
            "training_data_name": self.training_data_name
           
        }

        with open(os.path.join(base_path, "metadata.json"), "w") as outfile:
            json.dump(metadata, outfile)

        
    def pick_threshold(threshold_list):
        pass
    
    
    def predict(self, frame_sequence):
        transformed_frames = self.input_frames_transform(np.array(frame_sequence))
        transformed_frames = self.get_training_set(transformed_frames)
        frame_sequence = np.reshape(transformed_frames, (1,) + transformed_frames.shape)

        reconstructed_sequence = self.model.predict(frame_sequence,batch_size=self.batch_size)
        reconstruction_cost = np.linalg.norm(np.subtract(frame_sequence,reconstructed_sequence))
        return reconstruction_cost
    
    
    
    def load_model(self):
        base_path = resolve_model_path(self.model_name)
        path = os.path.join(base_path, 'model')
        self.model = load_model(path, custom_objects={'LayerNormalization': LayerNormalization})

    def load_last_model(self):
        base_path = resolve_model_path(self.model_name)
        path = os.path.join(base_path, 'model')
        self.load_model(path)    
        

        
        
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
