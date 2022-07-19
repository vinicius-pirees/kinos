import os
import json
import numpy as np
import cv2


import keras
from keras import layers
from kinos.utils.commons import get_clips
from keras.models import Sequential, load_model
import logging
from kinos.utils.commons import frame_to_gray, reduce_frame, resolve_model_path, get_last_model_path



logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)



class CNNAutoEncoder:
    def __init__(self, batch_size=5, epochs=15, sequence_size=12, model_name=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_size = sequence_size
        self.model = None
        if model_name is None:
            self.model_name = 'cnn_autoencoder'
        else:
            self.model_name = model_name

        self.training_data_name=None



    def cnn_autoencoder_frame(self, frame, input_size):
        frame = cv2.resize(frame, input_size)
        frame = frame_to_gray(frame)
        reduced_frame = reduce_frame(frame)
        return reduced_frame


    def input_frames_transform(self, train_frames):
        input_frames = []
        for frame in range(0,train_frames.shape[0]):
            input_frames.append(self.cnn_autoencoder_frame(train_frames[frame], (256,256)))
            
        return input_frames

    def get_training_set(self, input_frames):
        return get_clips(frames_list=input_frames, sequence_size=self.sequence_size)

    def fit(self, training_set, training_data_name=None):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)
        
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

        seq.add(layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same',batch_input_shape=(None, 12, 256, 256, 1)))
        seq.add(layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
        seq.add(layers.MaxPooling3D((2, 2, 2), padding='same'))
        seq.add(layers.Conv3D(8, (3, 3, 3), activation='relu', padding='same'))
        seq.add(layers.MaxPooling3D((2, 2, 2), padding='same'))

        seq.add(layers.Conv3D(8, (3, 3, 3), activation='relu', padding='same'))
        seq.add(layers.UpSampling3D((2, 2, 2)))
        seq.add(layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
        seq.add(layers.UpSampling3D((2, 2, 2)))
        seq.add(layers.Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same'))

        print(seq.summary())

        seq.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))
        seq.fit(final_training_set, final_training_set, batch_size=self.batch_size, epochs=self.epochs, shuffle=False)

        self.model = seq
        self.save_model()

    def predict(self, frame_sequence):
        transformed_frames = self.input_frames_transform(np.array(frame_sequence))
        transformed_frames = self.get_training_set(transformed_frames)
        transformed_frames = np.array(transformed_frames)
        
        
        frame_sequence = transformed_frames

        reconstructed_sequence = self.model.predict(frame_sequence,batch_size=self.batch_size)
        reconstruction_cost = np.linalg.norm(np.subtract(frame_sequence,reconstructed_sequence))
        return reconstruction_cost


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

    def load_model(self, path):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)
        
        self.model = load_model(path)

        with open(os.path.join(os.path.dirname(path), "metadata.json")) as infile:
            metadata = json.load(infile)
        
        self.training_data_name = metadata['training_data_name']

    def load_last_model(self):
        base_path = get_last_model_path(self.model_name)
        path = os.path.join(base_path, 'model')
        self.load_model(path)    