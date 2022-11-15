import os
import keras
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from keras.models import Sequential, load_model
from keras_layer_normalization import LayerNormalization
from keras import layers
import numpy as np



class AutoEncoder():
    def __init__(self):
        self.algorithm = None


    def fit(self, training_data):
        final_training_set = np.array(training_data)
        final_training_set = final_training_set.astype('float32') / 255
        final_training_set = np.reshape(final_training_set, (len(final_training_set), 256, 256, 1))

        input_img = keras.Input(shape=(256, 256, 1))

        x = layers.Conv2D(32, (11,1), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(1, (11, 11), activation='sigmoid', padding='same')(x)

        self.algorithm = keras.models.Model(input_img, decoded)
        self.algorithm.compile(optimizer='adam', loss='binary_crossentropy')


        self.algorithm.fit(final_training_set, final_training_set,
                epochs=4,
                batch_size=20,
                shuffle=True,
                )

    def predict(self, examples):

        test_frames_data = examples.astype('float32') / 255
        test_frames_data = np.reshape(test_frames_data, (len(test_frames_data), 256, 256, 1))

        transformed_frames = self.algorithm.predict(test_frames_data)

        transformed = transformed_frames.reshape(transformed_frames.shape[0], transformed_frames.shape[1]* transformed_frames.shape[2] * transformed_frames.shape[3])
        original  = test_frames_data.reshape(test_frames_data.shape[0], test_frames_data.shape[1]* test_frames_data.shape[2] * test_frames_data.shape[3])
        reconstruction_errors = np.linalg.norm(np.subtract(original,transformed), axis=1)

        return reconstruction_errors
