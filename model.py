import sys
import numpy as np
import h5py
from tensorflow.contrib.keras import models, layers, callbacks, utils

# path error on windows, workaround
Conv2D = layers.Conv2D
Flatten = layers.Flatten
Dense = layers.Dense
BatchNormalization = layers.BatchNormalization
Activation = layers.Activation
Dropout = layers.Dropout
Sequential = models.Sequential
ModelCheckpoint = callbacks.ModelCheckpoint
HDF5Matrix = utils.HDF5Matrix

dataset_xs = HDF5Matrix('dataset.hdf5', 'xs')
dataset_ys = HDF5Matrix('dataset.hdf5', 'ys')

model = Sequential([
    Conv2D(24, (5, 5), strides=(2, 2), padding='same', input_shape=(160, 40, 3)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(36, (5, 5), strides=(2, 2), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(48, (5-2, 5-2), strides=(2-1, 2-1), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Flatten(),
    Dense(512),
    BatchNormalization(),
    Activation('relu'),
    Dense(100),
    BatchNormalization(),
    Activation('relu'),
    Dense(1)
])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
print(model.summary())

if __name__ == '__main__':
    checkpointer = ModelCheckpoint(filepath='model.hdf5', verbose=1)
    model.fit(dataset_xs, dataset_ys, epochs=80, batch_size=128, shuffle='batch', callbacks=[checkpointer])
