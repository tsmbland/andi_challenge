import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')

from andi_funcs import track_generator_regression, load_tracks
from models import regression_model_1d
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# Load validation data
tracks_val = load_tracks('../Validation/tracks_1D.txt', dims=1)
exponents_val = np.loadtxt('../Validation/exponents_1D.txt')

# Run model
model = regression_model_1d()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
model.summary()
history = model.fit(track_generator_regression(dimensions=1, n=32, min_T=7, max_T=200), steps_per_epoch=200, epochs=100,
                    callbacks=[
                        ModelCheckpoint(filepath='../Models/1D.h5', monitor='val_mae', save_best_only=True,
                                        mode='min')],
                    validation_data=(tracks_val, exponents_val))

# Save performance metrics
np.savetxt('../TrainingMetrics/1D_mae.txt', history.history['mae'])
np.savetxt('../TrainingMetrics/1D_val_mae.txt', history.history['val_mae'])
