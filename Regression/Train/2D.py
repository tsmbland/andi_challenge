import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')

from andi_funcs import track_generator_regression, load_tracks
from models import regression_model_2d
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# Load validation data
tracks_val = load_tracks('../Validation/tracks_2D.txt', dims=2)
exponents_val = np.loadtxt('../Validation/exponents_2D.txt')

# Run model
model = regression_model_2d()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
model.summary()
history = model.fit(track_generator_regression(dimensions=2, n=32, min_T=7, max_T=200), steps_per_epoch=200, epochs=100,
                    callbacks=[
                        ModelCheckpoint(filepath='../Models/2D.h5', monitor='val_mae', save_best_only=True,
                                        mode='min')],
                    validation_data=(tracks_val, exponents_val))

# Save performance metrics
np.savetxt('../TrainingMetrics/2D_mae.txt', history.history['mae'])
np.savetxt('../TrainingMetrics/2D_val_mae.txt', history.history['val_mae'])
