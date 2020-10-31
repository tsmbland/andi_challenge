import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')

from andi_funcs import track_generator_regression, load_tracks
from models import regression_model_2d
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np

# Load validation data
tracks_val = load_tracks('../Validation/tracks_2D.txt', dims=2)
exponents_val = np.loadtxt('../Validation/exponents_2D.txt')
tracks_test = load_tracks('../Test/tracks_2D.txt', dims=2)

# Run model
model = regression_model_2d()
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()
history = model.fit(track_generator_regression(dimensions=2, n=32, min_T=5, max_T=1001), steps_per_epoch=200,
                    epochs=200,
                    callbacks=[
                        ModelCheckpoint(filepath='../Models/2D.h5', monitor='val_mae', save_best_only=True,
                                        mode='min')],
                    validation_data=(tracks_val, exponents_val), use_multiprocessing=True)

# Save performance metrics
np.savetxt('../TrainingMetrics/2D_mae.txt', history.history['mae'])
np.savetxt('../TrainingMetrics/2D_val_mae.txt', history.history['val_mae'])

# Evaluate on test data
model = load_model('../Models/2D.h5')
np.savetxt('../TestPredictions/2D.txt', model.predict(tracks_test))
