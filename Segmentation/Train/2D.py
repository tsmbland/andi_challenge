import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')

from andi_funcs import track_generator_segmentation, load_tracks
from models import segmentation_model_2d
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

"""
To do: custom metric which measures absolute position error

"""

# Load validation data
tracks_val = load_tracks('../Validation/tracks_2D.txt', dims=2)
exponents_val = np.loadtxt('../Validation/positions_2D.txt')

# Run model
model = segmentation_model_2d()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(track_generator_segmentation(dimensions=2, n=32, track_length=200), steps_per_epoch=200, epochs=100,
                    callbacks=[
                        ModelCheckpoint(filepath='../Models/2D.h5', monitor='val_accuracy',
                                        save_best_only=True,
                                        mode='min')],
                    validation_data=(tracks_val, exponents_val))

# Save performance metrics
np.savetxt('../TrainingMetrics/2D_accuracy.txt', history.history['accuracy'])
np.savetxt('../TrainingMetrics/2D_val_accuracy.txt', history.history['val_accuracy'])
