import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')

from andi_funcs import track_generator_classification, load_tracks
from models import classification_model_2d
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# Load validation data
tracks_val = load_tracks('../Validation/tracks_2D.txt', dims=2)
classes_val = np.loadtxt('../Validation/classes_2D.txt')

# Run model
model = classification_model_2d()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(track_generator_classification(dimensions=2, n=32, min_T=7, max_T=200), steps_per_epoch=200,
                    epochs=100,
                    callbacks=[
                        ModelCheckpoint(filepath='../Models/2D.h5', monitor='val_accuracy', save_best_only=True,
                                        mode='max')],
                    validation_data=(tracks_val, classes_val))

# Save performance metrics
np.savetxt('../TrainingMetrics/2D_accuracy.txt', history.history['accuracy'])
np.savetxt('../TrainingMetrics/2D_val_accuracy.txt', history.history['val_accuracy'])