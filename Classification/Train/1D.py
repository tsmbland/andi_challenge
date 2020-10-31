import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')

from andi_funcs import track_generator_classification, load_tracks
from models import classification_model_1d
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np

# Load validation data
tracks_val = load_tracks('../Validation/tracks_1D.txt', dims=1)
classes_val = np.loadtxt('../Validation/classes_1D.txt')
tracks_test = load_tracks('../Test/tracks_1D.txt', dims=1)

# Run model
model = classification_model_1d()
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(track_generator_classification(dimensions=1, n=32, min_T=5, max_T=1001), steps_per_epoch=200,
                    epochs=200,
                    callbacks=[
                        ModelCheckpoint(filepath='../Models/1D.h5', monitor='val_accuracy',
                                        save_best_only=True, mode='max')],
                    validation_data=(tracks_val, classes_val), use_multiprocessing=True)

# Save performance metrics
np.savetxt('../TrainingMetrics/1D_accuracy.txt', history.history['accuracy'])
np.savetxt('../TrainingMetrics/1D_val_accuracy.txt', history.history['val_accuracy'])

# Evaluate on test data
model = load_model('../Models/1D.h5')
np.savetxt('../TestPredictions/1D.txt', model.predict(tracks_test))
