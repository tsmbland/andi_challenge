import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')

from andi_funcs import track_generator_classification, import_tracks, import_labels, package_tracks
from models import classification_model_1d
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np

# Load validation data
tracks_val = package_tracks(import_tracks('../../Datasets/Validation/task2.txt')[0], dimensions=1, max_T=1001)
classes_val = import_labels('../../Datasets/Validation/ref2.txt')[0]
tracks_test = package_tracks(import_tracks('../../Datasets/Test/task2.txt')[0], dimensions=1, max_T=1001)

# Run model
model = classification_model_1d()
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(track_generator_classification(dimensions=1, n=32, min_T=5, max_T=1001), steps_per_epoch=200,
                    epochs=200,
                    callbacks=[
                        ModelCheckpoint(filepath='../Models/1D.h5', monitor='val_accuracy',
                                        save_best_only=True, mode='max')],
                    validation_data=(tracks_val, classes_val), use_multiprocessing=False)

# Save performance metrics
np.savetxt('1D_accuracy.txt', history.history['accuracy'])
np.savetxt('1D_val_accuracy.txt', history.history['val_accuracy'])

# Evaluate on test data
model = load_model('../Models/1D.h5')
np.savetxt('../../Datasets/Test/predictions_task2_1D.txt', model.predict(tracks_test, use_multiprocessing=True))
