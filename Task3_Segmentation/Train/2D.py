import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')

from andi_funcs import TrackGeneratorSegmentation, import_tracks, import_labels, package_tracks
from models import segmentation_model_2d
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np

# Load validation data
tracks_val = package_tracks(import_tracks('../../Datasets/Validation/task3.txt')[1], dimensions=2, max_T=200)
positions_val = import_labels('../../Datasets/Validation/ref3.txt')[1] - 1
tracks_test = package_tracks(import_tracks('../../Datasets/Test/task3.txt')[1], dimensions=2, max_T=200)

# Run model
model = segmentation_model_2d()
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(TrackGeneratorSegmentation(batches=200, batch_size=32, dimensions=2), epochs=200,
                    callbacks=[
                        ModelCheckpoint(filepath='../Models/2D.h5', monitor='val_accuracy', save_best_only=True,
                                        mode='max')],
                    validation_data=(tracks_val, positions_val), use_multiprocessing=True, workers=16)

# Save performance metrics
np.savetxt('2D_accuracy.txt', history.history['accuracy'])
np.savetxt('2D_val_accuracy.txt', history.history['val_accuracy'])

# Evaluate on test data
model = load_model('../Models/2D.h5')
np.savetxt('../../Datasets/Test/predictions_task3_2D.txt', model.predict(tracks_test, use_multiprocessing=True))
