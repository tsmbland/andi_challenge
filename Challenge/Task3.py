from tensorflow.keras.models import load_model
from andi_funcs import import_tracks, package_tracks, split_tracks
import os
import numpy as np

# Specify path to track data folder containing task3.txt
path = ''

"""
Import data

"""

# Import x data
tracks_1D, tracks_2D = import_tracks(path + '/task3.txt')
tracks_1D_packaged = package_tracks(tracks_1D, dimensions=1, max_T=200)
tracks_2D_packaged = package_tracks(tracks_2D, dimensions=2, max_T=200)

"""
1D

"""

thresh = 0.1  # arbitrary threshold for changepoint detection - depends on the desired precision/recall balance

# Position
model = load_model('../Task3_Segmentation/Models/1D.h5')
res = model.predict(package_tracks(tracks_1D, dimensions=1, max_T=200))
maxes = np.max(res, axis=1)
positions_1D = np.argmax(res, axis=1) + 1
positions_1D[maxes < thresh] = 100  # make a 'best guess' when a changepoint can't be found above the threshold

# Class
model = load_model('../Task2_Classification/Models/1D.h5')
d1_res1 = np.argmax(model.predict(tracks_1D_packaged[maxes < thresh]),
                    axis=1)  # analyse whole track together if switch not found
d1_res2 = np.argmax(model.predict(split_tracks(np.array(tracks_1D)[maxes > thresh], positions_1D[maxes > thresh])),
                    axis=1)
d1_res_1a = np.zeros(len(tracks_1D))
d1_res_1b = np.zeros(len(tracks_1D))
d1_res_1a[maxes < thresh] = d1_res1
d1_res_1a[maxes > thresh] = d1_res2[::2]
d1_res_1b[maxes < thresh] = d1_res1
d1_res_1b[maxes > thresh] = d1_res2[1::2]

# Task1_Exponent
model = load_model('../Task1_Exponent/Models/1D.h5')
d1_res1 = model.predict(tracks_1D_packaged[maxes < thresh]).flatten()
d1_res2 = model.predict(split_tracks(np.array(tracks_1D)[maxes > thresh], positions_1D[maxes > thresh])).flatten()
d1_res_2a = np.zeros(len(tracks_1D))
d1_res_2b = np.zeros(len(tracks_1D))
d1_res_2a[maxes < thresh] = d1_res1
d1_res_2a[maxes > thresh] = d1_res2[::2]
d1_res_2b[maxes < thresh] = d1_res1
d1_res_2b[maxes > thresh] = d1_res2[1::2]

"""
2D

"""

thresh = 0.15

# Position
model = load_model('../Task3_Segmentation/Models/2D.h5')
res = model.predict(package_tracks(tracks_2D, dimensions=2, max_T=200))
maxes = np.max(res, axis=1)
positions_2D = np.argmax(res, axis=1) + 1
positions_2D[maxes < thresh] = 100

# Class
model = load_model('../Task2_Classification/Models/2D.h5')
d2_res1 = np.argmax(model.predict(tracks_2D_packaged[maxes < thresh]), axis=1)
d2_res2 = np.argmax(
    model.predict(split_tracks(np.array(tracks_2D)[maxes > thresh], positions_2D[maxes > thresh], dimensions=2)),
    axis=1)
d2_res_1a = np.zeros(len(tracks_2D))
d2_res_1b = np.zeros(len(tracks_2D))
d2_res_1a[maxes < thresh] = d2_res1
d2_res_1a[maxes > thresh] = d2_res2[::2]
d2_res_1b[maxes < thresh] = d2_res1
d2_res_1b[maxes > thresh] = d2_res2[1::2]

# Task1_Exponent
model = load_model('../Task1_Exponent/Models/2D.h5')
d2_res1 = model.predict(tracks_2D_packaged[maxes < thresh]).flatten()
d2_res2 = model.predict(
    split_tracks(np.array(tracks_2D)[maxes > thresh], positions_2D[maxes > thresh], dimensions=2)).flatten()
d2_res_2a = np.zeros(len(tracks_2D))
d2_res_2b = np.zeros(len(tracks_2D))
d2_res_2a[maxes < thresh] = d2_res1
d2_res_2a[maxes > thresh] = d2_res2[::2]
d2_res_2b[maxes < thresh] = d2_res1
d2_res_2b[maxes > thresh] = d2_res2[1::2]

"""
Save

"""

# Get exponent = nan for 0-length trajectories when the changepoint is at either end - setting these to 1
d1_res_2a[d1_res_2a != d1_res_2a] = 1
d1_res_2b[d1_res_2b != d1_res_2b] = 1
d2_res_2a[d2_res_2a != d2_res_2a] = 1
d2_res_2b[d2_res_2b != d2_res_2b] = 1

file = 'task3.txt'
if os.path.exists(file):
    os.remove(file)
with open(file, 'a') as f:
    for i, p in enumerate(positions_1D):
        f.write('1;%s;%s;%s;%s;%s\n' % (p, d1_res_1a[i], d1_res_2a[i], d1_res_1b[i], d1_res_2b[i]))
    for i, p in enumerate(positions_2D):
        f.write('2;%s;%s;%s;%s;%s\n' % (p, d2_res_1a[i], d2_res_2a[i], d2_res_1b[i], d2_res_2b[i]))
