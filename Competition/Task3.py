import numpy as np
from tensorflow.keras.models import load_model
from andi_funcs import preprocess_tracks
import csv
import os


def import_x(direc):
    t = csv.reader(open(direc, 'r'), delimiter=';', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
    X = [[], []]
    for trajs in t:
        if int(trajs[0]) in [1, 2]:
            X[int(trajs[0]) - 1].append(trajs[1:])
    return X[0], X[1]


def import_y(direc):
    l = csv.reader(open(direc, 'r'), delimiter=';', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
    Y = [[], []]
    for labels in l:
        if int(labels[0]) in [1, 2]:
            Y[int(labels[0]) - 1].append(labels[1])
    return np.array(Y[0]), np.array(Y[1])


def package_tracks(tracks, max_T, dimensions):
    # Package into array
    n = len(tracks)
    tracks_array = np.zeros([n, max_T, dimensions])
    if dimensions == 1:
        for i, t in enumerate(tracks):
            tracks_array[i, :, 0] = t
    elif dimensions == 2:
        for i, t in enumerate(tracks):
            len_t = int(len(t) / 2)
            tracks_array[i, :, 0] = t[:len_t]
            tracks_array[i, :, 1] = np.array(t[len_t:]) - t[len_t]

    # Preprocess
    tracks_array = preprocess_tracks(tracks_array)
    return tracks_array


def split_tracks(tracks, positions, dimensions=1, max_T=200):
    g = 0
    tracks_array = np.zeros([len(tracks) * 2, max_T, dimensions])
    i = 0
    if dimensions == 1:
        for j, track in enumerate(tracks):
            tracks_array[i, max_T - max(positions[j] - g, 0):, 0] = track[:max(positions[j] - g, 0)]
            tracks_array[i + 1, min(positions[j] + g, 200):, 0] = track[min(positions[j] + g, 200):]
            i += 2
    elif dimensions == 2:
        for j, track in enumerate(tracks):
            len_t = int(len(track) / 2)
            d1 = track[:len_t].flatten()
            d2 = track[len_t:].flatten() - track[len_t]
            tracks_array[i, max_T - max(positions[j] - g, 0):, 0] = d1[:max(positions[j] - g, 0)]
            tracks_array[i, max_T - max(positions[j] - g, 0):, 1] = d2[:max(positions[j] - g, 0)]
            tracks_array[i + 1, min(positions[j] + g, 200):, 0] = d1[min(positions[j] + g, 200):]
            tracks_array[i + 1, min(positions[j] + g, 200):, 1] = d2[min(positions[j] + g, 200):]
            i += 2

    # Preprocess
    tracks_array = preprocess_tracks(tracks_array)
    return tracks_array


"""
Import data

"""

# Import x data
tracks_1D, tracks_2D = import_x('challenge_for_scoring/task3.txt')
tracks_1D_packaged = package_tracks(tracks_1D, dimensions=1, max_T=200)
tracks_2D_packaged = package_tracks(tracks_2D, dimensions=2, max_T=200)

"""
1D

"""

thresh = 0.1

# Position
model = load_model('../Segmentation/Models/1D.h5')
res = model.predict(package_tracks(tracks_1D, dimensions=1, max_T=200))
maxes = np.max(res, axis=1)
positions_1D = np.argmax(res, axis=1) + 1
positions_1D[maxes < thresh] = 100

# Class
model = load_model('../Classification/Models/1D.h5')
d1_res1 = np.argmax(model.predict(tracks_1D_packaged[maxes < thresh]), axis=1)
d1_res2 = np.argmax(model.predict(split_tracks(np.array(tracks_1D)[maxes > thresh], positions_1D[maxes > thresh])),
                    axis=1)
d1_res_1a = np.zeros(len(tracks_1D))
d1_res_1b = np.zeros(len(tracks_1D))
d1_res_1a[maxes < thresh] = d1_res1
d1_res_1a[maxes > thresh] = d1_res2[::2]
d1_res_1b[maxes < thresh] = d1_res1
d1_res_1b[maxes > thresh] = d1_res2[1::2]

# Exponent
model = load_model('../Regression/Models/1D.h5')
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
model = load_model('../Segmentation/Models/2D.h5')
res = model.predict(package_tracks(tracks_2D, dimensions=2, max_T=200))
maxes = np.max(res, axis=1)
positions_2D = np.argmax(res, axis=1) + 1
positions_2D[maxes < thresh] = 100

# Class
model = load_model('../Classification/Models/2D.h5')
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

# Exponent
model = load_model('../Regression/Models/2D.h5')
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
