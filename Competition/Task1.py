import numpy as np
from tensorflow.keras.models import load_model
from andi_funcs import preprocess_tracks
import csv
import os

path = ''

"""
Functions

"""


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
            tracks_array[i, max_T - len(t):, 0] = t
    elif dimensions == 2:
        for i, t in enumerate(tracks):
            len_t = int(len(t) / 2)
            tracks_array[i, max_T - len_t:, 0] = t[:len_t]
            tracks_array[i, max_T - len_t:, 1] = t[len_t:]

    # Preprocess
    tracks_array = preprocess_tracks(tracks_array)
    return tracks_array


"""
Import data

"""

# Import x data
tracks_1D, tracks_2D = import_x(path + '/task1.txt')

"""
1D

"""

model = load_model('../Exponent/Models/1D.h5')
res_1D = model.predict(package_tracks(tracks_1D, dimensions=1, max_T=2001)).flatten()

"""
2D

"""

model = load_model('../Exponent/Models/2D.h5')
res_2D = model.predict(package_tracks(tracks_2D, dimensions=2, max_T=2001)).flatten()

"""
Save

"""

file = 'task1.txt'
if os.path.exists(file):
    os.remove(file)
with open(file, 'a') as f:
    for i in res_1D:
        f.write('1;%s\n' % i)
    for i in res_2D:
        f.write('2;%s\n' % i)
