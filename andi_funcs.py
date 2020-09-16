import numpy as np
import andi
import matplotlib.pyplot as plt
import sys
import os

"""
To do:
- move all preprocessing into neural network (lambda layers)
- support for 3D tracks

"""

"""
Regression

"""


def generate_tracks_regression(n, dimensions, min_T=7, max_T=110):
    # Create tracks
    AD = andi.andi_datasets()
    X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=n, min_T=min_T, max_T=max_T, tasks=[1], dimensions=[dimensions])
    exponents = np.array(Y1[dimensions - 1])
    tracks = X1[dimensions - 1]

    # Package into array
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

    return tracks_array, exponents


def track_generator_regression(n, dimensions, min_T=7, max_T=110):
    while True:
        tracks, exponents = generate_tracks_regression(n, dimensions=dimensions, min_T=min_T, max_T=max_T)
        yield tracks, exponents


"""
Classification

"""


def generate_tracks_classification(n, dimensions, min_T=7, max_T=110):
    # Create tracks
    AD = andi.andi_datasets()
    X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=n, min_T=min_T, max_T=max_T, tasks=[2], dimensions=[dimensions])
    classes = np.array(Y2[dimensions - 1]).astype(int)
    tracks = X2[dimensions - 1]

    # Package into array
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

    return tracks_array, classes


def track_generator_classification(n, dimensions, min_T=7, max_T=110):
    while True:
        tracks, classes = generate_tracks_classification(n, dimensions=dimensions, min_T=min_T, max_T=max_T)
        yield tracks, classes


"""
Segmentation

"""


def generate_tracks_segmentation(n, dimensions, track_length=200):
    # Create tracks
    AD = andi.andi_datasets()
    X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=n, tasks=[3], dimensions=[dimensions], min_T=track_length,
                                             max_T=track_length + 1)
    positions = np.array(Y3[dimensions - 1])[:, 1].astype(int) - 1
    tracks = X3[dimensions - 1]

    # Package into array
    tracks_array = np.zeros([n, track_length, dimensions])
    if dimensions == 1:
        for i, t in enumerate(tracks):
            tracks_array[i, :, 0] = t
    elif dimensions == 2:
        for i, t in enumerate(tracks):
            len_t = int(len(t) / 2)
            tracks_array[i, :, 0] = t[:len_t]
            tracks_array[i, :, 1] = t[len_t:]

    # Preprocess
    tracks_array = preprocess_tracks(tracks_array)

    return tracks_array, positions


def track_generator_segmentation(n, dimensions, track_length=200):
    while True:
        tracks, positions = generate_tracks_segmentation(n, dimensions=dimensions, track_length=track_length)
        yield tracks, positions


"""
Misc

"""


def noisy_tracks(tracks, noise_range=(0, 1)):
    if tracks.shape[2] == 1:
        noises = np.random.uniform(noise_range[0], noise_range[1], tracks.shape[0])
        std_step = np.std(np.diff(tracks, axis=1), axis=1)
        for t in range(tracks.shape[0]):
            tracks_noisy = tracks[t, :, :] + np.random.normal(0, std_step[t] * noises[t], tracks.shape[1:])
            tracks[t, :, :] = tracks_noisy - tracks_noisy[0]
        return tracks
    elif tracks.shape[2] == 2:
        noises = np.random.uniform(noise_range[0], noise_range[1], tracks.shape[0])
        xs = tracks[:, :, 0]
        ys = tracks[:, :, 1]
        std_step = np.std(((np.diff(xs, axis=1) ** 2) + (np.diff(ys, axis=1) ** 2)) ** 0.5, axis=1)
        for t in range(xs.shape[0]):
            xs_noisy = xs[t, :] + np.random.normal(0, std_step[t] * noises[t], xs.shape[1])
            ys_noisy = ys[t, :] + np.random.normal(0, std_step[t] * noises[t], xs.shape[1])
            xs[t, :] = xs_noisy - xs_noisy[0]
            ys[t, :] = ys_noisy - ys_noisy[0]
        return np.dstack((xs, ys))


# def normalise_tracks(tracks):
#     if tracks.shape[2] == 1:
#         diff = abs(np.diff(tracks[:, :, 0], axis=1, prepend=0))
#         mean_step = np.mean(diff, axis=1)
#         return tracks[:, :, :] / (mean_step[:, None, None] * tracks.shape[1])
#     elif tracks.shape[2] == 2:
#         dx = np.diff(tracks[:, :, 0], prepend=0)
#         dy = np.diff(tracks[:, :, 1], prepend=0)
#         mean_step = np.mean(((dx ** 2) + (dy ** 2)) ** 0.5, axis=1)
#         return tracks[:, :, :] / (mean_step[:, None, None] * tracks.shape[1])


def preprocess_tracks(tracks):
    """
    Return normalised differences

    """

    if tracks.shape[2] == 1:
        diff = np.diff(tracks[:, :, 0], axis=1)
        meanstep = np.sum(abs(diff), axis=1) / np.sum(tracks[:, :, 0] != 0, axis=1)
        # return np.expand_dims(diff, axis=-1) / 10
        return np.expand_dims(diff / np.expand_dims(meanstep, axis=-1), axis=-1)

    elif tracks.shape[2] == 2:
        dx = np.diff(tracks[:, :, 0], axis=1)
        dy = np.diff(tracks[:, :, 1], axis=1)
        meanstep = np.expand_dims(np.sum(((dx ** 2) + (dy ** 2)) ** 0.5, axis=1) / np.sum(tracks[:, :, 0] != 0, axis=1),
                                  axis=-1)
        return np.dstack((dx / meanstep, dy / meanstep))


def save_tracks(tracks, dims, path):
    """
    Reshapes multi-tracks into 2D format to allow saving

    """
    if dims == 1:
        tracks = tracks[:, :, 0]
    if dims == 2:
        tracks = np.c_[tracks[:, :, 0], tracks[:, :, 1]]
    if dims == 3:
        tracks = np.c_[tracks[:, :, 0], tracks[:, :, 1], tracks[:, :, 2]]
    np.savetxt(path, tracks)


def load_tracks(path, dims):
    """
    Load and reshape

    """

    tracks = np.loadtxt(path)

    if dims == 1:
        tracks = np.expand_dims(tracks, axis=-1)
    if dims == 2:
        tracks = np.dstack((tracks[:, :tracks.shape[1] // 2], tracks[:, tracks.shape[1] // 2:]))

    return tracks
