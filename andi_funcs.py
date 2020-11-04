import numpy as np
import andi

"""
Exponent

"""


def generate_tracks_regression(n, dimensions, min_T=5, max_T=1001):
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


def track_generator_regression(n, dimensions, min_T=5, max_T=1001):
    while True:
        tracks, exponents = generate_tracks_regression(n, dimensions=dimensions, min_T=min_T, max_T=max_T)
        yield tracks, exponents


"""
Classification

"""


def generate_tracks_classification(n, dimensions, min_T=5, max_T=1001):
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


def track_generator_classification(n, dimensions, min_T=5, max_T=1001):
    while True:
        tracks, classes = generate_tracks_classification(n, dimensions=dimensions, min_T=min_T, max_T=max_T)
        yield tracks, classes


"""
Segmentation

"""


def generate_tracks_segmentation(n, dimensions):
    # Create tracks
    AD = andi.andi_datasets()
    X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=n, tasks=[3], dimensions=[dimensions])
    positions = np.array(Y3[dimensions - 1])[:, 1].astype(int) - 1
    tracks = X3[dimensions - 1]

    # Package into array
    tracks_array = np.zeros([n, 200, dimensions])
    if dimensions == 1:
        for i, t in enumerate(tracks):
            tracks_array[i, :, 0] = t
    elif dimensions == 2:
        for i, t in enumerate(tracks):
            len_t = int(len(t) / 2)
            tracks_array[i, :, 0] = t[:len_t]
            tracks_array[i, :, 1] = t[len_t:] - t[len_t]

    # Preprocess
    tracks_array = preprocess_tracks(tracks_array)
    return tracks_array, positions


def track_generator_segmentation(n, dimensions):
    while True:
        tracks, positions = generate_tracks_segmentation(n, dimensions=dimensions)
        yield tracks, positions


"""
Misc

"""


def preprocess_tracks(tracks):
    """
    Return normalised differences

    """

    if tracks.shape[2] == 1:
        diff = np.diff(tracks[:, :, 0], axis=1)
        meanstep = np.sum(abs(diff), axis=1) / np.sum(tracks[:, :, 0] != 0, axis=1)
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
