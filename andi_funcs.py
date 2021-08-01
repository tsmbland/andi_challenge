import numpy as np
import andi
import csv
from tensorflow.keras.utils import Sequence

"""
Dataset generators

"""


def generate_tracks_regression(n, dimensions, min_T=5, max_T=1001):
    """
    Generate tracks for training regression model

    Parameters:
    n: number of tracks to generate
    dimensions: number of dimensions (currently only supports 1 and 2)
    min_T: minimum track length
    max_T: maximum track length (e.g. for 1001 will generate tracks up to 1000 steps)

    Returns:
    tracks_array: a numpy array of shape [n, max_T, dimensions] containing the generated tracks
    exponents: a numpy array of length n, containing the anomalous exponent value for each track

    """
    # Create tracks
    np.random.seed()  # prevents data duplication
    AD = andi.andi_datasets()
    X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=n, min_T=min_T, max_T=max_T, tasks=[1], dimensions=[dimensions])
    exponents = np.array(Y1[dimensions - 1])
    tracks = X1[dimensions - 1]

    # Package into array and preprocess
    tracks_array = package_tracks(tracks=tracks, max_T=max_T, dimensions=dimensions)
    return tracks_array, exponents


class TrackGeneratorRegression(Sequence):
    def __init__(self, batches, batch_size, dimensions, min_T, max_T):
        self.batches = batches
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.min_T = min_T
        self.max_T = max_T

    def __getitem__(self, item):
        tracks, exponents = generate_tracks_regression(self.batch_size, dimensions=self.dimensions, min_T=self.min_T,
                                                       max_T=self.max_T)
        return tracks, exponents

    def __len__(self):
        return self.batches


def generate_tracks_classification(n, dimensions, min_T=5, max_T=1001):
    """
    Generate tracks for training classification model

    Parameters:
    n: number of tracks to generate
    dimensions: number of dimensions (currently only supports 1 and 2)
    min_T: minimum track length
    max_T: maximum track length (e.g. for 1001 will generate tracks up to 1000 steps)

    Returns:
    tracks_array: a numpy array of shape [n, max_T, dimensions] containing the generated tracks
    classes: a numpy array of length n, representing the model class for each track (see andi_datasets package)

    """

    # Create tracks
    np.random.seed()
    AD = andi.andi_datasets()
    X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=n, min_T=min_T, max_T=max_T, tasks=[2], dimensions=[dimensions])
    classes = np.array(Y2[dimensions - 1]).astype(int)
    tracks = X2[dimensions - 1]

    # Package into array and preprocess
    tracks_array = package_tracks(tracks=tracks, max_T=max_T, dimensions=dimensions)
    return tracks_array, classes


class TrackGeneratorClassification(Sequence):
    def __init__(self, batches, batch_size, dimensions, min_T, max_T):
        self.batches = batches
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.min_T = min_T
        self.max_T = max_T

    def __getitem__(self, item):
        tracks, classes = generate_tracks_classification(self.batch_size, dimensions=self.dimensions, min_T=self.min_T,
                                                         max_T=self.max_T)
        return tracks, classes

    def __len__(self):
        return self.batches


def generate_tracks_segmentation(n, dimensions):
    """
    Generate tracks for training segmentation model (all length 200)

    Parameters:
    n: number of tracks to generate
    dimensions: number of dimensions (currently only supports 1 and 2)

    Returns:
    tracks_array: a numpy array of shape [n, 200, dimensions] containing the generated tracks
    positions: a numpy array of length n, representing the switch point for each model

    """

    # Create tracks
    np.random.seed()
    AD = andi.andi_datasets()
    X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=n, tasks=[3], dimensions=[dimensions], min_T=200, max_T=201)
    positions = np.array(Y3[dimensions - 1])[:, 1].astype(int) - 1
    tracks = X3[dimensions - 1]

    # Package into array and preprocess
    tracks_array = package_tracks(tracks=tracks, max_T=200, dimensions=dimensions)
    return tracks_array, positions


class TrackGeneratorSegmentation(Sequence):
    def __init__(self, batches, batch_size, dimensions):
        self.batches = batches
        self.batch_size = batch_size
        self.dimensions = dimensions

    def __getitem__(self, item):
        tracks, positions = generate_tracks_segmentation(self.batch_size, dimensions=self.dimensions)
        return tracks, positions

    def __len__(self):
        return self.batches


"""
Track processing

"""


def package_tracks(tracks, max_T, dimensions):
    """
    Convert tracks from list format (i.e. output from andi_datasets) to numpy array
    This requires shorter tracks to be padded with zeros (up to length max_T)

    Parameters:
    tracks: tracks in list format (i.e. output from andi_datasets function)
    max_T: the maximum track length
    dimensions: number of track dimensions (1 or 2 supported)

    Returns:
    tracks_array: a numpy array containing padded and preprocessed tracks

    """

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


def preprocess_tracks(tracks):
    """
    Preprocess tracks by taking the difference between successive positions and normalising (dividing by the mean step
    size) -> input to models

    Parameters:
    tracks: numpy array of tracks (output from one of the above functions)

    Returns:
    tracks_processed: a numpy array of shape [n, max_T-1, d]

    """

    # 1D tracks
    if tracks.shape[2] == 1:
        diff = np.diff(tracks[:, :, 0], axis=1)
        meanstep = np.sum(abs(diff), axis=1) / np.sum(tracks[:, :, 0] != 0, axis=1)
        tracks_processed = np.expand_dims(diff / np.expand_dims(meanstep, axis=-1), axis=-1)

    # 2D tracks
    elif tracks.shape[2] == 2:
        dx = np.diff(tracks[:, :, 0], axis=1)
        dy = np.diff(tracks[:, :, 1], axis=1)
        meanstep = np.expand_dims(np.sum(((dx ** 2) + (dy ** 2)) ** 0.5, axis=1) / np.sum(tracks[:, :, 0] != 0, axis=1),
                                  axis=-1)
        tracks_processed = np.dstack((dx / meanstep, dy / meanstep))

    return tracks_processed


def split_tracks(tracks, positions, dimensions=1, max_T=200):
    """
    Split tracks according to positions (i.e. output from a segmentation model)

    Parameters:
    tracks: tracks in list format (i.e. output from andi_datasets)
    dimensions: number of track dimensions (1 and 2 supported)
    max_T: maximum track length

    Returns:
    split_tracks_array: numpy array of processed split tracks, shape [n * 2, max_T-1, dimensions]

    """

    g = 0  # can set to > 0 to exclude points in the immediate vicinity of the switchpoint
    split_tracks_array = np.zeros([len(tracks) * 2, max_T, dimensions])

    # 1D tracks
    if dimensions == 1:
        i = 0  # counter
        for j, track in enumerate(tracks):
            split_tracks_array[i, max_T - max(positions[j] - g, 0):, 0] = track[:max(positions[j] - g, 0)]
            split_tracks_array[i + 1, min(positions[j] + g, 199):, 0] = track[min(positions[j] + g, 199):] - track[
                min(positions[j] + g, 199)]
            i += 2

    # 2D tracks
    elif dimensions == 2:
        i = 0  # counter
        for j, track in enumerate(tracks):
            len_t = int(len(track) / 2)
            d1 = track[:len_t].flatten()
            d2 = track[len_t:].flatten() - track[len_t]
            split_tracks_array[i, max_T - max(positions[j] - g, 0):, 0] = d1[:max(positions[j] - g, 0)]
            split_tracks_array[i, max_T - max(positions[j] - g, 0):, 1] = d2[:max(positions[j] - g, 0)]
            split_tracks_array[i + 1, min(positions[j] + g, 199):, 0] = d1[min(positions[j] + g, 199):] - d1[
                min(positions[j] + g, 199)]
            split_tracks_array[i + 1, min(positions[j] + g, 199):, 1] = d2[min(positions[j] + g, 199):] - d2[
                min(positions[j] + g, 199)]
            i += 2

    # Preprocess
    split_tracks_array = preprocess_tracks(split_tracks_array)
    return split_tracks_array


"""
File handling

"""


def import_tracks(path):
    """
    Import tracks saved in the competition format. NB only imports 1D and 2D tracks

    Parameters:
    path: path to file

    Returns:
    1D and 2D trajectories in list format

    """

    t = csv.reader(open(path, 'r'), delimiter=';', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
    X = [[], []]
    for trajs in t:
        if int(trajs[0]) in [1, 2]:
            X[int(trajs[0]) - 1].append(trajs[1:])
    return X[0], X[1]


def import_labels(direc):
    """
    Import labels saved in the competition format. NB only imports 1D and 2D tracks
    For task 1 this is the exponent
    For task 2 this is the model
    For task 3 this is the switchpoint ONLY

    Parameters:
    path: path to file

    Returns:
    Labels for 1D and 2D tracks

    """

    l = csv.reader(open(direc, 'r'), delimiter=';', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
    Y = [[], []]
    for labels in l:
        if int(labels[0]) in [1, 2]:
            Y[int(labels[0]) - 1].append(labels[1])
    return np.array(Y[0]), np.array(Y[1])


"""
Other

"""


def rolling_ave(array, window):
    """
    Apply a rolling average to a 1D array. Rolling average window specified by window parameter. Can be useful to apply
    to output of segmentation CNN, but not strictly necessary

    """

    array_padded = np.c_[array[:, :int(window / 2)][:, :-1], array, array[:, -int(window / 2):][:, :-1]]
    cumsum = np.cumsum(array_padded, axis=1)
    return (cumsum[:, window:] - cumsum[:, :-window]) / window
