import numpy as np
import andi
import matplotlib.pyplot as plt
from andi_funcs import *

# AD = andi.andi_datasets()
# dataset1 = AD.create_dataset(T=200, N=20, exponents=[0.4], models=[2])
# dataset2 = AD.create_dataset(T=200, N=20, exponents=[1.6], models=[2])
# seg = AD.create_segmented_dataset(dataset1, dataset2)
#
# print(seg[:, :5])


# def generate_switching_tracks(n, track_length=200):
#     AD = andi.andi_datasets()
#     dataset1 = AD.create_dataset(T=track_length, N=n, exponents=[0.4], models=[2])
#     dataset2 = AD.create_dataset(T=track_length, N=n, exponents=[1.6], models=[2])
#     d = AD.create_segmented_dataset(dataset1, dataset2)
#     tracks = np.array(d[:, 5:])
#     tracks = np.dstack((tracks[:, :tracks.shape[1] // 2], tracks[:, tracks.shape[1] // 2:]))
#
#     # One hot array of switchpoints
#     switchpoints = d[:, 0].astype(int)
#     y = np.eye(track_length)[switchpoints]
#     return tracks, y


a, b, c, d = generate_switching_tracks(n=10, track_length=200)

print(a.shape)

for i, j in enumerate(a):
    print(c[i])
    print(d[i])

    plt.plot(j[:, 0])
    plt.axvline(b[i])
    plt.show()
