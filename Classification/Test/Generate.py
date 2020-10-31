from andi_funcs import *

"""
Global parameters

"""

n = 10000

"""
1D

"""

tracks_1D, classes_1D = generate_tracks_classification(n=n, dimensions=1, min_T=10, max_T=1001)
save_tracks(tracks_1D, dims=1, path='tracks_1D.txt')
np.savetxt('classes_1D.txt', classes_1D)

"""
2D

"""

tracks_2D, classes_2D = generate_tracks_classification(n=n, dimensions=2, min_T=10, max_T=1001)
save_tracks(tracks_2D, dims=2, path='tracks_2D.txt')
np.savetxt('classes_2D.txt', classes_2D)
