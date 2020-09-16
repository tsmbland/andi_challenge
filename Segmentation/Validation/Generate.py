from andi_funcs import *

"""
Global parameters

"""

n = 10000

"""
1D

"""

tracks_1D, positions_1D = generate_tracks_segmentation(n=n, dimensions=1, track_length=200)
save_tracks(tracks_1D, dims=1, path='tracks_1D.txt')
np.savetxt('positions_1D.txt', positions_1D)


"""
2D

"""

tracks_2D, positions_2D = generate_tracks_segmentation(n=n, dimensions=2, track_length=200)
save_tracks(tracks_2D, dims=2, path='tracks_2D.txt')
np.savetxt('positions_2D.txt', positions_2D)
