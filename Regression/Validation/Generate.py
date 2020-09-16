from andi_funcs import *

"""
Global parameters

"""

n = 10000

"""
1D

"""

tracks_1D, exponents_1D = generate_tracks_regression(n=n, dimensions=1, min_T=7, max_T=200)
save_tracks(tracks_1D, dims=1, path='tracks_1D.txt')
np.savetxt('exponents_1D.txt', exponents_1D)

"""
2D

"""

tracks_2D, exponents_2D = generate_tracks_regression(n=n, dimensions=2, min_T=7, max_T=200)
save_tracks(tracks_2D, dims=2, path='tracks_2D.txt')
np.savetxt('exponents_2D.txt', exponents_2D)

