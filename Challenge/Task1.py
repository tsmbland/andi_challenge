from tensorflow.keras.models import load_model
from andi_funcs import import_tracks, package_tracks
import os

# Specify path to track data folder containing task1.txt
path = ''

"""
Import data

"""

# Import x data
tracks_1D, tracks_2D = import_tracks(path + '/task1.txt')

"""
1D

"""

model = load_model('../Task1_Exponent/Models/1D.h5')
res_1D = model.predict(package_tracks(tracks_1D, dimensions=1, max_T=2001)).flatten()
# max_T set to 2001 as competition data contains some tracks longer than 1000

"""
2D

"""

model = load_model('../Task1_Exponent/Models/2D.h5')
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