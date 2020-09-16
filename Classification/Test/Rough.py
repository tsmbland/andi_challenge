import andi
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from andi_funcs import preprocess_tracks
from sklearn.metrics import f1_score

"""
1D

"""

# AD = andi.andi_datasets()
# X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=1000, tasks=2, dimensions=[1], save_dataset=True, min_T=10, max_T=1001)
# # X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=100, tasks=1, dimensions=[1], load_dataset=True, min_T=999, max_T=1000)
# classes_1D = np.array(Y2[0]).astype(int)
# tracks_1D = X2[0]
# model = load_model('../Models/1D.h5')
# res_1D = np.zeros(len(classes_1D))
#
# for i, t in enumerate(tracks_1D):
#     track = np.array(t)
#     track = np.expand_dims(track, axis=-1)
#     track = np.expand_dims(track, axis=0)
#     track = preprocess_tracks(track)
#     res_1D[i] = np.argmax(model.predict(track))
#
# f1 = f1_score(classes_1D, res_1D, average='micro')
# print(f1)

"""
2D

"""

AD = andi.andi_datasets()
X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=1000, tasks=2, dimensions=[2], save_dataset=True, min_T=10, max_T=1001)
classes_2D = np.array(Y2[1]).astype(int)
tracks_2D = X2[1]
model = load_model('../Models/2D.h5')
res_2D = np.zeros(len(classes_2D))

for i, t in enumerate(tracks_2D):
    track = np.array(t)
    track = np.dstack((track[:len(track) // 2], track[len(track) // 2:]))
    track = preprocess_tracks(track)
    res_2D[i] = np.argmax(model.predict(track))

f1 = f1_score(classes_2D, res_2D, average='micro')
print(f1)
