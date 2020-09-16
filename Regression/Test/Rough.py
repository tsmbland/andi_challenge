import andi
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from andi_funcs import preprocess_tracks

"""
1D

"""
# AD = andi.andi_datasets()
# X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=1000, tasks=1, dimensions=[1], save_dataset=True, min_T=10, max_T=1001)
# # X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=100, tasks=1, dimensions=[1], load_dataset=True, min_T=999, max_T=1000)
# exponents_1D = np.array(Y1[0])
# tracks_1D = X1[0]
# model = load_model('../Models/1D.h5')
# res_1D = np.zeros(len(exponents_1D))
#
# for i, t in enumerate(tracks_1D):
#     track = np.array(t)
#     track = np.expand_dims(track, axis=-1)
#     track = np.expand_dims(track, axis=0)
#     track = preprocess_tracks(track)
#     a = model.predict(track)
#     plt.scatter(exponents_1D[i], a)
#     res_1D[i] = a
#
# mae = np.mean(abs(exponents_1D - res_1D))
# print(mae)
#
# plt.plot([0, 2], [0, 2])
# plt.show()

"""
2D

"""

AD = andi.andi_datasets()
X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=1000, tasks=1, dimensions=[2], save_dataset=True, min_T=10, max_T=1001)
# X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=100, tasks=1, dimensions=[1], load_dataset=True, min_T=999, max_T=1000)
exponents_2D = np.array(Y1[1])
tracks_2D = X1[1]
model = load_model('../Models/2D.h5')
res_2D = np.zeros(len(exponents_2D))

for i, t in enumerate(tracks_2D):
    track = np.array(t)
    track = np.dstack((track[:len(track) // 2], track[len(track) // 2:]))
    track = preprocess_tracks(track)
    a = model.predict(track)
    plt.scatter(exponents_2D[i], a)
    res_2D[i] = a

mae = np.mean(abs(exponents_2D - res_2D))
print(mae)

plt.plot([0, 2], [0, 2])
plt.show()
