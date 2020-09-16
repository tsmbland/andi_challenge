import andi
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from andi_funcs import preprocess_tracks
from sklearn.metrics import confusion_matrix

"""
Distance seems higher for 2D tracks even though accuracy is higher for validation data
Something isn't right

"""

"""
1D

"""

AD = andi.andi_datasets()
X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=1000, tasks=3, dimensions=[1], save_dataset=True, min_T=200, max_T=201)
positions_1D = np.array(Y3[0])[:, 1].astype(int) - 1
tracks_1D = X3[0]
model = load_model('../Models/1D.h5')
res_1D = np.zeros(len(positions_1D))

for i, t in enumerate(tracks_1D):
    track = np.array(t)
    track = np.expand_dims(track, axis=-1)
    track = np.expand_dims(track, axis=0)
    track = preprocess_tracks(track)
    res_1D[i] = np.argmax(model.predict(track))

rmse = np.mean((res_1D - positions_1D) ** 2) ** 0.5
print(rmse)

conf = confusion_matrix(positions_1D, res_1D, labels=np.arange(200))
plt.imshow(conf)
plt.show()

"""
2D

"""

AD = andi.andi_datasets()
X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=1000, tasks=3, dimensions=[2], save_dataset=True, min_T=200, max_T=201)
positions_2D = np.array(Y3[1])[:, 1].astype(int) - 1
tracks_2D = X3[1]
model = load_model('../Models/2D.h5')
res_2D = np.zeros(len(positions_2D))

for i, t in enumerate(tracks_2D):
    track = np.array(t)
    track = np.dstack((track[:len(track) // 2], track[len(track) // 2:]))
    track = preprocess_tracks(track)
    res_2D[i] = np.argmax(model.predict(track))

rmse = np.mean((res_2D - positions_2D) ** 2) ** 0.5
print(rmse)

conf = confusion_matrix(positions_2D, res_2D, labels=np.arange(200))
plt.imshow(conf)
plt.show()
