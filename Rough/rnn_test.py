import andi
from andi_funcs import *
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Average, Conv1D, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import reverse

"""
Notes

Network trained on short tracks (e.g. 20) tends to underestimate alpha predictions for longer tracks (100)
- at 1000, it has no clue

Training on mixed length tracks (20 and 100)
- seems to do a bit worse on short tracks (20) than a network trained on short tracks
- may just take longer though as it has less training data of this length
- does surprisingly well on intermediate data (50)
- poorly on data outside of range (200)

One hyperparameter might be max size of training tracks
- there will come a point where covering large timespans isn't helpful
- so e.g. could cover 1-100. For longer tracks would do rolling window and average all measurements

Training on data between 10 and 100
- does pretty well, although a tiny bit worse on tracks of length 20 compared to network trained on 20-step tracks
- could be that the network isn't big enough to deal with different length tracks


"""


def get_model():
    # Inputs
    inputs = Input((None, 2))

    # LSTM layers
    x1 = LSTM(64, return_sequences=True)(inputs)
    x2 = LSTM(16)(x1)

    # Dense
    dense = Dense(1, activation='linear')(x2)

    # Model
    model = Model(inputs=inputs, outputs=dense)
    return model


def whole_model():
    inputs = Input((None, 2))
    model = get_model()
    out1 = model(inputs)
    out2 = model(reverse(inputs, axis=[2]))
    out = Average()([out1, out2])
    model = Model(inputs=inputs, outputs=out)
    return model


# # # Validation data
# val = generate_tracks(n=1000, track_length_vals=[20])
#
# # Run model
# model = whole_model()
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.summary()
# history = model.fit(track_generator(64, track_length_vals=range(10, 101)), steps_per_epoch=100, epochs=100,
#                     callbacks=[ModelCheckpoint(filepath='ModelRNN_10t0100.h5', monitor='val_mae', save_best_only=True,
#                                                mode='min')], validation_data=val)
#
# np.savetxt('mae.txt', history.history['mae'])
# np.savetxt('val_mae.txt', history.history['val_mae'])
plt.plot(np.loadtxt('mae.txt'))
plt.plot(np.loadtxt('val_mae.txt'))
plt.show()

# """
# Test model
#
# """
# model = load_model('ModelRNN_10t0100.h5')
#
# # Test data
# test_tracks, test_alphas = generate_tracks(n=1000, track_length_vals=[100])
#
# predicted = model.predict(test_tracks).flatten()
#
# # Calculate mae
# mae = np.mean(abs(predicted - test_alphas))
# print(mae)
#
# plt.scatter(test_alphas, predicted, s=1)
# plt.plot([0, 1], [0, 1])
# plt.show()
