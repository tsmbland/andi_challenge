from andi_funcs import *
import matplotlib.pyplot as plt

from tensorflow import reverse, tile, shape, expand_dims, cast, float32, constant
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, Add, Input, concatenate, GlobalMaxPooling1D, \
    Average, GlobalMaxPooling1D, Multiply
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint

"""
Notes:

Network trained on short tracks (e.g. 20) tends to underestimate alpha predictions for longer tracks (100)
- switching GlobalMaxPooling to GlobalAveragePooling makes this much worse

Model trained on tracks from 10-100
- seems a bit worse than RNN for short tracks (20: 0.116 vs 0.096)
- long tracks: about the same (100: 0.063 vs 0.069)

Model trained on 10-30
- does much better on 20-step tracks

When network is just trained on either 20 and 100 step tracks
- not bad for 20 (~0.1), but still worse than a single network (~0.9)

Could try concatenating the length of the sequence to the concatenated global layer
- in theory this should allow it to factor in the length of the sequence when deciding alpha at the dense layers

Could try removing causal convolutions

I think a smaller batch size would help
- 64 > 32 improves

I think gradually decreasing learning rate would help, or just make it lower from the start

Removing batchnorm
- doesn't seem to help

"""


def get_model(blocks):
    inputs = Input((None, 2))
    f = 32

    # Conv block 1: receptive field = 8
    if 1 in blocks:
        x1 = Conv1D(f, 2, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(f, 2, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(f, 2, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x1 = Add()([x1_bypass, x1])
        x1 = GlobalMaxPooling1D()(x1)
    else:
        x1 = None

    # Conv block 2: receptive field = 15
    if 2 in blocks:
        x2 = Conv1D(f, 3, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(f, 3, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(f, 3, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(x2)
        x2 = BatchNormalization()(x2)
        x2_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x2 = Add()([x2_bypass, x2])
        x2 = GlobalMaxPooling1D()(x2)
    else:
        x2 = None

    # Conv block 3: receptive field = 22
    if 3 in blocks:
        x3 = Conv1D(f, 4, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(f, 4, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(f, 4, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(x3)
        x3 = BatchNormalization()(x3)
        x3_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x3 = Add()([x3_bypass, x3])
        x3 = GlobalMaxPooling1D()(x3)
    else:
        x3 = None

    # Conv block 4: receptive field = 145
    if 4 in blocks:
        x4 = Conv1D(f, 10, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(f, 10, dilation_rate=5, padding='causal', activation='relu', kernel_initializer='he_normal')(x4)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(f, 10, dilation_rate=10, padding='causal', activation='relu', kernel_initializer='he_normal')(x4)
        x4 = BatchNormalization()(x4)
        x4_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x4 = Add()([x4_bypass, x4])
        x4 = GlobalMaxPooling1D()(x4)
    else:
        x4 = None

    # Concatenate conv blocks
    con = concatenate([i for i in [x1, x2, x3, x4] if i is not None])

    # Input length
    seq_len = expand_dims(tile(cast(shape(inputs)[1:2], float32) / constant(100, dtype=float32), shape(inputs)[0:1]),
                          -1)
    # seq_len = Dense(1, activation='relu')(seq_len)

    # Input length dense
    seq_len = Dense(64, activation='relu')(seq_len)
    seq_len = Dense(128, activation='softmax')(seq_len)

    # Concatenate sequence length
    # con = concatenate([con, seq_len])

    # Multiply layers
    con = Multiply()([con, seq_len])

    # Dense layers
    dense = Dense(512, activation='relu')(con)
    dense = Dense(256, activation='relu')(dense)

    # Output
    dense2 = Dense(1, activation='linear')(dense)

    # Model
    model = Model(inputs=inputs, outputs=dense2)
    return model


def whole_model(blocks=(1, 2, 3, 4)):
    inputs = Input((None, 2))
    model = get_model(blocks)
    conv_out1 = model(inputs)
    conv_out2 = model(reverse(inputs, axis=[2]))
    out = Average()([conv_out1, conv_out2])
    model = Model(inputs=inputs, outputs=out)
    return model


# Validation data
val = generate_tracks(n=1000, track_length_vals=[20])

# Run model
model = whole_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()
history = model.fit(track_generator(32, track_length_vals=[20, 100]), steps_per_epoch=200, epochs=10,
                    callbacks=[
                        ModelCheckpoint(filepath='ModelCNN_20_100_adam_multiply_padding.h5', monitor='mae',
                                        save_best_only=True,
                                        mode='min')], validation_data=val)

# np.savetxt('mae2.txt', history.history['mae'])
# np.savetxt('val_mae2.txt', history.history['val_mae'])

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.show()

# """
# Test model
#
# """
# model = load_model('ModelCNN_10to100.h5')
#
# # Test data
# test_tracks, test_alphas = generate_tracks(n=1000, track_length_vals=[20])
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
