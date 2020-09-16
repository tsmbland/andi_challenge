from andi_funcs import *
import matplotlib.pyplot as plt

from tensorflow import reverse, tile, shape, expand_dims, cast, float32, constant
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, Add, Input, concatenate, GlobalMaxPooling1D, \
    Average, GlobalAveragePooling1D, Flatten, Softmax
# from tf.math import divide
from tensorflow.keras.callbacks import ModelCheckpoint

"""
Notes:

Based on previous CNN but no global pooling as this loses spatial information
Instead, all conv output layers are combined into a single layer using 1x1 convolutions
This is then fed into dense layer
Two outputs of dense layer (both sigmoid):
- probability of a switch (not for now)
- position

I think this method has potential. A few possible improvements
- perhaps one-hot encoding doesn't make sense as it's so sparse. maybe do something like trigger word detection?

I think changing padding from causal to same helps things
- a fair bit actually, might make sense to change for other models



"""


def get_model(length, blocks):
    inputs = Input((length, 2))
    f = 32

    # Conv block 1: receptive field = 8
    if 1 in blocks:
        x1 = Conv1D(f, 2, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(f, 2, dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(f, 2, dilation_rate=4, padding='same', activation='relu', kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x1 = Add()([x1_bypass, x1])
    else:
        x1 = None

    # Conv block 2: receptive field = 15
    if 2 in blocks:
        x2 = Conv1D(f, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(f, 3, dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(f, 3, dilation_rate=4, padding='same', activation='relu', kernel_initializer='he_normal')(x2)
        x2 = BatchNormalization()(x2)
        x2_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x2 = Add()([x2_bypass, x2])
    else:
        x2 = None

    # Conv block 3: receptive field = 22
    if 3 in blocks:
        x3 = Conv1D(f, 4, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(f, 4, dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(f, 4, dilation_rate=4, padding='same', activation='relu', kernel_initializer='he_normal')(x3)
        x3 = BatchNormalization()(x3)
        x3_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x3 = Add()([x3_bypass, x3])
    else:
        x3 = None

    # Conv block 4: receptive field = 145
    if 4 in blocks:
        x4 = Conv1D(f, 10, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(f, 10, dilation_rate=5, padding='same', activation='relu', kernel_initializer='he_normal')(x4)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(f, 10, dilation_rate=10, padding='same', activation='relu', kernel_initializer='he_normal')(x4)
        x4 = BatchNormalization()(x4)
        x4_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x4 = Add()([x4_bypass, x4])
    else:
        x4 = None

    # Concatenate conv blocks
    con = concatenate([i for i in [x1, x2, x3, x4] if i is not None])

    # 1x1 filter
    con = Conv1D(32, 1, activation='relu', kernel_initializer='he_normal')(con)
    x5 = Conv1D(1, 1, kernel_initializer='he_normal')(con)

    # Location
    out2 = Softmax()(Flatten()(x5))

    # Model
    model = Model(inputs=inputs, outputs=out2)
    return model


def whole_model(length=200, blocks=(1, 2, 3, 4)):
    inputs = Input((length, 2))
    model = get_model(length, blocks)
    conv_out1 = model(inputs)
    conv_out2 = model(reverse(inputs, axis=[2]))
    out = Average()([conv_out1, conv_out2])
    model = Model(inputs=inputs, outputs=out)
    return model


# Validation data
val = generate_switching_tracks(n=100, track_length=200)

# Run model
model = whole_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(switching_track_generator(n=32, track_length=200), steps_per_epoch=200, epochs=10,
                    callbacks=[
                        ModelCheckpoint(filepath='ModelCNN_segmentation.h5', monitor='val_accuracy',
                                        save_best_only=True,
                                        mode='max')], validation_data=val)

# np.savetxt('cnn_mae.txt', history.history['mae'])
# np.savetxt('cnn_val_mae.txt', history.history['val_mae'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

# """
# Test model
#
# """
# model = load_model('ModelCNN_segmentation.h5')
#
# # Test data
# test_tracks, test_switchpoints = generate_switching_tracks(n=100, track_length=200)
# predicted = model.predict(test_tracks)
#
# test_one_hot = np.eye(200)[test_switchpoints]
#
# # plt.imshow()
# # plt.show()
# #
# # plt.imshow(predicted)
# # plt.show()
#
# for i in range(100):
#     plt.plot(test_tracks[i, :, 0])
#     plt.axvline(test_switchpoints[i])
#
#     if np.max(predicted[i]) > 0.1:
#         plt.axvline(np.argmax(predicted[i]), c='g')
#     else:
#         plt.axvline(np.argmax(predicted[i]), c='r')
#
#     plt.title('{0:.2f}'.format(np.max(predicted[i])))
#     plt.show()
#
#     # plt.plot(test_one_hot[i])
#     # plt.plot(predicted[i])
#     # plt.show()
#
# # # print(predicted)
# # # print(predicted.shape)
# # #
# # # # Calculate mae
# # # mae = np.mean(abs(predicted - test_alphas))
# # # print(mae)
# # #
# # # plt.scatter(test_alphas, predicted, s=1)
# # # plt.plot([0, 1], [0, 1])
# # # plt.show()
