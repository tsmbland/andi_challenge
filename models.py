from tensorflow import reverse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, Add, Input, concatenate, GlobalMaxPooling1D, \
    Softmax, Flatten
import tensorflow as tf

"""
CNN

"""


def conv_blocks(dimensions, blocks, length=None):
    """
    Convolutional blocks for exponent and classification analysis
    Adapted from CNN described in Granik et al 2019 and Bai et al 2018

    """

    inputs = Input((length, dimensions))
    f = 64

    # Conv block 1: receptive field = 16
    if 1 in blocks:
        x1 = Conv1D(f, 2, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(f, 2, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(f, 2, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(f, 2, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x1 = Add()([x1_bypass, x1])
    else:
        x1 = None

    # Conv block 2: receptive field = 31
    if 2 in blocks:
        x2 = Conv1D(f, 3, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(f, 3, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(f, 3, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(f, 3, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(x2)
        x2 = BatchNormalization()(x2)
        x2_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x2 = Add()([x2_bypass, x2])
    else:
        x2 = None

    # Conv block 3: receptive field = 46
    if 3 in blocks:
        x3 = Conv1D(f, 4, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(f, 4, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(f, 4, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(f, 4, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(x3)
        x3 = BatchNormalization()(x3)
        x3_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x3 = Add()([x3_bypass, x3])
    else:
        x3 = None

    # Conv block 4: receptive field = 136
    if 4 in blocks:
        x4 = Conv1D(f, 10, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(f, 10, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(x4)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(f, 10, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(x4)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(f, 10, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(x4)
        x4 = BatchNormalization()(x4)
        x4_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x4 = Add()([x4_bypass, x4])
    else:
        x4 = None

    # Conv block 5: receptive field = 286
    if 5 in blocks:
        x5 = Conv1D(f, 20, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        x5 = BatchNormalization()(x5)
        x5 = Conv1D(f, 20, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(x5)
        x5 = BatchNormalization()(x5)
        x5 = Conv1D(f, 20, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(x5)
        x5 = BatchNormalization()(x5)
        x5 = Conv1D(f, 20, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(x5)
        x5 = BatchNormalization()(x5)
        x5_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x5 = Add()([x5_bypass, x5])
    else:
        x5 = None

    # Concatenate conv blocks
    con = concatenate([i for i in [x1, x2, x3, x4, x5] if i is not None])
    return Model(inputs=inputs, outputs=con)


def conv_blocks_for_seg(dimensions, blocks, length=None):
    """
    Convolutional blocks used for segmentation
    Unlike the conv_blocks function, this uses standard convolutions rather than causal convolutions

    """
    inputs = Input((length, dimensions))
    f = 64

    # Conv block 1: receptive field = 31
    if 1 in blocks:
        x1 = Conv1D(f, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(f, 3, dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(f, 3, dilation_rate=4, padding='same', activation='relu', kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(f, 3, dilation_rate=8, padding='same', activation='relu', kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x1 = Add()([x1_bypass, x1])
    else:
        x1 = None

    # Conv block 2: receptive field = 61
    if 2 in blocks:
        x2 = Conv1D(f, 5, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(f, 5, dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(f, 5, dilation_rate=4, padding='same', activation='relu', kernel_initializer='he_normal')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(f, 5, dilation_rate=8, padding='same', activation='relu', kernel_initializer='he_normal')(x2)
        x2 = BatchNormalization()(x2)
        x2_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x2 = Add()([x2_bypass, x2])
    else:
        x2 = None

    # Conv block 3: receptive field = 91
    if 3 in blocks:
        x3 = Conv1D(f, 7, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(f, 7, dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(f, 7, dilation_rate=4, padding='same', activation='relu', kernel_initializer='he_normal')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(f, 7, dilation_rate=8, padding='same', activation='relu', kernel_initializer='he_normal')(x3)
        x3 = BatchNormalization()(x3)
        x3_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x3 = Add()([x3_bypass, x3])
    else:
        x3 = None

    # Conv block 4: receptive field = 211
    if 4 in blocks:
        x4 = Conv1D(f, 15, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(f, 15, dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')(x4)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(f, 15, dilation_rate=4, padding='same', activation='relu', kernel_initializer='he_normal')(x4)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(f, 15, dilation_rate=8, padding='same', activation='relu', kernel_initializer='he_normal')(x4)
        x4 = BatchNormalization()(x4)
        x4_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        x4 = Add()([x4_bypass, x4])
    else:
        x4 = None

    # Concatenate conv blocks
    con = concatenate([i for i in [x1, x2, x3, x4] if i is not None])
    return Model(inputs=inputs, outputs=con)


"""
Exponent

"""


def regression_model_1d(blocks=(1, 2, 3, 4)):
    inputs = Input((None, 1))

    # Convolutions
    c = GlobalMaxPooling1D()(conv_blocks(dimensions=1, blocks=blocks)(inputs))

    # Dense layers
    dense = Dense(512, activation='relu')(c)
    dense = Dense(256, activation='relu')(dense)
    out = Dense(1, activation='linear')(dense)

    # Full model
    model = Model(inputs=inputs, outputs=out)
    return model


def regression_model_2d(blocks=(1, 2, 3, 4)):
    inputs = Input((None, 2))

    # Convolutions
    c1 = GlobalMaxPooling1D()(conv_blocks(dimensions=2, blocks=blocks)(inputs))
    c2 = GlobalMaxPooling1D()(conv_blocks(dimensions=2, blocks=blocks)(reverse(inputs, axis=[2])))
    c = tf.math.maximum(c1, c2)

    # Dense layers
    dense = Dense(512, activation='relu')(c)
    dense = Dense(256, activation='relu')(dense)
    out = Dense(1, activation='linear')(dense)

    # Full model
    model = Model(inputs=inputs, outputs=out)
    return model


"""
Classification

"""


def classification_model_1d(blocks=(1, 2, 3, 4)):
    inputs = Input((None, 1))

    # Convolutions
    c = GlobalMaxPooling1D()(conv_blocks(dimensions=1, blocks=blocks)(inputs))

    # Dense layers
    dense = Dense(512, activation='relu')(c)
    dense = Dense(256, activation='relu')(dense)
    out = Dense(5, activation='softmax')(dense)

    # Full model
    model = Model(inputs=inputs, outputs=out)
    return model


def classification_model_2d(blocks=(1, 2, 3, 4)):
    inputs = Input((None, 2))

    # Convolutions
    c1 = GlobalMaxPooling1D()(conv_blocks(dimensions=2, blocks=blocks)(inputs))
    c2 = GlobalMaxPooling1D()(conv_blocks(dimensions=2, blocks=blocks)(reverse(inputs, axis=[2])))
    c = tf.math.maximum(c1, c2)

    # Dense layers
    dense = Dense(512, activation='relu')(c)
    dense = Dense(256, activation='relu')(dense)
    out = Dense(5, activation='softmax')(dense)

    # Full model
    model = Model(inputs=inputs, outputs=out)
    return model


"""
Segmentation

"""


def segmentation_model_1d(blocks=(1, 2, 3, 4)):
    inputs = Input((199, 1))

    # Convolutions
    c = conv_blocks_for_seg(dimensions=1, blocks=blocks, length=199)(inputs)

    # 1x1 filter
    con = Conv1D(512, 1, activation='relu')(c)
    con = Conv1D(256, 1, activation='relu')(con)
    x5 = Conv1D(1, 1)(con)
    out = Softmax()(Flatten()(x5))

    # Full model
    model = Model(inputs=inputs, outputs=out)
    return model


def segmentation_model_2d(blocks=(1, 2, 3, 4)):
    inputs = Input((199, 2))

    # Convolutions
    c1 = conv_blocks_for_seg(dimensions=2, blocks=blocks, length=199)(inputs)
    c2 = conv_blocks_for_seg(dimensions=2, blocks=blocks, length=199)(reverse(inputs, axis=[2]))
    c = tf.math.add(c1, c2)

    # 1x1 filter
    con = Conv1D(512, 1, activation='relu')(c)
    con = Conv1D(256, 1, activation='relu')(con)
    x5 = Conv1D(1, 1)(con)
    out = Softmax()(Flatten()(x5))

    # Full model
    model = Model(inputs=inputs, outputs=out)
    return model
