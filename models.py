from tensorflow import reverse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, Add, Input, concatenate, GlobalMaxPooling1D, \
    Softmax, Flatten
import tensorflow as tf

"""
Convolutional blocks

"""


def conv_blocks(dimensions, blocks, length=None):
    """
    Convolutional blocks for exponent and classification analysis
    Adapted from CNN described in Granik et al 2019, based on architecture from Bai et al 2018

    """

    inputs = Input((length, dimensions))
    f = 64

    # Conv block 1: receptive field = 16
    if 1 in blocks:
        block1 = Conv1D(f, 2, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        block1 = BatchNormalization()(block1)
        block1 = Conv1D(f, 2, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block1)
        block1 = BatchNormalization()(block1)
        block1 = Conv1D(f, 2, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block1)
        block1 = BatchNormalization()(block1)
        block1 = Conv1D(f, 2, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block1)
        block1 = BatchNormalization()(block1)
        x1_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block1 = Add()([x1_bypass, block1])
    else:
        block1 = None

    # Conv block 2: receptive field = 31
    if 2 in blocks:
        block2 = Conv1D(f, 3, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        block2 = BatchNormalization()(block2)
        block2 = Conv1D(f, 3, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block2)
        block2 = BatchNormalization()(block2)
        block2 = Conv1D(f, 3, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block2)
        block2 = BatchNormalization()(block2)
        block2 = Conv1D(f, 3, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block2)
        block2 = BatchNormalization()(block2)
        x2_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block2 = Add()([x2_bypass, block2])
    else:
        block2 = None

    # Conv block 3: receptive field = 46
    if 3 in blocks:
        block3 = Conv1D(f, 4, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        block3 = BatchNormalization()(block3)
        block3 = Conv1D(f, 4, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block3)
        block3 = BatchNormalization()(block3)
        block3 = Conv1D(f, 4, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block3)
        block3 = BatchNormalization()(block3)
        block3 = Conv1D(f, 4, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block3)
        block3 = BatchNormalization()(block3)
        x3_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block3 = Add()([x3_bypass, block3])
    else:
        block3 = None

    # Conv block 4: receptive field = 136
    if 4 in blocks:
        block4 = Conv1D(f, 10, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        block4 = BatchNormalization()(block4)
        block4 = Conv1D(f, 10, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block4)
        block4 = BatchNormalization()(block4)
        block4 = Conv1D(f, 10, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block4)
        block4 = BatchNormalization()(block4)
        block4 = Conv1D(f, 10, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block4)
        block4 = BatchNormalization()(block4)
        x4_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block4 = Add()([x4_bypass, block4])
    else:
        block4 = None

    # Conv block 5: receptive field = 286
    if 5 in blocks:
        block5 = Conv1D(f, 20, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        block5 = BatchNormalization()(block5)
        block5 = Conv1D(f, 20, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block5)
        block5 = BatchNormalization()(block5)
        block5 = Conv1D(f, 20, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block5)
        block5 = BatchNormalization()(block5)
        block5 = Conv1D(f, 20, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block5)
        block5 = BatchNormalization()(block5)
        x5_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block5 = Add()([x5_bypass, block5])
    else:
        block5 = None

    # Concatenate conv blocks
    con = concatenate([i for i in [block1, block2, block3, block4, block5] if i is not None])
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
        block1 = Conv1D(f, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        block1 = BatchNormalization()(block1)
        block1 = Conv1D(f, 3, dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')(
            block1)
        block1 = BatchNormalization()(block1)
        block1 = Conv1D(f, 3, dilation_rate=4, padding='same', activation='relu', kernel_initializer='he_normal')(
            block1)
        block1 = BatchNormalization()(block1)
        block1 = Conv1D(f, 3, dilation_rate=8, padding='same', activation='relu', kernel_initializer='he_normal')(
            block1)
        block1 = BatchNormalization()(block1)
        x1_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block1 = Add()([x1_bypass, block1])
    else:
        block1 = None

    # Conv block 2: receptive field = 61
    if 2 in blocks:
        block2 = Conv1D(f, 5, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        block2 = BatchNormalization()(block2)
        block2 = Conv1D(f, 5, dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')(
            block2)
        block2 = BatchNormalization()(block2)
        block2 = Conv1D(f, 5, dilation_rate=4, padding='same', activation='relu', kernel_initializer='he_normal')(
            block2)
        block2 = BatchNormalization()(block2)
        block2 = Conv1D(f, 5, dilation_rate=8, padding='same', activation='relu', kernel_initializer='he_normal')(
            block2)
        block2 = BatchNormalization()(block2)
        x2_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block2 = Add()([x2_bypass, block2])
    else:
        block2 = None

    # Conv block 3: receptive field = 91
    if 3 in blocks:
        block3 = Conv1D(f, 7, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        block3 = BatchNormalization()(block3)
        block3 = Conv1D(f, 7, dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')(
            block3)
        block3 = BatchNormalization()(block3)
        block3 = Conv1D(f, 7, dilation_rate=4, padding='same', activation='relu', kernel_initializer='he_normal')(
            block3)
        block3 = BatchNormalization()(block3)
        block3 = Conv1D(f, 7, dilation_rate=8, padding='same', activation='relu', kernel_initializer='he_normal')(
            block3)
        block3 = BatchNormalization()(block3)
        x3_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block3 = Add()([x3_bypass, block3])
    else:
        block3 = None

    # Conv block 4: receptive field = 211
    if 4 in blocks:
        block4 = Conv1D(f, 15, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        block4 = BatchNormalization()(block4)
        block4 = Conv1D(f, 15, dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')(
            block4)
        block4 = BatchNormalization()(block4)
        block4 = Conv1D(f, 15, dilation_rate=4, padding='same', activation='relu', kernel_initializer='he_normal')(
            block4)
        block4 = BatchNormalization()(block4)
        block4 = Conv1D(f, 15, dilation_rate=8, padding='same', activation='relu', kernel_initializer='he_normal')(
            block4)
        block4 = BatchNormalization()(block4)
        x4_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block4 = Add()([x4_bypass, block4])
    else:
        block4 = None

    # Concatenate conv blocks
    con = concatenate([i for i in [block1, block2, block3, block4] if i is not None])
    return Model(inputs=inputs, outputs=con)


"""
Task1_Exponent

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

    # Convolutions - run through twice, flipping x and y dimensions on second run
    conv = conv_blocks(dimensions=2, blocks=blocks)
    c1 = GlobalMaxPooling1D()(conv(inputs))
    c2 = GlobalMaxPooling1D()(conv(reverse(inputs, axis=[2])))
    c = tf.math.maximum(c1, c2)  # max pool outputs from the two passes

    # Dense layers
    dense = Dense(512, activation='relu')(c)
    dense = Dense(256, activation='relu')(dense)
    out = Dense(1, activation='linear')(dense)

    # Full model
    model = Model(inputs=inputs, outputs=out)
    return model


"""
Task2_Classification

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

    # Convolutions - run through twice, flipping x and y dimensions on second run
    conv = conv_blocks(dimensions=2, blocks=blocks)
    c1 = GlobalMaxPooling1D()(conv(inputs))
    c2 = GlobalMaxPooling1D()(conv(reverse(inputs, axis=[2])))
    c = tf.math.maximum(c1, c2)  # max pool outputs from the two passes

    # Dense layers
    dense = Dense(512, activation='relu')(c)
    dense = Dense(256, activation='relu')(dense)
    out = Dense(5, activation='softmax')(dense)

    # Full model
    model = Model(inputs=inputs, outputs=out)
    return model


"""
Task3_Segmentation

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

    # Convolutions - run through twice, flipping x and y dimensions on second run
    conv = conv_blocks_for_seg(dimensions=2, blocks=blocks, length=199)
    c1 = conv(inputs)
    c2 = conv(reverse(inputs, axis=[2]))
    c = tf.math.add(c1, c2)  # add outputs from the two passes

    # 1x1 filter
    con = Conv1D(512, 1, activation='relu')(c)
    con = Conv1D(256, 1, activation='relu')(con)
    x5 = Conv1D(1, 1)(con)
    out = Softmax()(Flatten()(x5))

    # Full model
    model = Model(inputs=inputs, outputs=out)
    return model
