from tensorflow import reverse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, Add, Input, concatenate, GlobalMaxPooling1D, \
    Average, Softmax, Flatten

"""
Regression

"""


def regression_model_complicated(dimensions, blocks, filters):
    inputs = Input((None, dimensions))
    f = filters

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

    # Dense layers
    dense = Dense(512, activation='relu')(con)
    dense = Dense(256, activation='relu')(dense)

    # Output
    dense2 = Dense(1, activation='linear')(dense)

    # Model
    model = Model(inputs=inputs, outputs=dense2)
    return model


def regression_model(dimensions, blocks):
    inputs = Input((None, dimensions))

    # Conv block 1: receptive field = 8
    if 1 in blocks:
        x1 = Conv1D(16, 2, padding='same', activation='relu')(inputs)
        x1 = Conv1D(32, 2, dilation_rate=2, padding='same', activation='relu')(x1)
        x1 = Conv1D(64, 2, dilation_rate=4, padding='same', activation='relu')(x1)
        x1 = GlobalMaxPooling1D()(x1)
    else:
        x1 = None

    # Conv block 2: receptive field = 15
    if 2 in blocks:
        x2 = Conv1D(16, 3, padding='same', activation='relu')(inputs)
        x2 = Conv1D(32, 3, dilation_rate=2, padding='same', activation='relu')(x2)
        x2 = Conv1D(64, 3, dilation_rate=4, padding='same', activation='relu')(x2)
        x2 = GlobalMaxPooling1D()(x2)
    else:
        x2 = None

    # Conv block 3: receptive field = 22
    if 3 in blocks:
        x3 = Conv1D(16, 4, padding='same', activation='relu')(inputs)
        x3 = Conv1D(32, 4, dilation_rate=2, padding='same', activation='relu')(x3)
        x3 = Conv1D(64, 4, dilation_rate=4, padding='same', activation='relu')(x3)
        x3 = GlobalMaxPooling1D()(x3)
    else:
        x3 = None

    # Conv block 4: receptive field = 145
    if 4 in blocks:
        x4 = Conv1D(16, 10, padding='same', activation='relu')(inputs)
        x4 = Conv1D(32, 10, dilation_rate=5, padding='same', activation='relu')(x4)
        x4 = Conv1D(64, 10, dilation_rate=10, padding='same', activation='relu')(x4)
        x4 = GlobalMaxPooling1D()(x4)
    else:
        x4 = None

    # Concatenate conv blocks
    con = concatenate([i for i in [x1, x2, x3, x4] if i is not None])

    # Dense layers
    dense = Dense(512, activation='relu')(con)
    dense = Dense(256, activation='relu')(dense)

    # Output
    dense2 = Dense(1, activation='linear')(dense)

    # Model
    model = Model(inputs=inputs, outputs=dense2)
    return model


def regression_model_1d(blocks=(1, 2, 3, 4)):
    inputs = Input((None, 1))
    model = regression_model(dimensions=1, blocks=blocks)
    out = model(inputs)
    model = Model(inputs=inputs, outputs=out)
    return model


def regression_model_2d(blocks=(1, 2, 3, 4)):
    inputs = Input((None, 2))
    model = regression_model(dimensions=2, blocks=blocks)
    out1 = model(inputs)
    out2 = model(reverse(inputs, axis=[2]))
    out = Average()([out1, out2])
    model = Model(inputs=inputs, outputs=out)
    return model


"""
Classification

"""


def classification_model_complicated(dimensions, blocks, filters):
    inputs = Input((None, dimensions))
    f = filters

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

    # Dense layers
    dense = Dense(512, activation='relu')(con)
    dense = Dense(256, activation='relu')(dense)

    # Output
    dense2 = Dense(5, activation='softmax')(dense)

    # Model
    model = Model(inputs=inputs, outputs=dense2)
    return model


def classification_model(dimensions, blocks):
    inputs = Input((None, dimensions))

    # Conv block 1: receptive field = 8
    if 1 in blocks:
        x1 = Conv1D(16, 2, padding='same', activation='relu')(inputs)
        x1 = Conv1D(32, 2, dilation_rate=2, padding='same', activation='relu')(x1)
        x1 = Conv1D(64, 2, dilation_rate=4, padding='same', activation='relu')(x1)
        x1 = GlobalMaxPooling1D()(x1)
    else:
        x1 = None

    # Conv block 2: receptive field = 15
    if 2 in blocks:
        x2 = Conv1D(16, 3, padding='same', activation='relu')(inputs)
        x2 = Conv1D(32, 3, dilation_rate=2, padding='same', activation='relu')(x2)
        x2 = Conv1D(64, 3, dilation_rate=4, padding='same', activation='relu')(x2)
        x2 = GlobalMaxPooling1D()(x2)
    else:
        x2 = None

    # Conv block 3: receptive field = 22
    if 3 in blocks:
        x3 = Conv1D(16, 4, padding='same', activation='relu')(inputs)
        x3 = Conv1D(32, 4, dilation_rate=2, padding='same', activation='relu')(x3)
        x3 = Conv1D(64, 4, dilation_rate=4, padding='same', activation='relu')(x3)
        x3 = GlobalMaxPooling1D()(x3)
    else:
        x3 = None

    # Conv block 4: receptive field = 145
    if 4 in blocks:
        x4 = Conv1D(16, 10, padding='same', activation='relu')(inputs)
        x4 = Conv1D(32, 10, dilation_rate=5, padding='same', activation='relu')(x4)
        x4 = Conv1D(64, 10, dilation_rate=10, padding='same', activation='relu')(x4)
        x4 = GlobalMaxPooling1D()(x4)
    else:
        x4 = None

    # Concatenate conv blocks
    con = concatenate([i for i in [x1, x2, x3, x4] if i is not None])

    # Dense layers
    dense = Dense(512, activation='relu')(con)
    dense = Dense(256, activation='relu')(dense)

    # Output
    dense2 = Dense(5, activation='softmax')(dense)

    # Model
    model = Model(inputs=inputs, outputs=dense2)
    return model


def classification_model_1d(blocks=(1, 2, 3, 4)):
    inputs = Input((None, 1))
    model = classification_model(dimensions=1, blocks=blocks)
    out = model(inputs)
    model = Model(inputs=inputs, outputs=out)
    return model


def classification_model_2d(blocks=(1, 2, 3, 4)):
    inputs = Input((None, 2))
    model = classification_model(dimensions=2, blocks=blocks)
    out1 = model(inputs)
    out2 = model(reverse(inputs, axis=[2]))
    out = Average()([out1, out2])
    model = Model(inputs=inputs, outputs=out)
    return model


"""
Segmentation

"""


def segmentation_model_complicated(dimensions, blocks, filters, length=200):
    inputs = Input((length, dimensions))
    f = filters

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


def segmentation_model(dimensions, blocks, length=199):
    inputs = Input((length, dimensions))

    # Conv block 1: receptive field = 8
    if 1 in blocks:
        x1 = Conv1D(16, 2, padding='same', activation='relu')(inputs)
        x1 = Conv1D(32, 2, dilation_rate=2, padding='same', activation='relu')(x1)
        x1 = Conv1D(64, 2, dilation_rate=4, padding='same', activation='relu')(x1)
    else:
        x1 = None

    # Conv block 2: receptive field = 15
    if 2 in blocks:
        x2 = Conv1D(16, 3, padding='same', activation='relu')(inputs)
        x2 = Conv1D(32, 3, dilation_rate=2, padding='same', activation='relu')(x2)
        x2 = Conv1D(64, 3, dilation_rate=4, padding='same', activation='relu')(x2)
    else:
        x2 = None

    # Conv block 3: receptive field = 22
    if 3 in blocks:
        x3 = Conv1D(16, 4, padding='same', activation='relu')(inputs)
        x3 = Conv1D(32, 4, dilation_rate=2, padding='same', activation='relu')(x3)
        x3 = Conv1D(64, 4, dilation_rate=4, padding='same', activation='relu')(x3)
    else:
        x3 = None

    # Conv block 4: receptive field = 145
    if 4 in blocks:
        x4 = Conv1D(16, 10, padding='same', activation='relu')(inputs)
        x4 = Conv1D(32, 10, dilation_rate=5, padding='same', activation='relu')(x4)
        x4 = Conv1D(64, 10, dilation_rate=10, padding='same', activation='relu')(x4)
    else:
        x4 = None

    # Concatenate conv blocks
    con = concatenate([i for i in [x1, x2, x3, x4] if i is not None])

    # 1x1 filter
    con = Conv1D(32, 1, activation='relu')(con)
    x5 = Conv1D(1, 1)(con)

    # Location
    out2 = Softmax()(Flatten()(x5))

    # Model
    model = Model(inputs=inputs, outputs=out2)
    return model


def segmentation_model_1d(blocks=(1, 2, 3, 4), length=199):
    inputs = Input((length, 1))
    model = segmentation_model(dimensions=1, blocks=blocks, length=length)
    out = model(inputs)
    model = Model(inputs=inputs, outputs=out)
    return model


def segmentation_model_2d(blocks=(1, 2, 3, 4), length=199):
    inputs = Input((None, 2))
    model = segmentation_model(dimensions=2, blocks=blocks, length=length)
    out1 = model(inputs)
    out2 = model(reverse(inputs, axis=[2]))
    out = Average()([out1, out2])
    model = Model(inputs=inputs, outputs=out)
    return model
