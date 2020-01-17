from tensorflow.python.keras.layers import Input, Dense, Flatten, Reshape, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.python.keras.models import Model


def conv(filters):
    def blcok(x):
        x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        return x

    return blcok


def upscale(filters):
    def block(x):
        x = Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        return x

    return block


def Encoder():
    input_ = Input(shape=(64, 64, 3))
    x = input_

    x = conv(128)(x)
    x = conv(256)(x)
    x = conv(512)(x)
    x = conv(1024)(x)

    x = Dense(128)(Flatten()(x))

    x = Dense(4 * 4 * 1024)(x)
    x = Reshape((4, 4, 1024))(x)
    x = upscale(512)(x)
    return Model(input_, x)


def Decoder():
    input_ = Input(shape=(8, 8, 512))
    x = input_

    x = upscale(256)(x)
    x = upscale(128)(x)
    x = upscale(64)(x)

    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(input_, x)


encoder = Encoder()
