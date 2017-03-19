import time

import matplotlib.pyplot as plt
import numpy as np
from keras import objectives
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Reshape, Flatten, UpSampling2D
from keras.models import Model

from helperfunctions import genvec

batch_size = 32
sidelen = 96
original_shape = (batch_size, 1, sidelen, sidelen)
latent_dim = 16
intermediate_dim = 256

x = Input(batch_shape=original_shape)
a = Convolution2D(128, 5, 5, border_mode='same', activation='relu')(x)
b = MaxPooling2D(pool_size=(4, 4))(a)
c = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(b)
d = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(c)
d_reshaped = Flatten()(d)
h = Dense(intermediate_dim, activation='relu')(d_reshaped)
z_mean = Dense(latent_dim)(h)

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
i = Dense(8 * 24 * 24, activation='relu')
j = Reshape((8, 24, 24))
k = Convolution2D(128, 3, 3, border_mode='same', activation='relu')
l = UpSampling2D((4, 4))
m = Convolution2D(128, 3, 3, border_mode='same', activation='relu')
n = Convolution2D(128, 3, 3, border_mode='same', activation='relu')
decoder_mean = Convolution2D(1, 3, 3, border_mode='same', activation='sigmoid')

h_decoded = decoder_h(z_mean)
i_decoded = i(h_decoded)
j_decoded = j(i_decoded)
k_decoded = k(j_decoded)
l_decoded = l(k_decoded)
m_decoded = m(l_decoded)
n_decoded = n(m_decoded)
x_decoded_mean = decoder_mean(n_decoded)


def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    return xent_loss


vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)

computer = "desktop"

if computer == "laptop":
    x_train = np.load("/home/exa/Documents/PythonData/images_all_processed.npy")

elif computer == "desktop":
    x_train = np.load("D:\\conlangstuff\\images_all_processed.npy")

x_train = x_train.reshape((x_train.shape[0], 1, sidelen, sidelen))
vae.load_weights("omniglot_16_1.sav")

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_i_decoded = i(_h_decoded)
_j_decoded = j(_i_decoded)
_k_decoded = k(_j_decoded)
_l_decoded = l(_k_decoded)
_m_decoded = m(_l_decoded)
_n_decoded = n(_m_decoded)
_x_decoded_mean = decoder_mean(_n_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 20  # figure with upperboundxupperbound digits
digit_size = sidelen
figure = np.zeros((digit_size * n, digit_size * n))

def showgrid(n, lowerbound, upperbound, type):
    np.random.seed(int(time.time()))
    grid_x = np.linspace(lowerbound, upperbound, n)
    grid_y = np.linspace(lowerbound, upperbound, n)
    dim1 = np.random.uniform(-1,1,16)
    dim2 = np.random.uniform(-1,1,16)
    offset = np.random.uniform(-3,3,16)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            if type == "randaxes":
                z_sample = (dim1*yi+dim2*xi+offset).reshape((1,16))
            elif type == "random":
                z_sample = genvec().reshape((1,16))
            else:
                z_sample = genvec(type="existing", x_train=x_train, encoder=encoder).reshape((1,16))
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(25, 25))
    plt.imshow(figure, cmap='Greys')
    plt.show()

if __name__ == "__main__":
    showgrid(20, -5, 5, "existing")