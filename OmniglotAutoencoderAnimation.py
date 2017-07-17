import time

import numpy as np
import pygame
import scipy.misc as misc
import scipy.ndimage as ndimage
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Reshape, Flatten, UpSampling2D
from keras.models import Model

from helperfunctions import genvec, Beziercurve

batch_size = 32
sidelen = 96
original_shape = (batch_size, 1, sidelen, sidelen)
latent_dim = 16
intermediate_dim = 256

x = Input(batch_shape=original_shape)
a = Conv2D(128, (5, 5), padding='same', activation='relu')(x)
b = MaxPooling2D(pool_size=(4, 4))(a)
c = Conv2D(128, (3,3), padding='same', activation='relu')(b)
d = Conv2D(16, (3,3), padding='same', activation='relu')(c)
d_reshaped = Flatten()(d)
h = Dense(intermediate_dim, activation='relu')(d_reshaped)
z_mean = Dense(latent_dim)(h)

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
i = Dense(8 * 24 * 24, activation='relu')
j = Reshape((8, 24, 24))
k = Conv2D(128, (3,3), padding='same', activation='relu')
l = UpSampling2D((4, 4))
m = Conv2D(128, (3,3), padding='same', activation='relu')
n = Conv2D(128, (3,3), padding='same', activation='relu')
decoder_mean = Conv2D(1, 3, 3, border_mode='same', activation='sigmoid')

h_decoded = decoder_h(z_mean)
i_decoded = i(h_decoded)
j_decoded = j(i_decoded)
k_decoded = k(j_decoded)
l_decoded = l(k_decoded)
m_decoded = m(l_decoded)
n_decoded = n(m_decoded)
x_decoded_mean = decoder_mean(n_decoded)

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss='binary_crossentropy')

computer = "desktop"

if computer == "laptop":
    x_train = np.load("/home/exa/Documents/PythonData/images_all_processed.npy")

elif computer == "desktop":
    x_train = np.load("D:\\conlangstuff\\images_all_processed.npy")

x_train = x_train.reshape((x_train.shape[0], 1, sidelen, sidelen))
vae.load_weights("omniglot_16_1.sav")

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)
print(encoder.output_shape)

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

n_frame = 500
pygame.init()
x_dim, y_dim = 600, 600
screen = pygame.display.set_mode((x_dim, y_dim))
screen.fill((0, 0, 0))
numpoints = 2
np.random.seed(int(time.time()))
type = "random"

z = genvec(type = type)
zs = [genvec(type = type) for i in range(numpoints)]
iteration = 0
while True:
    pygame.event.get()
    starttime = time.time()
    if iteration % n_frame == 0:
        t = 0
        zs = [z] + [genvec(type = type) for i in range(numpoints - 1)]
        print("changing course")
    if t == 0:
        time.sleep(.5)
    iteration+= 1
    t += 1/n_frame
    print(t)
    z = Beziercurve(zs, t)
    frame = generator.predict(z.reshape((1,latent_dim)))
    print(frame.shape)
    #frame[frame < .25] = 0
    #frame[frame >= .75] = 1
    base = (frame * 255).reshape((sidelen, sidelen))
    base = ndimage.gaussian_filter(base, sigma=1)
    resized = misc.imresize(base, (x_dim, y_dim), interp='bicubic')
    resized[resized < 128] = 0
    resized[resized >= 128] = 255
    resized = resized[:, :, None].repeat(3, -1).astype("uint8")
    surface = pygame.surfarray.make_surface(resized)
    newscreen = pygame.transform.scale(surface, (x_dim, y_dim))
    screen.fill((0, 0, 0))
    screen.blit(newscreen, (0, 0))
    pygame.display.flip()
    endtime = time.time()
    print("processing image ", iteration, "frame time: ", endtime-starttime)

