import numpy as np
import matplotlib.pyplot as plt
import time
import pygame
import pylab
import scipy.ndimage as ndimage
import scipy.misc as misc
import scipy.stats as stats

from keras.layers import Input, Dense, Lambda, Convolution2D, MaxPooling2D, Reshape, Flatten, UpSampling2D, AveragePooling2D
from keras.models import Model, model_from_json
from keras import backend as K
from keras import objectives

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

def Beziercurve(points, t):
    if len(points) == 2:
        return points[0] * (1 - t) + points[1] * t
    else:
        intpoints = [Beziercurve([points[i], points[i + 1]], t) for i in range(len(points) - 1)]
        return Beziercurve(intpoints, t)


lowerbound = -10
upperbound = 10

n_frame = 500
pygame.init()
x_dim, y_dim = 1200, 600
screen = pygame.display.set_mode((x_dim, y_dim))
screen.fill((0, 0, 0))
numpoints = 2
np.random.seed(int(time.time()))


def genvec(type="random"):
    if type == "random":
        return np.random.uniform(lowerbound, upperbound, latent_dim)
    else:
        x = np.random.randint(0, x_train.shape[0])
        z = encoder.predict(x_train[x].reshape((1, 1, sidelen, sidelen)))
        return z

def process(frame, x, y, threshold=128):
    base = (frame * 255).reshape((sidelen, sidelen))
    base = ndimage.gaussian_filter(base, sigma=1)
    resized = misc.imresize(base, (x, y), interp='bicubic')
    resized[resized < 128] = 0
    resized[resized >= 128] = 255
    #resized = resized[:, :, None].repeat(3, -1).astype("uint8")
    return resized

type = "animation"

if type == "pictures":
    while True:
        pygame.event.get()
        x =np.random.randint(0, x_train.shape[0])
        z = encoder.predict(x_train[x].reshape((1, 1, sidelen, sidelen)))
        original = x_train[x].reshape((sidelen, sidelen))
        baseoriginal = (original * 255).reshape((sidelen, sidelen))
        baseoriginal = baseoriginal[:, :, None].repeat(3, -1).astype("uint8")
        surfaceoriginal = pygame.surfarray.make_surface(baseoriginal)
        newscreenoriginal = pygame.transform.scale(surfaceoriginal, (x_dim//2, y_dim))
        print(z.shape)
        frame = generator.predict(z.reshape((1,latent_dim)))
        print(frame.shape)
        frame[frame < .5] = 0
        frame[frame >= .5] = 1
        base = (frame * 255).reshape((sidelen, sidelen))
        base = base[:, :, None].repeat(3, -1).astype("uint8")
        surface = pygame.surfarray.make_surface(base)
        newscreen = pygame.transform.scale(surface, (x_dim//2, y_dim))
        screen.fill((0, 0, 0))
        screen.blit(newscreenoriginal, (0, 0))
        screen.blit(newscreen, (600, 0))
        pygame.display.flip()
        endtime = time.time()
        time.sleep(1)

else: #  type == "animation"
    iteration= 0
    t = 0
    type = "existing"
    z = genvec(type=type)
    zs = [genvec(type=type) for i in range(numpoints)]
    while True:
        pygame.event.get()
        starttime = time.time()
        if iteration % n_frame == 0:
            t = 0
            x = np.random.randint(0, x_train.shape[0])
            original = x_train[x].reshape((sidelen, sidelen))
            original = process(original, 600, 600)
            originalsurface = pygame.surfarray.make_surface(original)
            screen.blit(originalsurface, (0, 0))
            newz = encoder.predict(x_train[x].reshape((1, 1, sidelen, sidelen)))
            zs = [z] + [newz]
            print("changing course")
        if t == 0:
            time.sleep(1)
        iteration += 1
        t += 1 / n_frame
        print(t)
        z = Beziercurve(zs, t)
        frame = generator.predict(z.reshape((1, latent_dim)))
        resized = process(frame, 600, 600, threshold=100)
        surface = pygame.surfarray.make_surface(resized)
        screen.fill((0, 0, 0))
        screen.blit(surface, (600, 0))
        screen.blit(originalsurface, (0, 0))
        pygame.display.flip()
        endtime = time.time()
        print("processing image ", iteration, "frame time: ", endtime - starttime)

