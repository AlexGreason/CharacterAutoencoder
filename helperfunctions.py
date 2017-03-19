import numpy as np
import matplotlib.pyplot as plt
import time
import pygame
import pylab
import scipy.ndimage as ndimage
import scipy.misc as misc
import scipy.stats as stats

def Beziercurve(points, t):
    if len(points) == 2:
        return points[0] * (1 - t) + points[1] * t
    else:
        intpoints = [Beziercurve([points[i], points[i + 1]], t) for i in range(len(points) - 1)]
        return Beziercurve(intpoints, t)

def genvec(latent_dim = 16, lowerbound = -10, upperbound = 10, sidelen = 96, type="random", encoder=None, x_train=None):
    if type == "random":
        return np.random.uniform(lowerbound, upperbound, latent_dim)
    else:
        x = np.random.randint(0, x_train.shape[0])
        z = encoder.predict(x_train[x].reshape((1, 1, sidelen, sidelen)))
        return z

def process(frame, x, y, sidelen=96, threshold=128):
    base = (frame * 255).reshape((sidelen, sidelen))
    base = ndimage.gaussian_filter(base, sigma=1)
    resized = misc.imresize(base, (x, y), interp='bicubic')
    resized[resized < 128] = 0
    resized[resized >= 128] = 255
    #resized = resized[:, :, None].repeat(3, -1).astype("uint8")
    return resized