import time

import numpy as np
import pylab
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Reshape, Flatten, UpSampling2D
from keras.models import Model
from matplotlib.widgets import Slider, Button

from helperfunctions import process, genvec

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
decoder_mean = Conv2D(1, (3,3), padding='same', activation='sigmoid')

h_decoded = decoder_h(z_mean)
i_decoded = i(h_decoded)
j_decoded = j(i_decoded)
k_decoded = k(j_decoded)
l_decoded = l(k_decoded)
m_decoded = m(l_decoded)
n_decoded = n(m_decoded)
x_decoded_mean = decoder_mean(n_decoded)

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss="binary_crossentropy")

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

numpoints = 2
np.random.seed(int(time.time()))
type = "existing"

pylab.figure(num=1, figsize=(10,10))
ax = pylab.subplot(111)
pylab.subplots_adjust(left=0.3, bottom=0.4)
l = ax.imshow(np.array([[1]+[0]*599]*600, dtype='int64'), cmap="Greys", animated=True)
#pylab.axis([0, 1, -10, 10])
lower = -20
upper = 20
axcolor = 'lightgoldenrodyellow'
axes = []
sliders = []
for x in range(16):
    axes.append(pylab.axes([0.15, x*.025, 0.75, 0.01], axisbg=axcolor))
for x in range(16):
    sliders.append(Slider(axes[x], str(x), lower, upper, valinit=0))


def update(val):
    values = [sliders[x].val for x in range(16)]
    data = process(generator.predict((np.array(values)).reshape((1,16))).reshape(96,96), 600, 600)
    l.set_data(data)
    pylab.draw()
[sliders[x].on_changed(update) for x in range(16)]


resetax = pylab.axes([0, 0.025, 0.1, 0.04])
resetb = Button(resetax, 'Random', color=axcolor, hovercolor='0.975')
def reset(event):
    val = genvec(type=type, x_train=x_train, encoder=encoder)
    print(val)
    for x in range(16):
        print(val[0][x])
        sliders[x].set_val(val[0][x])
resetb.on_clicked(reset)

savax = pylab.axes([0, 0.2, 0.1, 0.04])
savb = Button(savax, 'Save', color=axcolor, hovercolor='0.975')
def save(event):
    values = []
    for x in range(16):
        try:
            values.append(float("{0:.3f}".format(sliders[x].val[0])))
        except:
            values.append(float("{0:.3f}".format(sliders[x].val)))
    print(values)
    print()
savb.on_clicked(save)

loadax = pylab.axes([0, 0.1, 0.1, 0.04])
loadb = Button(loadax, 'Load', color=axcolor, hovercolor='0.975')
def load(event):
    values = input("Enter character vector:")
    values = values[1:-1]
    values = values.replace(" ", "")
    values = values.split(",")
    values = [float(x) for x in values]
    for x in range(16):
        sliders[x].set_val(values[x])
loadb.on_clicked(load)

pylab.show()

# Various figures for omniglot_16_1
#arabic numeral 4: [-15.33, -100, -40.33, -16, 52.33, 43.67, -60.67, 98.67, -28.33, 48.33, 91, 45.33, 64.33, -14, 66.33, -100]
#also a 4: [7.667, -100.0, -39.0, -9.0, 15.667, 54.667, -53.333, 95.0, -64.333, 55.0, 80.333, 61.333, 84.333, -55.667, 75.333, -49.333]

#distance between latent vectors may not be a good similarity metric

# u and a sideways J above it [-5.433, -10.0, 7.933, 8.367, 3.767, -1.733, -3.867, -3.267, -1.8, -9.733, -3.7, -5.433, -10.0, 2.067, 5.033, -1.433]
# circle with lines coming off top and bottom [-5.433, 5.2, 7.933, 8.367, 3.767, -1.733, -3.867, -3.267, -1.8, -9.733, -3.7, -5.433, -10.0, 2.067, 5.033, -1.433]
# perpendicularity symbol with hat [-5.6, -8.167, 6.467, -10.0, -10.0, 0.3, 3.833, 2.133, 0.833, 1.567, 4.667, -7.233, -10.0, -10.0, -10.0, 3.2]
# bowl with line over it [3.567, 4.733, -4.133, -1.533, -10.0, -10.0, 10.0, -0.733, -10.0, -3.9, -0.4, -10.0, 4.733, -5.633, -2.733, 1.1]
# ladder on its side: [3.033, 7.433, 2.033, 7.367, 5.6, -4.6, 10.0, 10.0, -9.567, -4.033, 4.9, -4.367, 1.2, 10.0, 2.4, 10.0]
# plus sign, but the the left and bottom forming a loop and the right bent 90 degrees upward [-1.667, -10.6, -20.0, -13.867, 6.8, -6.533, -5.933, -20.0, -20.0, 20.0, 20.0, 15.533, -20.0, -20.0, -4.333, 3.0]
# T crossed with mirrored J, but with a really sharp angle on the mirrored J [-20.0, 4.933, 7.933, 2.8, 3.533, -20.0, 20.0, 15.733, -20.0, 20.0, 6.2, -3.333, 5.533, -5.267, -6.267, 0.733]
# tall vertical line with smaller vertical lines on both sides [6.4, 16.6, -4.267, -0.533, 8.667, 3.0, -1.067, -0.267, -0.4, -1.2, -6.467, 0.667, -11.467, -5.267, -4.4, -4.333]
# the left edge of nevada with a little curly bit coming off the shallow angle [1.733, 20.0, 10.133, 13.933, -11.0, -14.6, -13.667, 6.867, -4.2, -2.867, 8.933, 14.4, 3.533, 11.067, -16.6, -7.867]
# lowish aspect ratio rectangle with line pointing upward coming off left upper corner [18.8, 4.067, -2.6, 6.667, -15.667, 17.8, 4.667, -16.067, -2.867, 7.467, 15.267, 7.933, 5.6, -4.667, 3.2, 16.867]
# vertical line with line coming off to right and sharply bending down on top and short line coming off to right on bottom [1.0, 20.0, 20.0, -20.0, -11.867, 20.0, -3.733, -20.0, -8.0, -20.0, 20.0, -20.0, -1.333, -12.067, 2.4, -20.0]
# I [10.0, -5.4, -2.867, -6.133, -16.2, -10.067, 6.933, 1.067, -14.6, -1.267, -2.333, -16.933, 10.133, -20.0, -20.0, -1.133]
# weird thing (W with a Y as the middle?) [20.0, 8.933, -8.8, 15.067, -20.0, 5.133, -3.2, -5.8, -20.0, 8.8, 20.0, 14.8, 9.4, 20.0, -6.733, 13.933]
# perpendicularity symbol [7.533, -20.0, 9.333, 14.467, 4.467, -7.067, 0.067, -20.0, 4.733, -3.067, 7.933, -12.333, 2.333, -4.133, -20.0, 20.0]
# sorta a W but with left bent things on the left two and a short right one [-0.592, -4.437, -1.53, 3.726, 1.723, -5.548, -7.522, -13.294, -19.16, -2.086, 12.332, 8.667, -18.933, 11.0, -13.267, 20.0]
# a thing [14.027, 13.222, -12.16, -11.198, 9.491, 19.545, 9.137, 6.489, -15.164, 1.695, -6.774, 6.257, 0.994, -3.902, -20.0, -6.467]
# capital letter H [-0.467, 1.8, -2.133, -0.667, 2.2, 4.533, 2.467, 4.533, -6.4, -0.667, 1.0, -0.2, 0.6, -1.6, -1.267, -0.4]
