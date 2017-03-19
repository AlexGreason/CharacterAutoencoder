import numpy as np
from keras import objectives
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Reshape, Flatten, UpSampling2D
from keras.models import Model

batch_size = 32
sidelen = 96
original_shape = (batch_size, 1, sidelen, sidelen)
latent_dim = 16
intermediate_dim = 256
nb_epoch = 0

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

weights = "omniglot_16_1.sav"
model = "omniglot_16_1.json"

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)

computer = "desktop"

if computer == "laptop":
    x_train = np.load("/home/exa/Documents/PythonData/images_all_processed.npy")

elif computer == "desktop":
    x_train = np.load("D:\\conlangstuff\\images_all_processed.npy")

x_train = x_train.reshape((x_train.shape[0], 1, sidelen, sidelen))


vae.load_weights("omniglot_16_1.sav")
if nb_epoch > 0:
    for epoch in range(nb_epoch):
        print("epoch", epoch)
        vae.fit(x_train, x_train,
                shuffle=True,
                verbose=1,
                nb_epoch=1,
                batch_size=batch_size,
                validation_split=.1)
        json_string = vae.to_json()
        open('1dimlatent-1.json', 'w').write(json_string)
        vae.save_weights("1dimlatent-1.sav", overwrite=True)