# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:04:24 2023

@author: westa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.io import savemat

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from matfileport import MatPort

batch_size = 100
original_dim = 45
latent_dim = 2  # Latent variable is set to 2 dimensions for easy plotting later
intermediate_dim = 32
target_dim = 54
epochs = 40

# Load the MNIST dataset
train_route = 'C:\\westame\\D1\\matlab\\P4_VAE\\mat\\231116\\'
# Create an object MatPort_NN associated with the training route for training the network
MatPort_NN = MatPort(train_route)
# Call the auto_call_fem method to read the FEM data sets used for training and testing from that path
# x: displacement field, used for input; y: maximum damage value, used for plotting; z: damage field, used for output
(x_train, y_train_, z_train, x_test, y_test_, z_test) = MatPort_NN.auto_call_2()
# x_scale = np.amax(x_train)
x_scale = 11
x_train = x_train.astype('float32') / x_scale
x_test = x_test.astype('float32') / x_scale
y_train_ = y_train_.reshape(-1)
y_test_ = y_test_.reshape(-1)
y_train = to_categorical(y_train_ - 1, target_dim)
y_test = to_categorical(y_test_ - 1, target_dim)

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

y = Input(shape=(target_dim,))  # Input category
yh = Dense(latent_dim)(y)  # Directly constructing the mean for each category

# Calculate the mean and variance of p(Z|X)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# Reparameterization layer, effectively adding noise to the input
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder layer, which is the generator part
decoder_h = Dense(intermediate_dim, activation='relu')
dense_hh = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
hh = dense_hh(h_decoded)
x_decoded_mean = decoder_mean(hh)

# Build the model
vae = Model([x, y], [x_decoded_mean, yh])

# xent_loss is the reconstruction loss, kl_loss is the KL loss
xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean - yh) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

# add_loss is a new method for more flexible addition of various losses
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

# Fit the model
vae.fit([x_train, y_train], 
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_test, y_test], None))

# Build encoder to observe the distribution of each number in the latent space
encoder = Model(x, z_mean)

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
# plt.xlabel('X-axis label', fontsize=12)  # Set xlabel fontsize
# plt.ylabel('Y-axis label', fontsize=12)  # Set ylabel fontsize
# plt.title('Scatter Plot', fontsize=14)   # Set title fontsize
font_s = 12 
plt.xticks(fontsize=10)  # Set xticks fontsize
plt.yticks(fontsize=font_s)  # Set yticks fontsize
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=font_s)  # Set the fontsize for color bar ticks
plt.show()

# (x_d00) = MatPort_NN.one_call("x_1")
# d00 = x_d00.astype('float32') / x_scale
# d00_encoded = encoder.predict(d00, batch_size=batch_size)
# plt.scatter(d00_encoded[:, 0], d00_encoded[:, 1], marker='x', color='red', s=100)

# Build the generator
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# Observe how the two dimensions of latent variables affect the output results
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Use the quantiles of a normal distribution to construct latent variable pairs
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
font_s = 12 
plt.xticks(fontsize=12)  # Set xticks fontsize
plt.yticks(fontsize=font_s)  # Set yticks fontsize
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=font_s)  # Set the fontsize for color bar ticks
plt.show()

# (x_d00) = MatPort_NN.one_call("x_1")
# d00 = x_d00.astype('float32') / x_scale
# d00_encoded = encoder.predict(d00, batch_size=batch_size)
# plt.scatter(d00_encoded[:, 0], d00_encoded[:, 1], marker='x', color='red', s=100)

# Build the generator
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_hh = dense_hh(_h_decoded)
_x_decoded_mean = decoder_mean(_hh)
generator = Model(decoder_input, _x_decoded_mean)
x_gen = generator.predict(d00_encoded)
x_gen = x_gen.reshape(-1) * x_scale
plt.plot(x_gen)
