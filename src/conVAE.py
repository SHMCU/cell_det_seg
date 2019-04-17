import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.io as sio
import os
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras import optimizers
#from keras.datasets import mnist
from keras import callbacks
import cv2
import visdom
from glob import glob
from sklearn.metrics import confusion_matrix


def createConvVAE():
    # input image dimensions
    img_rows, img_cols, img_chns = 96, 96, 3
    mask1_rows, mask1_cols, mask1_chns = 96, 96, 1
    # number of convolutional filters to use
    filters = 32
    # convolution kernel size
    num_conv = 4
    
    batch_size = 32
    if K.image_data_format() == 'channels_first':
        original_img_size = (img_chns, img_rows, img_cols)
        mask1_size = (mask1_chns, mask1_rows, mask1_cols)
        mask2_size = (mask1_chns, mask1_rows, mask1_cols)
        mask3_size = (mask1_chns, mask1_rows, mask1_cols)
    else:
        original_img_size = (img_rows, img_cols, img_chns)
        mask1_size = (mask1_rows, mask1_cols, mask1_chns)
        mask2_size = (mask1_rows, mask1_cols, mask1_chns)
        mask3_size = (mask1_rows, mask1_cols, mask1_chns)
        
        
    latent_dim = 32
    
    intermediate_dim = 256
    epsilon_std = 1.0
    epochs = 500
    
    x = Input(shape=original_img_size)
    mask_1 = Input(shape=mask1_size)
    mask_2 = Input(shape=mask2_size)
    mask_3 = Input(shape=mask3_size)
    conv_1 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', strides=(2, 2), activation='relu')(x)
    conv_2 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_1)
    conv_3 = Conv2D(filters*2,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_2)
    conv_4 = Conv2D(filters*2,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_3)
    
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)
    
    hidden1 = Dense(intermediate_dim, activation='relu')(flat)
    
    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)
    
    z_mean1 = Dense(latent_dim)(hidden1)
    z_log_var1 = Dense(latent_dim)(hidden1)
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    z1 = Lambda(sampling, output_shape=(latent_dim,))([z_mean1, z_log_var1])
    
    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(2 * filters * 6 * 6, activation='relu')
    
    decoder_hid1 = Dense(intermediate_dim, activation='relu')
    decoder_upsample1 = Dense(2 * filters * 6 * 6, activation='relu')
    
    if K.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters*2, 4, 4)
    else:
        output_shape = (batch_size, 6, 6, filters*2)
    
    decoder_reshape = Reshape(output_shape[1:])
    
    # decoder branch for mask1
    decoder_deconv_1 = Conv2DTranspose(filters*2,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=2,
                                       activation='relu')
    decoder_deconv_2 = Conv2DTranspose(filters*2,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=2,
                                       activation='relu')
    if K.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters, 29, 29)
    else:
        output_shape = (batch_size, 29, 29, filters)
    decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                              kernel_size=num_conv,
                                              strides=(2, 2),
                                              padding='same',
                                              activation='relu')
    decoder_mean_squash = Conv2DTranspose(mask1_chns,
                                          kernel_size=num_conv,
                                          padding='same',
                                          strides = (2, 2),
                                          activation='relu')
    
    hid_decoded = decoder_hid(z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)
    
    
    # decoder branch for mask2
    decoder_deconv_1_mask2 = Conv2DTranspose(filters*2,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=2,
                                       activation='relu')
    decoder_deconv_2_mask2 = Conv2DTranspose(filters*2,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=2,
                                       activation='relu')
    if K.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters, 29, 29)
    else:
        output_shape = (batch_size, 29, 29, filters)
    decoder_deconv_3_upsamp_mask2 = Conv2DTranspose(filters,
                                              kernel_size=num_conv,
                                              strides=(2, 2),
                                              padding='same',
                                              activation='relu')
    decoder_mean_squash_mask2 = Conv2DTranspose(mask1_chns,
                                          kernel_size=num_conv,
                                          padding='same',
                                          strides = (2, 2),
                                          activation='relu')
    
    deconv_1_decoded_mask2 = decoder_deconv_1_mask2(reshape_decoded)
    deconv_2_decoded_mask2 = decoder_deconv_2_mask2(deconv_1_decoded_mask2)
    x_decoded_relu_mask2 = decoder_deconv_3_upsamp_mask2(deconv_2_decoded_mask2)
    x_decoded_mean_squash_mask2 = decoder_mean_squash_mask2(x_decoded_relu_mask2)
    
    # decoder branch for mask3
    decoder_deconv_1_mask3 = Conv2DTranspose(filters*2,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=2,
                                       activation='relu')
    decoder_deconv_2_mask3 = Conv2DTranspose(filters*2,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=2,
                                       activation='relu')
    if K.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters, 29, 29)
    else:
        output_shape = (batch_size, 29, 29, filters)
    decoder_deconv_3_upsamp_mask3 = Conv2DTranspose(filters,
                                              kernel_size=5,
                                              strides=(2, 2),
                                              padding='same',
                                              activation='relu')
    decoder_mean_squash_mask3 = Conv2DTranspose(filters,
                                          kernel_size=3,
                                          padding='same',
                                          strides = (2, 2),
                                          activation='relu')
    
    decoder_mean_squash_mask4 = Conv2D(mask1_chns, kernel_size=3,
                                       padding='same',
                                       strides=(1,1),
                                       activation = 'relu')
    
    decoder_reshape1 = Reshape(output_shape[1:])
    
    hid_decoded1 = decoder_hid1(z1)
    up_decoded1 = decoder_upsample1(hid_decoded1)
    reshape_decoded1 = decoder_reshape(up_decoded1)
    
    deconv_1_decoded_mask3 = decoder_deconv_1_mask3(reshape_decoded1)
    
    deconv_1_decoded_mask3_ = concatenate([deconv_1_decoded_mask3, conv_3], axis = 3)
    deconv_2_decoded_mask3 = decoder_deconv_2_mask3(deconv_1_decoded_mask3_)
    
    deconv_2_decoded_mask3_ = concatenate([deconv_2_decoded_mask3, conv_2], axis = 3)
    x_decoded_relu_mask3 = decoder_deconv_3_upsamp_mask3(deconv_2_decoded_mask3_)
    
    x_decoded_relu_mask3_ = concatenate([x_decoded_relu, x_decoded_relu_mask2, x_decoded_relu_mask3, conv_1], axis = 3)
    
    x_decoded_mean_squash_mask3 = decoder_mean_squash_mask3(x_decoded_relu_mask3_)
    
    x_decoded_mean_squash_mask4 = decoder_mean_squash_mask4(x_decoded_mean_squash_mask3)
    
    
    vae = Model(x, [x_decoded_mean_squash, x_decoded_mean_squash_mask2, x_decoded_mean_squash_mask4])
    
    return vae

# Custom loss layer
'''class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash, mask_2, x_decoded_mean_squash_mask2, mask_3, x_decoded_mean_squash_mask3):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        
        mask_2 = K.flatten(mask_2)
        x_decoded_mean_squash_mask2 = K.flatten(x_decoded_mean_squash_mask2)
        
        mask_3 = K.flatten(mask_3)
        x_decoded_mean_squash_mask3 = K.flatten(x_decoded_mean_squash_mask3)
        
        #xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.mean_squared_error(x, x_decoded_mean_squash) + img_rows * img_cols * metrics.mean_squared_error(mask_2, x_decoded_mean_squash_mask2) + img_rows * img_cols * metrics.mean_squared_error(mask_3, x_decoded_mean_squash_mask3)
        
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss1 = - 0.5 * K.mean(1 + z_log_var1 - K.square(z_mean1) - K.exp(z_log_var1), axis=-1)
        return K.mean(xent_loss + kl_loss + kl_loss1)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        mask_2 = inputs[2]
        x_decoded_mean_squash_mask2 = inputs[3]
        mask_3 = inputs[4]
        x_decoded_mean_squash_mask3 = inputs[5]
        
        loss = self.vae_loss(x, x_decoded_mean_squash, mask_2, x_decoded_mean_squash_mask2, mask_3, x_decoded_mean_squash_mask3)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x

if train:
    y = CustomVariationalLayer()([mask_1, x_decoded_mean_squash, mask_2, x_decoded_mean_squash_mask2, mask_3, x_decoded_mean_squash_mask4])
    vae = Model([x, mask_1, mask_2, mask_3], y)
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    vae.compile(optimizer=opt, loss=None)
    vae.summary()
else:
    vae = Model(x, [x_decoded_mean_squash, x_decoded_mean_squash_mask2, x_decoded_mean_squash_mask4])'''