#!/usr/bin/env python3
import os
import sys
import numpy as np
import random
from keras import models
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Reshape, Dropout
from keras.layers import Input
from keras.optimizers import Adam, Adagrad, Adadelta, Adamax, SGD
from keras.callbacks import CSVLogger
from keras.layers.advanced_activations import LeakyReLU
import scipy
import h5py
from data import denormalize4gan


def build_enc( shape ) :
    return build_discriminator(shape, build_disc=False)

def build_discriminator( shape, build_disc=True ) :
    def conv2d( x, filters, shape=(4, 4), **kwargs ) :
        x = Conv2D( filters, shape, strides=(2, 2),
            padding='same',
            kernel_initializer='glorot_uniform',
            **kwargs )( x )
        x = BatchNormalization(momentum=0.3)( x )
        x = LeakyReLU(alpha=0.2)( x )
        return x

    face = Input( shape=shape )
    x = face

    x = Conv2D( 64, (4, 4), strides=(2, 2),
        padding='same',
        kernel_initializer='glorot_uniform' )( x )
    x = LeakyReLU(alpha=0.2)( x )
    x = conv2d( x, 128 )
    x = conv2d( x, 256 )
    x = conv2d( x, 512 )

    if build_disc:
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid',
            kernel_initializer='glorot_uniform')( x )
        return models.Model( inputs=face, outputs=x )
    else:
        x = Conv2D(1, (4, 4), activation='tanh')(x)
        return models.Model( inputs=face, outputs=x )



def build_gen( shape ) :
    def deconv2d( x, filters, shape=(4, 4) ) :
        x= Conv2DTranspose( filters, shape, padding='same',
            strides=(2, 2), kernel_initializer='glorot_uniform' )(x)
        x = BatchNormalization(momentum=0.3)( x )
        x = LeakyReLU(alpha=0.2)( x )
        return x

    noise = Input( shape=(1, 1, 10) )
    x = noise
    x= Conv2DTranspose( 512, (4, 4),
        kernel_initializer='glorot_uniform' )(x)
    x = BatchNormalization(momentum=0.3)( x )
    x = LeakyReLU(alpha=0.2)( x )
    x = deconv2d( x, 256 )
    x = deconv2d( x, 128 )
    x = deconv2d( x, 64 )

    x = Conv2D( 64, (3, 3), padding='same',
        kernel_initializer='glorot_uniform' )( x )
    x = BatchNormalization(momentum=0.3)( x )
    x = LeakyReLU(alpha=0.2)( x )

    x= Conv2DTranspose( 3, (4, 4), padding='same', activation='tanh',
        strides=(2, 2), kernel_initializer='glorot_uniform' )(x)

    return models.Model( inputs=noise, outputs=x )
