#import logging, os
#logging.disable(logging.WARNING)
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
import os
import pandas as pd
from glob import glob
import sys
from tqdm import tqdm
import gc
from datetime import datetime
from rsa_funcs import fit_rsa,make_RDM,get_triu
import pickle
from matplotlib import pyplot as plt
now = datetime.now

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

# save_dir = '../tf_weights/test001'
# iter = 0
# n_epochs = 30
# weights_init = 'None'

save_dir = str(sys.argv[1])
iter = int(sys.argv[2])
n_epochs = int(sys.argv[3])
weights_init = str(sys.argv[4])




if iter==0:
    print(f'save_dir | {save_dir}')

def safe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

safe_mkdir(save_dir)

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################


def get_mse(arr_gt,arr_recon):
    #arr_gt = data_comorb-data_td
    #arr_recon = (recon_onlyComorb-twins)
    mse_mean = ((arr_gt-arr_gt.mean(axis=0))**2).mean()
    mse_model = ((arr_gt-arr_recon)**2).mean()
    varexp = 1-mse_model/mse_mean
    return max(0,varexp)

def plot_brain(mat,vmin=None,vmax=None,cmap='bwr'):
    #mat = mat-mat[0,0,0]
    #mat = (mat-mat.min()) / (mat.max()-mat.min())
    #mat = (mat-mat.mean()) / mat.std()
    #vmin = mat.min()*.25
    #vmax = mat.max()*.75
    mat = mat.astype(np.float32)
    if not vmin:
        vmin = 0-abs(mat.max())
    if not vmax:
        vmax = 0+abs(mat.max())
    #print((mat.min(),mat.max()))
    #print(mat[0,0,0])
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1);plt.imshow(np.rot90(mat[:,:,32]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([]);plt.colorbar(shrink=.48)
    plt.subplot(1,3,2);plt.imshow(np.rot90(mat[:,32,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([]);plt.colorbar(shrink=.48)
    plt.subplot(1,3,3);plt.imshow(np.rot90(mat[32,:,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([]);plt.colorbar(shrink=.48)


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
      args (tensor): mean and log of variance of Q(z|X)
    # Returns:
      z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def get_MRI_CVAE_3D(input_shape=(64,64,64,1),
                    latent_dim=2,
                    beta=1,
                    disentangle=False,
                    gamma=1,
                    bias=True,
                    batch_size = 64,
                    kernel_size = 3,
                    filters = 32,
                    intermediate_dim = 128,
                    opt=None):

    image_size, _, _, channels = input_shape
    #epochs = 10
    nlayers = 2

    # build encoder model
    #tg_inputs = Input(shape=input_shape, name='tg_inputs')
    tg_inputs1 = Input(shape=input_shape, name='tg_inputs1')
    tg_inputs2 = Input(shape=input_shape, name='tg_inputs2')
    tg_inputs1_2 = Input(shape=input_shape, name='tg_inputs1_2')
    bg_inputs = Input(shape=input_shape, name='bg_inputs')

    ## Encoder shared by all participants
    z_conv1 = Conv3D(filters=filters*2,
            kernel_size=kernel_size,
            activation='relu',
            strides=2,
            use_bias=bias,
            padding='same')

    z_conv2 = Conv3D(filters=filters*4,
            kernel_size=kernel_size,
            activation='relu',
            strides=2,
            use_bias=bias,
            padding='same')


    # generate latent vector Q(z|X)
    z_h_layer = Dense(intermediate_dim, activation='relu', use_bias=bias)
    z_mean_layer = Dense(latent_dim, name='z_mean', use_bias=bias)
    z_log_var_layer = Dense(latent_dim, name='z_log_var', use_bias=bias)
    z_layer = Lambda(sampling, output_shape=(latent_dim,), name='z')

    def z_encoder_func(inputs):
        z_h = inputs
        z_h = z_conv1(z_h)
        z_h = z_conv2(z_h)
        # shape info needed to build decoder model
        shape = K.int_shape(z_h)
        z_h = Flatten()(z_h)
        z_h = z_h_layer(z_h)
        z_mean =  z_mean_layer(z_h)
        z_log_var =  z_log_var_layer(z_h)
        z = z_layer([z_mean, z_log_var])
        return z_mean, z_log_var, z, shape

    

    #### Encoder for features shared by ASD and ADHD                      

    s_conv1 = Conv3D(filters=filters*2,
            kernel_size=kernel_size,
            activation='relu',
            strides=2,
            use_bias=bias,
            padding='same')

    s_conv2 = Conv3D(filters=filters*4,
            kernel_size=kernel_size,
            activation='relu',
            strides=2,
            use_bias=bias,
            padding='same')

                        
    # generate latent vector Q(z|X)
    s_h_layer = Dense(intermediate_dim, activation='relu', use_bias=bias)
    s_mean_layer = Dense(latent_dim, name='s_mean', use_bias=bias)
    s_log_var_layer = Dense(latent_dim, name='s_log_var', use_bias=bias)
    s_layer = Lambda(sampling, output_shape=(latent_dim,), name='s')

    def s_encoder_func(inputs):
        s_h = inputs
        s_h = s_conv1(s_h)
        s_h = s_conv2(s_h)
        # shape info needed to build decoder model
        shape = K.int_shape(s_h)
        s_h = Flatten()(s_h)
        s_h = s_h_layer(s_h)
        s_mean =  s_mean_layer(s_h)
        s_log_var =  s_log_var_layer(s_h)
        s = s_layer([s_mean, s_log_var])
        return s_mean, s_log_var, s, shape



    #### Encoder for features unique to ASD

        
    s1_conv1 = Conv3D(filters=filters*2,
            kernel_size=kernel_size,
            activation='relu',
            strides=2,
            use_bias=bias,
            padding='same')

    s1_conv2 = Conv3D(filters=filters*4,
            kernel_size=kernel_size,
            activation='relu',
            strides=2,
            use_bias=bias,
            padding='same')

                        
    # generate latent vector Q(z|X)
    s1_h_layer = Dense(intermediate_dim, activation='relu', use_bias=bias)
    s1_mean_layer = Dense(latent_dim, name='s1_mean', use_bias=bias)
    s1_log_var_layer = Dense(latent_dim, name='s1_log_var', use_bias=bias)
    s1_layer = Lambda(sampling, output_shape=(latent_dim,), name='s1')

    def s1_encoder_func(inputs):
        s1_h = inputs
        s1_h = s1_conv1(s1_h)
        s1_h = s1_conv2(s1_h)
        # shape info needed to build decoder model
        shape = K.int_shape(s1_h)
        s1_h = Flatten()(s1_h)
        s1_h = s_h_layer(s1_h)
        s1_mean =  s_mean_layer(s1_h)
        s1_log_var =  s_log_var_layer(s1_h)
        s1 = s1_layer([s1_mean, s1_log_var])
        return s1_mean, s1_log_var, s1, shape

    #### Encoder for features unique to ADHD

    s2_conv1 = Conv3D(filters=filters*2,
            kernel_size=kernel_size,
            activation='relu',
            strides=2,
            use_bias=bias,
            padding='same')

    s2_conv2 = Conv3D(filters=filters*4,
            kernel_size=kernel_size,
            activation='relu',
            strides=2,
            use_bias=bias,
            padding='same')

                        
    # generate latent vector Q(z|X)
    s2_h_layer = Dense(intermediate_dim, activation='relu', use_bias=bias)
    s2_mean_layer = Dense(latent_dim, name='s2_mean', use_bias=bias)
    s2_log_var_layer = Dense(latent_dim, name='s2_log_var', use_bias=bias)
    s2_layer = Lambda(sampling, output_shape=(latent_dim,), name='s2')

    def s2_encoder_func(inputs):
        s2_h = inputs
        s2_h = s2_conv1(s2_h)
        s2_h = s2_conv2(s2_h)
        # shape info needed to build decoder model
        shape = K.int_shape(s2_h)
        s2_h = Flatten()(s2_h)
        s2_h = s_h_layer(s2_h)
        s2_mean =  s_mean_layer(s2_h)
        s2_log_var =  s_log_var_layer(s2_h)
        s2 = s2_layer([s2_mean, s2_log_var])
        return s2_mean, s2_log_var, s2, shape


    #### Combining and decoding     

    
    tg_s1s_mean, tg_s1s_log_var, tg_s1s, shape_s1s = s_encoder_func(tg_inputs1) # ASD patient-shared features
    tg_s1_mean, tg_s1_log_var, tg_s1, shape_s1 = s1_encoder_func(tg_inputs1) # ASD specific features
    tg_z1_mean, tg_z1_log_var, tg_z1, shape_tg_z1 = z_encoder_func(tg_inputs1) # ASD shared features 
    
                        
    tg_s2s_mean, tg_s2s_log_var, tg_s2s, shape_s2s = s_encoder_func(tg_inputs2)
    tg_s2_mean, tg_s2_log_var, tg_s2, shape_s2 = s2_encoder_func(tg_inputs2) #
    tg_z2_mean, tg_z2_log_var, tg_z2, shape_tg_z2 = z_encoder_func(tg_inputs2) #
 
    tg_s12s_mean, tg_s12s_log_var, tg_s12s, shape_s12s = s_encoder_func(tg_inputs1_2)
    tg_s12s1_mean, tg_s12s1_log_var, tg_s12s1, shape_s12s1 = s1_encoder_func(tg_inputs1_2)
    tg_s12s2_mean, tg_s12s2_log_var, tg_s12s2, shape_s12s2 = s2_encoder_func(tg_inputs1_2)
    tg_z12_mean, tg_z12_log_var, tg_z12, shape_tg_z12 = z_encoder_func(tg_inputs1_2) #

                        
    bg_z_mean, bg_z_log_var, bg_z, shape_z = z_encoder_func(bg_inputs) # Aidas and Stefano team hax


      # instantiate encoder models
    #z_encoder = tf.keras.models.Model(tg_inputs, [tg_z_mean, tg_z_log_var, tg_z], name='z_encoder')
    #s_encoder = tf.keras.models.Model(tg_inputs, [tg_s_mean, tg_s_log_var, tg_s], name='s_encoder')

    bg_z_encoder = tf.keras.models.Model(bg_inputs, [bg_z_mean, bg_z_log_var, bg_z], name='bg_z_encoder')

    tg1_z_encoder = tf.keras.models.Model(tg_inputs1, [tg_z1_mean, tg_z1_log_var, tg_z1], name='tg1_z_encoder')
    tg2_z_encoder = tf.keras.models.Model(tg_inputs2, [tg_z2_mean, tg_z2_log_var, tg_z2], name='tg2_z_encoder')
    tg1_2_z_encoder = tf.keras.models.Model(tg_inputs1_2, [tg_z12_mean, tg_z12_log_var, tg_z12], name='tg1_2_z_encoder')

    tg1_s1_encoder = tf.keras.models.Model(tg_inputs1, [tg_s1_mean, tg_s1_log_var, tg_s1], name='tg1_s1_encoder')
    tg2_s2_encoder = tf.keras.models.Model(tg_inputs2, [tg_s2_mean, tg_s2_log_var, tg_s2], name='tg2_s2_encoder')

    tg1_s_encoder = tf.keras.models.Model(tg_inputs1, [tg_s1s_mean, tg_s1s_log_var, tg_s1s], name='tg1_s_encoder')
    tg2_s_encoder = tf.keras.models.Model(tg_inputs2, [tg_s2s_mean, tg_s2s_log_var, tg_s2s], name='tg2_s_encoder')

    tg1_2_s_encoder = tf.keras.models.Model(tg_inputs1_2, [tg_s12s_mean, tg_s12s_log_var, tg_s12s], name='tg1_2_s_encoder')
    tg1_2_s1_encoder = tf.keras.models.Model(tg_inputs1_2, [tg_s12s1_mean, tg_s12s1_log_var, tg_s12s1], name='tg1_2_s1_encoder')
    tg1_2_s2_encoder = tf.keras.models.Model(tg_inputs1_2, [tg_s12s2_mean, tg_s12s2_log_var, tg_s12s2], name='tg1_2_s2_encoder')


    encoders = {'bg_z_encoder' : bg_z_encoder,
                'tg1_z_encoder' : tg1_z_encoder,
                'tg2_z_encoder' : tg2_z_encoder,
                'tg1_2_z_encoder' : tg1_2_z_encoder,
                'tg1_s1_encoder' : tg1_s1_encoder,
                'tg2_s2_encoder' : tg2_s2_encoder,
                'tg1_s_encoder' : tg1_s_encoder,
                'tg2_s_encoder' : tg2_s_encoder,
                'tg1_2_s_encoder' : tg1_2_s_encoder,
                'tg1_2_s1_encoder' : tg1_2_s1_encoder,
                'tg1_2_s2_encoder' : tg1_2_s2_encoder}


      # build decoder model
    latent_inputs = Input(shape=(4*latent_dim,), name='z_sampling')

    x = Dense(intermediate_dim, activation='relu', use_bias=bias)(latent_inputs)
    x = Dense(shape_z[1] * shape_z[2] * shape_z[3] * shape_z[4], activation='relu', use_bias=bias)(x)
    x = Reshape((shape_z[1], shape_z[2], shape_z[3],shape_z[4]))(x)

    for i in range(nlayers):
        x = Conv3DTranspose(filters=filters,
                          kernel_size=kernel_size,
                          activation='relu',
                          strides=2,
                          use_bias=bias,
                          padding='same')(x)
        filters //= 2

    outputs = Conv3DTranspose(filters=1,
                            kernel_size=kernel_size,
                            activation='sigmoid',
                            padding='same',
                            use_bias=bias,
                            name='decoder_output')(x)

    # instantiate decoder model
    cvae_decoder = Model(latent_inputs, outputs, name='decoder')
      # decoder.summary()

    def zeros_like(x):
        return tf.zeros_like(x)

    zeros = tf.keras.layers.Lambda(zeros_like)(tg_z1)
    #tg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, tg_s], -1))
    # CONC: [SHARED,SPECIFIC-PATIENTS,SPECIFIC-ASD,SPECIFIC-ADHD]
    tg1_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z1, tg_s1s,tg_s1,zeros], -1))
    tg2_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z2, tg_s2s,zeros,tg_s2], -1))                        
    tg1_2_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z12, tg_s12s,tg_s12s1,tg_s12s2], -1))                        
    
    

    bg_outputs = cvae_decoder(tf.keras.layers.concatenate([bg_z, zeros,zeros,zeros], -1)) 

 #   fg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, zeros], -1))

    # instantiate VAE model
    cvae = tf.keras.models.Model(inputs=[bg_inputs,tg_inputs1,tg_inputs2,tg_inputs1_2], 
                              outputs=[bg_outputs, tg1_outputs,tg2_outputs,tg1_2_outputs], 
                              name='contrastive_vae')



    #### DISENTANGLE SHARED from the rest                    
    s_concat = tf.keras.layers.concatenate([tg_s1s,tg_s1,tg_s2s,tg_s2,tg_s12s,tg_s12s1,tg_s12s2], -1)
    z_concat = tf.keras.layers.concatenate([tg_z1,tg_z2,tg_z12,bg_z], -1)
    if disentangle:
        discriminator = Dense(1, activation='sigmoid')

        z1 = Lambda(lambda x: x[:int(batch_size/2),:])(z_concat) #(tg_z)
        z2 = Lambda(lambda x: x[int(batch_size/2):,:])(z_concat) #(tg_z)
        s1 = Lambda(lambda x: x[:int(batch_size/2),:])(s_concat) #(tg_s)
        s2 = Lambda(lambda x: x[int(batch_size/2):,:])(s_concat) #(tg_s)

        q_bar = tf.keras.layers.concatenate(
          [tf.keras.layers.concatenate([s1, z2], axis=1),
          tf.keras.layers.concatenate([s2, z1], axis=1)],
          axis=0)

        q = tf.keras.layers.concatenate(
          [tf.keras.layers.concatenate([s1, z1], axis=1),
          tf.keras.layers.concatenate([s2, z2], axis=1)],
          axis=0)

        q_bar_score = (discriminator(q_bar)+.1) *.85 # +.1 * .85 so that it's 0<x<1 # assuming joint s z distribution
        q_score = (discriminator(q)+.1) *.85 # assuming that they're indepoendent 
        
        tc_loss = K.log(q_score / (1 - q_score)) 
        discriminator_loss = - K.log(q_score) - K.log(1 - q_bar_score)
    else:
        tc_loss = 0
        discriminator_loss = 0

    #### DISENTANGLE ASD-specific from COMORBID
    feat_specific_ASD = tg_s1
    feat_comorbid_ASD = tg_s1s
    if disentangle:
        discriminator = Dense(1, activation='sigmoid')

        z1_ASD = Lambda(lambda x: x[:int(batch_size/2),:])(feat_specific_ASD) #(tg_z)
        z2_ASD = Lambda(lambda x: x[int(batch_size/2):,:])(feat_specific_ASD) #(tg_z)
        s1_ASD = Lambda(lambda x: x[:int(batch_size/2),:])(feat_comorbid_ASD) #(tg_s)
        s2_ASD = Lambda(lambda x: x[int(batch_size/2):,:])(feat_comorbid_ASD) #(tg_s)

        q_bar_ASD = tf.keras.layers.concatenate(
          [tf.keras.layers.concatenate([s1_ASD, z2_ASD], axis=1),
          tf.keras.layers.concatenate([s2_ASD, z1_ASD], axis=1)],
          axis=0)

        q_ASD = tf.keras.layers.concatenate(
          [tf.keras.layers.concatenate([s1_ASD, z1_ASD], axis=1),
          tf.keras.layers.concatenate([s2_ASD, z2_ASD], axis=1)],
          axis=0)

        q_bar_score_ASD = (discriminator(q_bar_ASD)+.1) *.85 # +.1 * .85 so that it's 0<x<1 # assuming joint s z distribution
        q_score_ASD = (discriminator(q_ASD)+.1) *.85 # assuming that they're indepoendent 
        
        tc_loss_ASD = K.log(q_score_ASD / (1 - q_score_ASD)) 
        discriminator_loss_ASD = - K.log(q_score_ASD) - K.log(1 - q_bar_score_ASD)
    else:
        tc_loss_ASD = 0
        discriminator_loss_ASD = 0



    #### DISENTANGLE ASD-specific from COMORBID
    feat_specific_ADHD = tg_s2
    feat_comorbid_ADHD = tg_s2s
    if disentangle:
        discriminator = Dense(1, activation='sigmoid')

        z1_ADHD = Lambda(lambda x: x[:int(batch_size/2),:])(feat_specific_ADHD) #(tg_z)
        z2_ADHD = Lambda(lambda x: x[int(batch_size/2):,:])(feat_specific_ADHD) #(tg_z)
        s1_ADHD = Lambda(lambda x: x[:int(batch_size/2),:])(feat_comorbid_ADHD) #(tg_s)
        s2_ADHD = Lambda(lambda x: x[int(batch_size/2):,:])(feat_comorbid_ADHD) #(tg_s)

        q_bar_ADHD = tf.keras.layers.concatenate(
          [tf.keras.layers.concatenate([s1_ADHD, z2_ADHD], axis=1),
          tf.keras.layers.concatenate([s2_ADHD, z1_ADHD], axis=1)],
          axis=0)

        q_ADHD = tf.keras.layers.concatenate(
          [tf.keras.layers.concatenate([s1_ADHD, z1_ADHD], axis=1),
          tf.keras.layers.concatenate([s2_ADHD, z2_ADHD], axis=1)],
          axis=0)

        q_bar_score_ADHD = (discriminator(q_bar_ADHD)+.1) *.85 # +.1 * .85 so that it's 0<x<1 # assuming joint s z distribution
        q_score_ADHD = (discriminator(q_ADHD)+.1) *.85 # assuming that they're indepoendent 
        
        tc_loss_ADHD = K.log(q_score_ADHD / (1 - q_score_ADHD)) 
        discriminator_loss_ADHD = - K.log(q_score_ADHD) - K.log(1 - q_bar_score_ADHD)
    else:
        tc_loss_ADHD = 0
        discriminator_loss_ADHD = 0

    
                        
    reconstruction_loss = tf.keras.losses.mse(K.flatten(tg_inputs1), K.flatten(tg1_outputs)) # ASD MSE
    reconstruction_loss += tf.keras.losses.mse(K.flatten(tg_inputs2), K.flatten(tg2_outputs)) # ADHD MSE
    reconstruction_loss += tf.keras.losses.mse(K.flatten(tg_inputs1_2), K.flatten(tg1_2_outputs)) # ASD+ADHD MSE
    reconstruction_loss += tf.keras.losses.mse(K.flatten(bg_inputs), K.flatten(bg_outputs)) # Control MSE
    reconstruction_loss *= input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]


    # KL ASD
    kl_loss = 1 + tg_s1s_log_var - tf.keras.backend.square(tg_s1s_mean) - tf.keras.backend.exp(tg_s1s_log_var)
    kl_loss += 1 + tg_s1_log_var - tf.keras.backend.square(tg_s1_mean) - tf.keras.backend.exp(tg_s1_log_var)
    kl_loss += 1 + tg_z1_log_var - tf.keras.backend.square(tg_z1_mean) - tf.keras.backend.exp(tg_z1_log_var)
    
    #KL ADHD
    kl_loss += 1 + tg_s2s_log_var - tf.keras.backend.square(tg_s2s_mean) - tf.keras.backend.exp(tg_s2s_log_var)
    kl_loss += 1 + tg_s2_log_var - tf.keras.backend.square(tg_s2_mean) - tf.keras.backend.exp(tg_s2_log_var)
    kl_loss += 1 + tg_z2_log_var - tf.keras.backend.square(tg_z2_mean) - tf.keras.backend.exp(tg_z2_log_var)
    
    #KL ASD+ADHD
    kl_loss += 1 + tg_s12s_log_var - tf.keras.backend.square(tg_s12s_mean) - tf.keras.backend.exp(tg_s12s_log_var)
    kl_loss += 1 + tg_s12s1_log_var - tf.keras.backend.square(tg_s12s1_mean) - tf.keras.backend.exp(tg_s12s1_log_var)
    kl_loss += 1 + tg_s12s2_log_var - tf.keras.backend.square(tg_s12s2_mean) - tf.keras.backend.exp(tg_s12s2_log_var)
    kl_loss += 1 + tg_z12_log_var - tf.keras.backend.square(tg_z12_mean) - tf.keras.backend.exp(tg_z12_log_var)
    
    #KL CONTROL
    kl_loss += 1 + bg_z_log_var - tf.keras.backend.square(bg_z_mean) - tf.keras.backend.exp(bg_z_log_var)
                        
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    
    cvae_loss = tf.keras.backend.mean(reconstruction_loss + beta*kl_loss + gamma*tc_loss + gamma*discriminator_loss + gamma*tc_loss_ASD + gamma*discriminator_loss_ASD + gamma*tc_loss_ADHD + gamma*discriminator_loss_ADHD) # if increasing TC loss, might be a good idea to also increase DC loss (discriminator_loss*gamma)
    cvae.add_loss(cvae_loss)
    
    if type(opt)==type(None):
        opt = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,name='Adam')
    
#     opt = tf.keras.optimizers.SGD(
#     learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')

    #opt = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.9, epsilon=1e-07, centered=False, name='RMSprop')
    
    #cvae.compile(optimizer='rmsprop',run_eagerly=True)
    cvae.compile(optimizer=opt,run_eagerly=True)
    

    #return cvae, cvae_fg, z_encoder, s_encoder, cvae_decoder
    #return cvae, z_encoder, s_encoder, cvae_decoder
    return cvae, encoders,cvae_decoder



#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

def divide_chunks(l, n):
    # Yield successive n-sized
    # chunks from l.
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]
        

def cvae_dashboard():
    
    from IPython import display
    import sys
    
    ## PROGRESS PLOTTING
    # if np.mod(epoch,2)==0:
    #     plt.close()
        

    plt.ioff()
    display.clear_output(wait=True);
    display.display(plt.gcf());
    
    #plt.ioff()
    
    idx = train_chunks[batch_idx]
    mat_in = [data_td[idx],data_asd[idx],data_adhd[idx],data_asd_adhd[idx]]
    mat_out = cvae.predict([data_td[idx],data_asd[idx],data_adhd[idx],data_asd_adhd[idx]])
    
    
    ## For brain and correlation plotting
    idx = 0
    tg1_2_z_features = encoders['tg1_2_z_encoder'].predict(data_asd_adhd)[idx]
    tg1_2_s_features = encoders['tg1_2_s_encoder'].predict(data_asd_adhd)[idx]
    tg1_2_s1_features = encoders['tg1_2_s1_encoder'].predict(data_asd_adhd)[idx]
    tg1_2_s2_features = encoders['tg1_2_s2_encoder'].predict(data_asd_adhd)[idx]
    
    tg1_z_features = encoders['tg1_z_encoder'].predict(data_asd)[idx]
    tg1_s_features = encoders['tg1_s_encoder'].predict(data_asd)[idx]
    tg1_s1_features = encoders['tg1_s1_encoder'].predict(data_asd)[idx]
    tg1_s2_features = encoders['tg2_s2_encoder'].predict(data_asd)[idx]
    
    tg2_z_features = encoders['tg2_z_encoder'].predict(data_adhd)[idx]
    tg2_s_features = encoders['tg2_s_encoder'].predict(data_adhd)[idx]
    tg2_s1_features = encoders['tg1_s1_encoder'].predict(data_adhd)[idx]
    tg2_s2_features = encoders['tg2_s2_encoder'].predict(data_adhd)[idx]
    
    zeroes_ = np.zeros(tg1_2_s2_features.shape)
    chunks = list(divide_chunks(np.arange(nsubs), batch_size))
    
    recon = np.concatenate([cvae_decoder.predict(np.hstack((tg1_2_z_features[idx],tg1_2_s_features[idx],tg1_2_s1_features[idx],tg1_2_s2_features[idx])))[:,:,:,:,0] for idx in chunks],axis=0)
    twins = np.concatenate([cvae_decoder.predict(np.hstack((tg1_2_z_features[idx],zeroes_[idx],zeroes_[idx],zeroes_[idx])))[:,:,:,:,0] for idx in chunks],axis=0)
    recon_onlyASD = np.concatenate([cvae_decoder.predict(np.hstack((tg1_2_z_features[idx],zeroes_[idx],tg1_2_s1_features[idx],zeroes_[idx])))[:,:,:,:,0] for idx in chunks],axis=0)
    recon_onlyADHD = np.concatenate([cvae_decoder.predict(np.hstack((tg1_2_z_features[idx],zeroes_[idx],zeroes_[idx],tg1_2_s2_features[idx])))[:,:,:,:,0] for idx in chunks],axis=0)
    recon_onlyComorb = np.concatenate([cvae_decoder.predict(np.hstack((tg1_2_z_features[idx],tg1_2_s_features[idx],zeroes_[idx],zeroes_[idx])))[:,:,:,:,0] for idx in chunks],axis=0)

    mse_comorb.append(  get_mse(data_comorb,recon_onlyComorb)  )
    mse_recon.append(  get_mse(data_asd_adhd,recon)  )
    mse_asd.append(  get_mse(data_asd,recon_onlyASD)  )
    mse_adhd.append(  get_mse(data_adhd,recon_onlyADHD)  )
    
    
    ncols = 4;nrows=7
    #if np.mod(epoch,5)==0:
    #    plt.close()
    plt.subplots(nrows,ncols,figsize=(20,20)); # MAKE THE FIGURE
    
    plt.subplot(nrows,int(ncols/2),1) # PLOT LOSS
    
    plot_loss = loss[int(len(loss)*.2)::]
    #plot_loss_val = val_loss[int(len(loss)*.2)::]
    
    xs = np.arange(len(plot_loss))+1
    m,b = np.polyfit(xs,plot_loss,deg=1)
    #m_val,b_val = np.polyfit(xs,plot_loss_val,deg=1)
    
    plt.plot(plot_loss)
    #plt.plot(plot_loss_val)
    plt.plot(xs, m*xs + b)
    plt.title(f'Epoch {epoch} batch {batch_idx}/{len(train_chunks)} | Loss {loss[-1]:.2f}, beta: {m:.4f}, dur: {str(now()-t00)}')
    
    ##### SUBPLOT 3 ##### 
    plt.subplot(nrows,ncols,3) # PLOT LOSS LAST 50
    hb = 500
    if len(loss)>hb:
        plot_loss = loss[-hb::]
        #plot_loss_val = val_loss[-hb::]
    
        xs = np.arange(len(plot_loss))
        m,b = np.polyfit(xs,plot_loss,deg=1)
        #m_val,b_val = np.polyfit(xs,plot_loss_val,deg=1)
        plt.plot(plot_loss)
        #plt.plot(plot_loss_val)
        plt.plot(xs, m*xs + b)
        #plt.title(hist)
        plt.title(f'Loss last {hb} it, beta {m:.4f}')



    plt.subplot(nrows,ncols,5)
    plt.plot(mse_recon);plt.title(f'MSE recon {mse_recon[-1]:.2f}')
    plt.subplot(nrows,ncols,6)
    plt.plot(mse_asd);plt.title(f'mse_asd {mse_asd[-1]:.2f}')
    plt.subplot(nrows,ncols,7)
    plt.plot(mse_adhd);plt.title(f'mse_adhd {mse_adhd[-1]:.2f}')
    plt.subplot(nrows,ncols,8)
    plt.plot(mse_comorb);plt.title(f'mse_comorb {mse_comorb[-1]:.2f}')

    plt.subplot(nrows,ncols,9)
    plt.imshow(mat_in[0][0,:,:,32]);plt.xticks([]);plt.yticks([]);plt.title('TD in')
    plt.subplot(nrows,ncols,10)
    plt.imshow(mat_out[0][0,:,:,32,0]);plt.xticks([]);plt.yticks([]);plt.title('TD out')
    plt.subplot(nrows,ncols,11)
    plt.imshow(mat_in[0][0,:,:,32]-mat_out[0][0,:,:,32,0],cmap='bwr');plt.xticks([]);plt.yticks([]);plt.title('TD diff')
    
    
    plt.subplot(nrows,ncols,13)
    plt.imshow(mat_in[3][0,:,:,32]);plt.xticks([]);plt.yticks([]);plt.title('ASD_ADHD in')
    plt.subplot(nrows,ncols,14)
    plt.imshow(mat_out[3][0,:,:,32,0]);plt.xticks([]);plt.yticks([]);plt.title('ASD_ADHD out')
    plt.subplot(nrows,ncols,15)
    plt.imshow(mat_in[3][0,:,:,32]-mat_out[3][0,:,:,32,0],cmap='bwr');plt.xticks([]);plt.yticks([]);plt.title('ASD_ADHD diff')
    
    plt.subplot(nrows,ncols,17) # Recon plot SAG
    cmap = 'bwr'
    vmin = -.4
    vmax = .4
    recon_minus_twins = (recon-twins).mean(axis=0)
    mat = recon_minus_twins
    plt.imshow(np.rot90(mat[32,:,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([]);plt.title('recon_minus_twins')

    plt.subplot(nrows,ncols,18) # ASD plot SAG
    recon_onlyASD_minus_twins = (recon_onlyASD-twins).mean(axis=0)
    mat = recon_onlyASD_minus_twins
    plt.imshow(np.rot90(mat[32,:,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([]);plt.title('recon_onlyASD_minus_twins')
    
    plt.subplot(nrows,ncols,19) # ADHD plot SAG
    recon_onlyADHD_minus_twins = (recon_onlyADHD-twins).mean(axis=0)
    mat = recon_onlyADHD_minus_twins
    plt.imshow(np.rot90(mat[32,:,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([]);plt.title('recon_onlyADHD_minus_twins')
    
    plt.subplot(nrows,ncols,20) # COMORB plot SAG
    recon_onlyComorb_minus_twins = (recon_onlyComorb-twins).mean(axis=0)
    mat = recon_onlyComorb_minus_twins
    plt.imshow(np.rot90(mat[32,:,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([]);plt.title('recon_onlyComorb_minus_twins')  
    
    plt.subplot(nrows,ncols,21) #Recon plot COR
    mat = recon_minus_twins
    plt.imshow(np.rot90(mat[:,32,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([]);plt.title('recon_minus_twins')  
    
    plt.subplot(nrows,ncols,22) #ASD plot COR 
    mat = recon_onlyASD_minus_twins
    plt.imshow(np.rot90(mat[:,32,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([]);plt.title('recon_onlyASD_minus_twins')  
    
    plt.subplot(nrows,ncols,23)  #ADHD plot COR
    mat = recon_onlyADHD_minus_twins
    plt.imshow(np.rot90(mat[:,32,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([]);plt.title('recon_onlyADHD_minus_twins')  
    
    plt.subplot(nrows,ncols,24) #COMORB plot COR   
    mat = recon_onlyComorb_minus_twins
    plt.imshow(np.rot90(mat[:,32,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([]);plt.title('recon_onlyComorb_minus_twins')  
    
    
    plt.subplot(nrows,ncols,12) # ASD corr plot
    
    v1 = data_comorb_mean.flatten()
    v2 = recon_onlyComorb_minus_twins.flatten()
    idx = v1!=0
    r_comorb_GT = np.corrcoef(v1[idx],v2[idx])[0,1]
    
    v1 = data_comorb_mean.flatten()
    v2 = recon_onlyASD_minus_twins.flatten()
    idx = v1!=0
    r_comorb_ASD =np.corrcoef(v1[idx],v2[idx])[0,1]
    r_comorb_minus_ASD.append(r_comorb_GT-r_comorb_ASD)     
    plt.plot(r_comorb_minus_ASD)
    plt.title(f'Comrb corr. with GT: {r_comorb_GT:.2f}, with ASD {r_comorb_ASD:.2f}')
    
    plt.subplot(nrows,ncols,16) # ADHD corr plot
    
    v1 = data_comorb_mean.flatten()
    v2 = recon_onlyComorb_minus_twins.flatten()
    idx = v1!=0
    r_comorb_GT = np.corrcoef(v1[idx],v2[idx])[0,1]
    
    
    v1 = data_comorb_mean.flatten()
    v2 = recon_onlyADHD_minus_twins.flatten()
    idx = v1!=0
    r_comorb_ADHD =np.corrcoef(v1[idx],v2[idx])[0,1]
    r_comorb_minus_ADHD.append(r_comorb_GT-r_comorb_ADHD)
    
    plt.plot(r_comorb_minus_ADHD)
    plt.title(f'Comrb corr. with GT: {r_comorb_GT:.2f}, with ADHD {r_comorb_ADHD:.2f}')
    


    idx = 2
    tg1_2_z_features = encoders['tg1_2_z_encoder'].predict(data_asd_adhd)[idx]
    tg1_2_s_features = encoders['tg1_2_s_encoder'].predict(data_asd_adhd)[idx]
    tg1_2_s1_features = encoders['tg1_2_s1_encoder'].predict(data_asd_adhd)[idx]
    tg1_2_s2_features = encoders['tg1_2_s2_encoder'].predict(data_asd_adhd)[idx]
    
    tg1_z_features = encoders['tg1_z_encoder'].predict(data_asd)[idx]
    tg1_s_features = encoders['tg1_s_encoder'].predict(data_asd)[idx]
    tg1_s1_features = encoders['tg1_s1_encoder'].predict(data_asd)[idx]
    tg1_s2_features = encoders['tg2_s2_encoder'].predict(data_asd)[idx]
    
    tg2_z_features = encoders['tg2_z_encoder'].predict(data_adhd)[idx]
    tg2_s_features = encoders['tg2_s_encoder'].predict(data_adhd)[idx]
    tg2_s1_features = encoders['tg1_s1_encoder'].predict(data_adhd)[idx]
    tg2_s2_features = encoders['tg2_s2_encoder'].predict(data_adhd)[idx]
    
    corr_z_s.append( fit_rsa(make_RDM(tg1_2_z_features),make_RDM(tg1_2_s_features)) )
    corr_s1_s.append( fit_rsa(make_RDM(tg1_2_s1_features),make_RDM(tg1_2_s_features)) )
    corr_s2_s.append( fit_rsa(make_RDM(tg1_2_s2_features),make_RDM(tg1_2_s_features)) )
    corr_s1_s2.append( fit_rsa(make_RDM(tg1_2_s2_features),make_RDM(tg1_2_s1_features)) )
    
    plt.subplot(nrows,ncols,25)
    plt.plot(corr_z_s);plt.title(f'corr_z_s {corr_z_s[-1]:.2f}')
    
    
    plt.subplot(nrows,ncols,26)
    plt.plot(corr_s1_s);plt.title(f'corr_s1_s {corr_s1_s[-1]:.2f}')
    
    
    plt.subplot(nrows,ncols,27)
    plt.plot(corr_s2_s);plt.title(f'corr_s2_s {corr_s2_s[-1]:.2f}')
    
    
    plt.subplot(nrows,ncols,28)
    plt.plot(corr_s1_s2);plt.title(f'corr_s1_s2 {corr_s1_s2[-1]:.2f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f'dashboard_{iter}.png'))
    

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################


data_td = np.load('../Data/synth-data-01-TD.npy')
data_asd = np.load('../Data/synth-data-01-ASD.npy')
data_adhd = np.load('../Data/synth-data-01-ADHD.npy')
data_asd_adhd = np.load('../Data/synth-data-01-ADHD_ASD.npy')
data_comorb = np.load('../Data/synth-data-01-COMORB.npy')


data_td_mean = data_td.mean(axis=0)
data_asd_mean = data_asd.mean(axis=0)
data_adhd_mean = data_adhd.mean(axis=0)
data_asd_adhd_mean = data_asd_adhd.mean(axis=0)
data_comorb_mean = data_comorb.mean(axis=0)


nsubs = data_td.shape[0]
batch_size= 20

r_comorb_minus_ASD = []
r_comorb_minus_ADHD = []

mse_comorb = []
mse_recon = []
mse_asd = []
mse_adhd = []

corr_z_s = []
corr_s1_s = []
corr_s2_s = []
corr_s1_s2 = []

cvae, encoders,cvae_decoder = get_MRI_CVAE_3D(input_shape=(64,64,64,1),
                latent_dim=4,
                beta = 1,
                disentangle=True,
                gamma= 100,
                bias=True,
                batch_size = batch_size,
                kernel_size = 3,
                filters = 32,
                intermediate_dim = 128,
                opt=None)
if weights_init!='None':
    print(f'initializing with {weights_init}')
    cvae.load_weights(weights_init)


do_train = True
if do_train==True:
    loss = []
    t00 = now()
    #n_epochs = 100
    for epoch in tqdm(range(n_epochs)):
        train_chunks = list(divide_chunks(np.random.permutation(np.arange(nsubs)), batch_size))
        nbatches = len(train_chunks)
        for batch_idx in range(nbatches):
            idx = train_chunks[batch_idx]
            hist = cvae.train_on_batch([data_td[idx],data_asd[idx],data_adhd[idx],data_asd_adhd[idx]])
            if np.isnan(hist):
                raise Exception('loss is NaN')
            np.isnan(hist)
            loss.append(hist)
        if np.mod(epoch,10)==9:
            cvae_dashboard()
    if epoch>0:
        cvae.save_weights(os.path.join(save_dir,f'CVAE_init_attmp_{iter}')) # SAVE WEIGHTS


np.save(os.path.join(save_dir,f'varexp_attmp_{iter}.npy'),np.array(mse_recon))

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################


if epoch>20:
    idx = 2
    tg1_2_z_features = encoders['tg1_2_z_encoder'].predict(data_asd_adhd)[idx]
    tg1_2_s_features = encoders['tg1_2_s_encoder'].predict(data_asd_adhd)[idx]
    tg1_2_s1_features = encoders['tg1_2_s1_encoder'].predict(data_asd_adhd)[idx]
    tg1_2_s2_features = encoders['tg1_2_s2_encoder'].predict(data_asd_adhd)[idx]
    
    tg1_z_features = encoders['tg1_z_encoder'].predict(data_asd)[idx]
    tg1_s_features = encoders['tg1_s_encoder'].predict(data_asd)[idx]
    tg1_s1_features = encoders['tg1_s1_encoder'].predict(data_asd)[idx]
    tg1_s2_features = encoders['tg2_s2_encoder'].predict(data_asd)[idx]
    
    tg2_z_features = encoders['tg2_z_encoder'].predict(data_adhd)[idx]
    tg2_s_features = encoders['tg2_s_encoder'].predict(data_adhd)[idx]
    tg2_s1_features = encoders['tg1_s1_encoder'].predict(data_adhd)[idx]
    tg2_s2_features = encoders['tg2_s2_encoder'].predict(data_adhd)[idx]
    
    corr_z_s.append( fit_rsa(make_RDM(tg1_2_z_features),make_RDM(tg1_2_s_features)) )
    corr_s1_s.append( fit_rsa(make_RDM(tg1_2_s1_features),make_RDM(tg1_2_s_features)) )
    corr_s2_s.append( fit_rsa(make_RDM(tg1_2_s2_features),make_RDM(tg1_2_s_features)) )
    corr_s1_s2.append( fit_rsa(make_RDM(tg1_2_s2_features),make_RDM(tg1_2_s1_features)) )

    zeroes_ = np.zeros(tg1_2_s2_features.shape)
    chunks = list(divide_chunks(np.arange(nsubs), batch_size))
    #idx = chunks[0]
    
    recon = np.concatenate([cvae_decoder.predict(np.hstack((tg1_2_z_features[idx],tg1_2_s_features[idx],tg1_2_s1_features[idx],tg1_2_s2_features[idx])))[:,:,:,:,0] for idx in chunks],axis=0)
    twins = np.concatenate([cvae_decoder.predict(np.hstack((tg1_2_z_features[idx],zeroes_[idx],zeroes_[idx],zeroes_[idx])))[:,:,:,:,0] for idx in chunks],axis=0)
    recon_onlyASD = np.concatenate([cvae_decoder.predict(np.hstack((tg1_2_z_features[idx],zeroes_[idx],tg1_2_s1_features[idx],zeroes_[idx])))[:,:,:,:,0] for idx in chunks],axis=0)
    recon_onlyADHD = np.concatenate([cvae_decoder.predict(np.hstack((tg1_2_z_features[idx],zeroes_[idx],zeroes_[idx],tg1_2_s2_features[idx])))[:,:,:,:,0] for idx in chunks],axis=0)
    recon_onlyComorb = np.concatenate([cvae_decoder.predict(np.hstack((tg1_2_z_features[idx],tg1_2_s_features[idx],zeroes_[idx],zeroes_[idx])))[:,:,:,:,0] for idx in chunks],axis=0)

    np.savez(
                        file=os.path.join(save_dir,f'latent_recons_{iter}.npz'),
                        tg1_2_z_features = tg1_2_z_features,
                        tg1_2_s_features = tg1_2_s_features,
                        tg1_2_s1_features = tg1_2_s1_features,
                        tg1_2_s2_features = tg1_2_s2_features,
                        tg1_z_features = tg1_z_features,
                        tg1_s_features = tg1_s_features,
                        tg1_s1_features = tg1_s1_features,
                        tg1_s2_features = tg1_s2_features,
                        tg2_z_features = tg2_z_features,
                        tg2_s_features = tg2_s_features,
                        tg2_s1_features = tg2_s1_features,
                        tg2_s2_features = tg2_s2_features,
                        recon = recon,
                        twins = twins,
                        recon_onlyASD = recon_onlyASD,
                        recon_onlyADHD = recon_onlyADHD,
                        recon_onlyComorb = recon_onlyComorb,
                        r_comorb_minus_ASD = np.array(r_comorb_minus_ASD),
                        r_comorb_minus_ADHD = np.array(r_comorb_minus_ADHD),
                        mse_comorb = np.array(mse_comorb),
                        mse_recon = np.array(mse_recon),
                        mse_asd = np.array(mse_asd),
                        mse_adhd = np.array(mse_adhd),
                        corr_z_s = np.array(corr_z_s),
                        corr_s1_s = np.array(corr_s1_s),
                        corr_s2_s = np.array(corr_s2_s),
                        corr_s1_s2 = np.array(corr_s1_s2)
                        )


if epoch>20:
    cmap = 'bwr'
    vmin = -.4
    vmax = .4
    
    plt.figure(figsize=(7,10))
    
    mat = (recon-twins).mean(axis=0)
    plt.subplot(4,3,1)
    plt.imshow(np.rot90(mat[:,:,32]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([])
    plt.ylabel('Reconstruction')
    
    plt.subplot(4,3,2)
    plt.imshow(np.rot90(mat[:,32,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([])
    
    plt.subplot(4,3,3)
    plt.imshow(np.rot90(mat[32,:,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([])
    
    
    mat = (recon_onlyASD-twins).mean(axis=0)
    plt.subplot(4,3,4)
    plt.imshow(np.rot90(mat[:,:,32]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([])
    plt.ylabel('ASD only')
    
    plt.subplot(4,3,5)
    plt.imshow(np.rot90(mat[:,32,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([])
    
    plt.subplot(4,3,6)
    plt.imshow(np.rot90(mat[32,:,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([])
    
    mat = (recon_onlyADHD-twins).mean(axis=0)
    plt.subplot(4,3,7)
    plt.imshow(np.rot90(mat[:,:,32]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([])
    plt.ylabel('ADHD only')
    
    plt.subplot(4,3,8)
    plt.imshow(np.rot90(mat[:,32,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([])
    
    plt.subplot(4,3,9)
    plt.imshow(np.rot90(mat[32,:,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([])
    
    mat = (recon_onlyComorb-twins).mean(axis=0)
    plt.subplot(4,3,10)
    plt.imshow(np.rot90(mat[:,:,32]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([])
    plt.ylabel('COMORB only')
    
    plt.subplot(4,3,11)
    plt.imshow(np.rot90(mat[:,32,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([])
    
    plt.subplot(4,3,12)
    plt.imshow(np.rot90(mat[32,:,:]),cmap=cmap,vmin=vmin,vmax=vmax);plt.xticks([]);plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f'neuroplots_{iter}.png'))
