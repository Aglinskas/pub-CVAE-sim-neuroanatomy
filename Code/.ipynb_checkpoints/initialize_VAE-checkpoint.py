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


save_dir = str(sys.argv[1])
data_fn = str(sys.argv[2])
data_csv_fn = str(sys.argv[3])
iter = int(sys.argv[4])
n_epochs = int(sys.argv[5])
weights_init = str(sys.argv[6])


if iter==0:
    print(f'save_dir | {save_dir}')
    print(f'data_fn | {data_fn}')
    print(f'data_csv_fn | {data_csv_fn}')

def safe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

safe_mkdir(save_dir)

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

def get_MRI_VAE_3D(input_shape=(64,64,64,1),
                   latent_dim=2,
                   batch_size = 32,
                   disentangle=False,
                   gamma=1,
                   kernel_size = 3,
                   filters = 16,
                   intermediate_dim = 128,
                   opt=None):

    #TODO: add discriminator loss, see if there is improvement. Perhaps try on shapes dataset if it's easier...

    image_size, _, _, channels = input_shape
    
    #epochs = 10
    nlayers = 2
      
      # VAE model = encoder + decoder
      # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(nlayers):
        filters *= 2
        x = Conv3D(filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                strides=2,
                padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    x = Dense(shape[1] * shape[2] * shape[3] * shape[4], activation='relu')(x)
    x = Reshape((shape[1], shape[2], shape[3],shape[4]))(x)

    for i in range(nlayers):
        x = Conv3DTranspose(filters=filters,
                          kernel_size=kernel_size,
                          activation='relu',
                          strides=2,
                          padding='same')(x)
        filters //= 2

    outputs = Conv3DTranspose(filters=1,
                            kernel_size=kernel_size,
                            activation='sigmoid',
                            padding='same',
                            name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    #     decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    if disentangle:
        discriminator = Dense(1, activation='sigmoid')

        z1 = Lambda(lambda x: x[:int(batch_size/2),:int(latent_dim/2)])(z)
        z2 = Lambda(lambda x: x[int(batch_size/2):,:int(latent_dim/2)])(z)
        s1 = Lambda(lambda x: x[:int(batch_size/2),int(latent_dim/2):])(z)
        s2 = Lambda(lambda x: x[int(batch_size/2):,int(latent_dim/2):])(z)
        
        q_bar = tf.keras.layers.concatenate(
          [tf.keras.layers.concatenate([s1, z2], axis=1),
          tf.keras.layers.concatenate([s2, z1], axis=1)],
          axis=0)
        q = tf.keras.layers.concatenate(
          [tf.keras.layers.concatenate([s1, z1], axis=1),
          tf.keras.layers.concatenate([s2, z2], axis=1)],
          axis=0)
        
#         q_bar_score = discriminator(q_bar)
#         q_score = discriminator(q)        
#         tc_loss = K.log(q_score / (1 - q_score)) 

        q_bar_score = (discriminator(q_bar)+.1) *.85 # +.1 * .85 so that it's 0<x<1
        q_score = (discriminator(q)+.1) *.85 
        tc_loss = K.log(q_score / (1 - q_score)) 

        discriminator_loss = - K.log(q_score) - K.log(1 - q_bar_score)

    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= image_size * image_size


    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    if disentangle:
        vae_loss = K.mean(reconstruction_loss) + K.mean(kl_loss) + gamma * K.mean(tc_loss) + gamma * K.mean(discriminator_loss)
    else:
        vae_loss = K.mean(reconstruction_loss) + K.mean(kl_loss)

    vae.add_loss(vae_loss)
    
    if type(opt)==type(None):
        opt = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,name='Adam')
        
    #vae.compile(optimizer='rmsprop')
    vae.compile(optimizer=opt)
    

    if disentangle:
        vae.metrics_tensors = [reconstruction_loss, kl_loss, tc_loss, discriminator_loss]
        #     vae.summary()
    return encoder, decoder, vae


def divide_chunks(l, n):
    # Yield successive n-sized
    # chunks from l.
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]




if iter==0:
    print(np.__version__)
    print(tf.__version__)

if data_fn.endswith('.npy'):
    training_data = np.load(data_fn,allow_pickle=True)
elif data_fn.endswith('.npz'):
    training_data = np.load(data_fn)['data_arr']
    
training_data_mean = training_data.mean(axis=0)

if iter==0:
    print(training_data.shape)

df = pd.read_csv(data_csv_fn)
n = df.shape[0]
patient_idx = np.nonzero(df['dx'].values==1)[0]
control_idx = np.nonzero(df['dx'].values==0)[0]

rdm_tx_s = make_RDM(df.iloc[df['dx'].values==1]['adhd_tx'].values,data_scale='ratio', metric='euclidean')
rdm_tx_z = make_RDM(df.iloc[df['dx'].values==1]['td_tx'].values,data_scale='ratio', metric='euclidean')

batch_size = 10

encoder, decoder, vae = get_MRI_VAE_3D(input_shape=(64,64,64,1),
                   latent_dim=4,
                   batch_size = batch_size,
                   disentangle=False,
                   gamma=100,
                   kernel_size = 3,
                   filters = 64,
                   intermediate_dim = 128,
                   opt=None)

if weights_init!='None':
    print(f'initializing with {weights_init}')
    vae.load_weights(weights_init)


varExps = []
loss = []
t00 = now()
n_batches = int(n/batch_size)


for epoch in range(n_epochs):
    train_chunks = list(divide_chunks(np.random.permutation(np.arange(int(n))), batch_size))
    for batch_idx in range(n_batches):
            
        batch = training_data[train_chunks[batch_idx],:,:,:]
        hist = vae.train_on_batch(batch);
        loss.append(hist);
        
        mse = ((batch-vae.predict(batch)[:,:,:,:,0])**2).mean()
        mse_mean = ((batch-batch.mean(axis=0))**2).mean()
        varExp = 1-mse/mse_mean
        varExps.append(varExp)
        
        assert not np.isnan(hist),'loss is NaN - somethings wrong'

l_patients = encoder.predict(training_data[patient_idx,:,:,:])[0]
vae.save_weights(os.path.join(save_dir,f'VAE_init_attmp_{iter}')) # SAVE WEIGHTS
np.save(os.path.join(save_dir,f'varexp_attmp_{iter}'),np.array(varExps))
#np.savez_compressed(os.path.join(save_dir,f'latents_attmp_{iter}'),z_patients=z_patients[-1],s_patients=s_patients[-1])

#test_chunks = list(divide_chunks(np.arange(n), batch_size))
#recons = np.concatenate([vae.predict(training_data[test_chunk,:,:,:])[:,:,:,:,0] for test_chunk in test_chunks],axis=0)

np.savez_compressed(os.path.join(save_dir,f'training_log_{iter}'),epoch=epoch,
                                batch_idx=batch_idx,
                                duration=(now()-t00),
                                t00=t00,
                                loss=loss,
                                l_patients=l_patients,
                                varExps=varExps)







