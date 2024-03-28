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

### Arguments from command-line
save_dir = str(sys.argv[1])
data_fn = str(sys.argv[2])
data_csv_fn = str(sys.argv[3])
iter = int(sys.argv[4])
n_epochs = int(sys.argv[5])
weights_init = str(sys.argv[6])

if iter==0: # Print boiler-plate once
    print(f'save_dir | {save_dir}')
    print(f'data_fn | {data_fn}')
    print(f'data_csv_fn | {data_csv_fn}')

def safe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

safe_mkdir(save_dir)


# CVAE Functions
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
    nlayers = 2

    # build encoder model
    tg_inputs = Input(shape=input_shape, name='tg_inputs')
    bg_inputs = Input(shape=input_shape, name='bg_inputs')

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

    tg_z_mean, tg_z_log_var, tg_z, shape_z = z_encoder_func(tg_inputs)


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

    tg_s_mean, tg_s_log_var, tg_s, shape_s = s_encoder_func(tg_inputs)
    bg_z_mean, bg_z_log_var, bg_z, _ = z_encoder_func(bg_inputs) 


      # instantiate encoder models
    z_encoder = tf.keras.models.Model(tg_inputs, [tg_z_mean, tg_z_log_var, tg_z], name='z_encoder')
    s_encoder = tf.keras.models.Model(tg_inputs, [tg_s_mean, tg_s_log_var, tg_s], name='s_encoder')


      # build decoder model
    latent_inputs = Input(shape=(2*latent_dim,), name='z_sampling')

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

    def zeros_like(x):
        return tf.zeros_like(x)

    tg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, tg_s], -1))
    zeros = tf.keras.layers.Lambda(zeros_like)(tg_z)

    bg_outputs = cvae_decoder(tf.keras.layers.concatenate([bg_z, zeros], -1))

    # instantiate VAE model
    cvae = tf.keras.models.Model(inputs=[tg_inputs, bg_inputs], 
                              outputs=[tg_outputs, bg_outputs], 
                              name='contrastive_vae')

    if disentangle:
        discriminator = Dense(1, activation='sigmoid')

        z1 = Lambda(lambda x: x[:int(batch_size/2),:])(tg_z)
        z2 = Lambda(lambda x: x[int(batch_size/2):,:])(tg_z)
        s1 = Lambda(lambda x: x[:int(batch_size/2),:])(tg_s)
        s2 = Lambda(lambda x: x[int(batch_size/2):,:])(tg_s)

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


    reconstruction_loss = tf.keras.losses.mse(K.flatten(tg_inputs), K.flatten(tg_outputs)) 
    reconstruction_loss += tf.keras.losses.mse(K.flatten(bg_inputs), K.flatten(bg_outputs)) 
    reconstruction_loss *= input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]


    kl_loss = 1 + tg_z_log_var - tf.keras.backend.square(tg_z_mean) - tf.keras.backend.exp(tg_z_log_var)
    kl_loss += 1 + tg_s_log_var - tf.keras.backend.square(tg_s_mean) - tf.keras.backend.exp(tg_s_log_var)
    kl_loss += 1 + bg_z_log_var - tf.keras.backend.square(bg_z_mean) - tf.keras.backend.exp(bg_z_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    cvae_loss = tf.keras.backend.mean(reconstruction_loss + beta*kl_loss + gamma*tc_loss + gamma*discriminator_loss) # if increasing TC loss, might be a good idea to also increase DC loss (discriminator_loss*gamma)
    cvae.add_loss(cvae_loss)
    
    if type(opt)==type(None):
        opt = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,name='Adam')
    
    cvae.compile(optimizer=opt,run_eagerly=True)
    
    return cvae, z_encoder, s_encoder, cvae_decoder


def divide_chunks(l, n):
    # Yield successive n-sized
    # chunks from l.
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]
        

def cvae_dashboard():
    ## Dashboard function for progress plotting
    
    plt.ioff()

    
    ncols = 4;nrows=7

    plt.subplots(nrows,ncols,figsize=(20,20)); # Premake the figure

    ### Generate Necessary files
    patient_batch = DX_batch
    control_batch = TD_batch
    predictions = cvae_predict([patient_batch, control_batch])
    
    sigma = (np.e ** s_encoder.predict(patient_batch)[1]).mean()
    sigmas.append(sigma)

    mu = s_encoder.predict(patient_batch)[0]
    mus.append(np.mean([mu[:,0].std() for i in range(mu.shape[1])]))

    prediction = predictions[0]

    cmat_actual = np.corrcoef(np.vstack((patient_batch.reshape(patient_batch.shape[0],-1),control_batch.reshape(control_batch.shape[0],-1))))
    cmat_pred = np.corrcoef(np.vstack((predictions[0].reshape(predictions[0].shape[0],-1),predictions[1].reshape(predictions[1].shape[0],-1))))
    c_sim.append(np.corrcoef(get_triu(cmat_pred),get_triu(cmat_actual))[0,1])

    
    ### RSA Values
    
    z = z_encoder.predict(training_data[patient_idx,:,:,:])[2]
    s = s_encoder.predict(training_data[patient_idx,:,:,:])[2]


    rsa_vals.append( [fit_rsa(make_RDM(z),rdm_tx_z),
                        fit_rsa(make_RDM(z),rdm_tx_s),
                        fit_rsa(make_RDM(s),rdm_tx_z),
                        fit_rsa(make_RDM(s),rdm_tx_s),] )

    corr_z_s.append(fit_rsa(make_RDM(s),make_RDM(z)))

    ##### SUBPLOT 1 & 2 ##### 

    plt.subplot(nrows,int(ncols/2),1) # PLOT LOSS

    plot_loss = loss[int(len(loss)*.2)::]

    xs = np.arange(len(plot_loss))+1
    m,b = np.polyfit(xs,plot_loss,deg=1)

    plt.plot(plot_loss)
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

    ##### SUBPLOT 4 ##### 
    plt.subplot(nrows,ncols,4)
    plt.hist(prediction[0,:,:,0].flatten(),alpha=.5)
    plt.hist(patient_batch[0,:,:].flatten(),alpha=.5)
    plt.legend(['predicted','actual'])
    plt.title('in/out histograms')

    ##### SUBPLOT 5 ##### 
    plt.subplot(nrows,ncols,5) #RSA over time
    plt.plot(c_sim)
    plt.title(f'in/out RSA: {c_sim[-1].round(2)}')

    if len(c_sim)>5: # PLOT LS LINE
        xs = np.arange(len(c_sim))+1
        m,b = np.polyfit(xs,c_sim,deg=1)
        plt.plot(xs, m*xs + b)
        plt.title(f'in/out RSA: {c_sim[-1].round(2)}, b={m:.4f}')


    ##### SUBPLOT 6 ##### 
    plt.subplot(nrows,ncols,6)
    if len(c_sim)>hb:
        #plot_loss = loss[-hb::]
        xs = np.arange(len(c_sim[-hb::]))
        m,b = np.polyfit(xs,c_sim[-hb::],deg=1)
        plt.plot(c_sim[-hb::])
        plt.plot(xs, m*xs + b)
        #plt.title(hist)
        plt.title(f'in/outRSA last {hb} it, b={m:.4f}')

    # ##### SUBPLOT 7 ##### 
    plt.subplot(nrows,ncols,7)


    # ##### SUBPLOT 8 ##### 
    plt.subplot(nrows,ncols,8)
    plt.plot([max(0,val) for val in varExps])
    plt.title(f'VarExp: {varExps[-1]:.2f}')
    

    ##### SUBPLOT 9 ##### 
    plt.subplot(nrows,ncols,9)
    plt.plot(sigmas)
    plt.title(f'sigmas | {sigmas[-1]:.4f}')


    ##### SUBPLOT 10 ##### 
    plt.subplot(nrows,ncols,10)
    plt.plot(mus)
    plt.title(f'Mu variance {mus[-1]:.4f}')


    ##### SUBPLOT 11 ##### 
    plt.subplot(nrows,ncols,11)
    plt.imshow(cmat_actual);

    ##### SUBPLOT 12 ##### 
    plt.subplot(nrows,ncols,12)
    plt.imshow(cmat_pred)
    

    # #############################################
    # ###################Reconstructions###########
    # #############################################

    ##### SUBPLOT 13 #####     
    rand_sub = np.random.randint(low=0,high=500) # Random subject
    rand_sub = 0

    #### AXIAL SLICES  ####
    mid = 32
    plt.subplot(nrows,ncols,13)
    plt.imshow(patient_batch[rand_sub,:,:,mid])
    plt.xticks([]);plt.yticks([]);plt.title('actual')
    ##### SUBPLOT 14 #####     
    plt.subplot(nrows,ncols,14)
    plt.imshow(prediction[rand_sub,:,:,mid,0])
    plt.xticks([]);plt.yticks([]);plt.title('predicted')
    # ##### SUBPLOT 15 #####     
    plt.subplot(nrows,ncols,15)
    plt.imshow(abs(patient_batch[rand_sub,:,:,mid]-prediction[rand_sub,:,:,mid,0]))
    plt.xticks([]);plt.yticks([]);plt.title('difference')
    
    
    plt.subplot(nrows,ncols,16)
    plt.plot(np.array(rsa_vals)[:,2::])
    plt.legend(['TX Z','TX S'])
    plt.title('S')

    plt.subplot(nrows,ncols,20)
    plt.plot(np.array(rsa_vals)[:,0:2])
    plt.legend(['TX Z','TX S'])
    plt.title('Z')

    #### SAGITAL SLICES  ####
    plt.subplot(nrows,ncols,17)
    plt.imshow(np.rot90(patient_batch[rand_sub,mid,:,:]))
    plt.xticks([]);plt.yticks([]);plt.title('actual')
    ##### SUBPLOT 14 #####     
    plt.subplot(nrows,ncols,18)
    plt.imshow(np.rot90(prediction[rand_sub,mid,:,:,0]))
    plt.xticks([]);plt.yticks([]);plt.title('predicted')
    # ##### SUBPLOT 15 #####     
    plt.subplot(nrows,ncols,19)
    plt.imshow(np.rot90( abs(patient_batch[rand_sub,mid,:,:]-prediction[rand_sub,mid,:,:,0]) ))
    plt.xticks([]);plt.yticks([]);plt.title('difference')

    predictions = cvae_predict([patient_batch,control_batch])
    input_shape = training_data.shape[1::]
    reconstruction_loss = tf.keras.losses.mse(K.flatten(patient_batch), K.flatten(predictions[0])) 
    reconstruction_loss += tf.keras.losses.mse(K.flatten(control_batch), K.flatten(predictions[1])) 
    reconstruction_loss *= input_shape[0] * input_shape[1]


    tg_z_mean, tg_z_log_var, tg_z = z_encoder.predict(patient_batch)
    tg_s_mean, tg_s_log_var, tg_s = s_encoder.predict(patient_batch)

    bg_z_mean, bg_z_log_var, bg_z = z_encoder.predict(control_batch)

    kl_loss1 = 1 + tg_z_log_var - tf.keras.backend.square(tg_z_mean) - tf.keras.backend.exp(tg_z_log_var)
    kl_loss2 = 1 + tg_s_log_var - tf.keras.backend.square(tg_s_mean) - tf.keras.backend.exp(tg_s_log_var)
    kl_loss3 = 1 + bg_z_log_var - tf.keras.backend.square(bg_z_mean) - tf.keras.backend.exp(bg_z_log_var)

    kl_loss1 = tf.keras.backend.sum(kl_loss1, axis=-1)
    kl_loss2 = tf.keras.backend.sum(kl_loss2, axis=-1)
    kl_loss3 = tf.keras.backend.sum(kl_loss3, axis=-1)
    kl_loss = kl_loss1+kl_loss2+kl_loss3
    kl_loss *= -0.5


    discriminator = Dense(1, activation='sigmoid')
    z1 = Lambda(lambda x: x[:int(batch_size/2),:])(tg_z)
    z2 = Lambda(lambda x: x[int(batch_size/2):,:])(tg_z)
    s1 = Lambda(lambda x: x[:int(batch_size/2),:])(tg_s)
    s2 = Lambda(lambda x: x[int(batch_size/2):,:])(tg_s)

    q_bar = tf.keras.layers.concatenate(
      [tf.keras.layers.concatenate([s1, z2], axis=1),
      tf.keras.layers.concatenate([s2, z1], axis=1)],
      axis=0)

    q = tf.keras.layers.concatenate(
      [tf.keras.layers.concatenate([s1, z1], axis=1),
      tf.keras.layers.concatenate([s2, z2], axis=1)],
      axis=0)

    
    q_bar_score = (discriminator(q_bar)+.1) *.85 # +.1 * .85 so that it's 0<x<1
    q_score = (discriminator(q)+.1) *.85 
    
    tc_loss = K.log(q_score / (1 - q_score)) 
    discriminator_loss = - K.log(q_score) - K.log(1 - q_bar_score)
    discriminator_loss

    loss_mse.append(reconstruction_loss.numpy())
    loss_kl.append(kl_loss.numpy().mean())
    loss_dc.append(tc_loss.numpy().mean())
    loss_tc.append(discriminator_loss.numpy().mean())


    plt.subplot(nrows,ncols,21) # MSE 
    plt.plot(loss_mse[int(len(loss_mse)*.2)::])
    #plt.plot(val_mse[int(len(loss_mse)*.2)::])

    plt.title(f'MSE | {loss_mse[-1]:.4f}')

    plt.subplot(nrows,ncols,22) # KL loss
    plt.plot(loss_kl)
    plt.title(f'KL | {loss_kl[-1]:.4f}')    


    plt.subplot(nrows,ncols,23) # TC     
    plt.plot(loss_tc)
    plt.title(f'Total Correlation loss | {loss_tc[-1]:.4f}')    


    plt.subplot(nrows,ncols,24) # Disc         
    plt.plot(loss_dc)
    plt.title(f'discriminator_loss | {loss_dc[-1]:.4f}')    


    tg_s = s_encoder.predict(patient_batch)
    tg_z = z_encoder.predict(patient_batch)
    bg_z = z_encoder.predict(control_batch)

    plt.subplot(nrows,ncols,25)
    plt.hist(tg_s[2].flatten(),alpha=.5);
    plt.hist(tg_z[2].flatten(),alpha=.5);
    plt.hist(bg_z[2].flatten(),alpha=.5);
    plt.legend(['tg_s','tg_z','bg_z'])
    plt.title('Z')


    plt.subplot(nrows,ncols,26)
    plt.hist(tg_s[0].flatten(),alpha=.5);
    plt.hist(tg_z[0].flatten(),alpha=.5);
    plt.hist(bg_z[0].flatten(),alpha=.5);
    plt.legend(['tg_s','tg_z','bg_z'])
    plt.title('Mus')


    plt.subplot(nrows,ncols,27)
    plt.hist(tg_s[1].flatten(),alpha=.5);
    plt.hist(tg_z[1].flatten(),alpha=.5);
    plt.hist(bg_z[1].flatten(),alpha=.5);
    plt.legend(['tg_s','tg_z','bg_z'])
    plt.title('Sigmas')


    plt.subplot(nrows,ncols,28)
    plt.plot(corr_z_s)
    plt.title(f'Corr S/Z: {corr_z_s[-1]:3f}')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f'dashboard_{iter}.png'))
    
    
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

batch_size = 50

def cvae_predict(batch):
    
    z = z_encoder.predict(batch[0])[2]
    z_ = np.zeros(z.shape)
    recon1 = cvae_decoder(np.hstack((z,z_)))
    
    z = z_encoder.predict(batch[1])[2]
    s = s_encoder.predict(batch[1])[2]
    recon2 = cvae_decoder(np.hstack((z,s)))
    
    return [recon1.numpy(),recon2.numpy()]

cvae, z_encoder, s_encoder, cvae_decoder = get_MRI_CVAE_3D(input_shape=(64,64,64,1),
                latent_dim=2,
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

varExps = []
loss = []
rsa_vals = []
mus = []
sigmas = []
c_sim = []
loss_mse = []
loss_kl = []
loss_dc = []
loss_tc = []
corr_z_s = []
z_patients = []
s_patients = []
t00 = now()
train_chunks = list(divide_chunks(np.random.permutation(np.arange(int(n/2))), batch_size))
n_batches = len(train_chunks)

# Main Training loop
for epoch in range(n_epochs):
    train_chunks = list(divide_chunks(np.random.permutation(np.arange(int(n/2))), batch_size))
    for batch_idx in range(n_batches):
            
        DX_batch = training_data[patient_idx[train_chunks[batch_idx]],:,:,:]
        TD_batch = training_data[control_idx[train_chunks[batch_idx]],:,:,:]
        hist = cvae.train_on_batch([DX_batch,TD_batch]);
        loss.append(hist);
        mse = ((np.array([DX_batch,TD_batch])-np.array(cvae_predict([DX_batch,TD_batch]))[:,:,:,:,:,0])**2).mean()
        mse_mean = ((np.concatenate((DX_batch,TD_batch))-training_data_mean)**2).mean()
        varExp = 1-mse/mse_mean
        varExps.append(varExp)
        
        assert not np.isnan(hist),'loss is NaN - somethings wrong'

    z_patients.append(z_encoder.predict(training_data[patient_idx,:,:,:])[0])
    s_patients.append(s_encoder.predict(training_data[patient_idx,:,:,:])[0])

    if n_epochs>0:
        cvae_dashboard()

## Save generated files and model
cvae.save_weights(os.path.join(save_dir,f'CVAE_init_attmp_{iter}')) # SAVE WEIGHTS
np.save(os.path.join(save_dir,f'varexp_attmp_{iter}'),np.array(varExps))
np.savez_compressed(os.path.join(save_dir,f'latents_attmp_{iter}'),z_patients=z_patients[-1],s_patients=s_patients[-1])
np.savez_compressed(os.path.join(save_dir,f'training_log_{iter}'),epoch=epoch,
                                batch_idx=batch_idx,
                                duration=(now()-t00),
                                t00=t00,
                                mus=mus,
                                sigmas=sigmas,
                                c_sim=c_sim,
                                loss_mse=loss_mse,
                                loss_kl=loss_kl,
                                loss_dc=loss_dc,
                                loss_tc=loss_tc,
                                loss=loss,
                                varExps=varExps,
                                rsa_vals=rsa_vals,
                                corr_z_s=corr_z_s,
                                z_patients=z_patients,
                                s_patients=s_patients)

del hist
del DX_batch
del TD_batch
del cvae
del z_encoder
del s_encoder
del cvae_decoder
del training_data_mean
del varExps
del z_patients
del s_patients
