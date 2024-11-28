from glob import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow
from joblib import Parallel, delayed

import os
import shutil
import platform

import tensorflow as tf
from tensorflow.keras import utils, regularizers, callbacks, backend
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding1D, BatchNormalization, Flatten, Reshape, Conv1D, MaxPooling1D, Dropout, Add, LSTM, Embedding
from tensorflow.keras.initializers import glorot_normal, glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix


from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Flatten, Dense, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import GlobalAveragePooling1D
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ReduceLROnPlateau

def network_new(input_shape, ncat, learning_rate=0.00021544346900318823, reg=0.0032, dropout=0.1, seed=1):
    """Define the CNN structure with improvements.

    Parameters
    ----------
    input_shape : int
        Shape of the input spectra.
    ncat : int
        Number of categories.
    learning_rate : float
        Learning rate.
    reg : float
        Regularization factor.
    dropout : float
        Dropout rate.
    seed : int
        Seed of initializer.

    Returns
    -------
    model : tensorflow.keras.Model
        A model instance of the network.
    """
    X_input = Input(input_shape, name='Input_Spec')
    initializer = HeNormal(seed=seed)

    def conv_block(X, filters, kernel_size=5, reg=0.0032):
        # Shortcut (residual connection) path
        X_shortcut = X

        # Main path
        X = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same',
                   kernel_regularizer=l1(reg),
                   kernel_initializer=initializer)(X)
        X = BatchNormalization(axis=2)(X)
        X = Activation('relu')(X)
        X = MaxPooling1D(pool_size=2)(X)

        # Adjust the shortcut path to match the dimensions of the main path
        X_shortcut = Conv1D(filters, kernel_size=1, padding='same')(X_shortcut)
        X_shortcut = MaxPooling1D(pool_size=2)(X_shortcut)  # Add pooling to the shortcut as well

        # Add the main path and shortcut
        X = Add()([X, X_shortcut])
        return X

    # Convolutional layers with residual connections
    X = conv_block(X_input, filters=16)
    X = conv_block(X, filters=32)
    X = conv_block(X, filters=64)
    X = conv_block(X, filters=128)

    # Flatten to fully connected dense layer.
    X = Flatten()(X)
    X = Dense(512, kernel_regularizer=l2(reg), activation='relu')(X)
    X = Dropout(rate=dropout)(X)
    X = Dense(256, kernel_regularizer=l2(reg), activation='relu')(X)
    X = Dropout(rate=dropout)(X)
    
    # Output layer with softmax activation.
    X = Dense(ncat, kernel_regularizer=l2(reg), activation='softmax', name='Output_Classes')(X)

    model = Model(inputs=X_input, outputs=X, name='Enhanced_SNnet')
    
    # Set up optimizer with learning rate scheduler
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

galaxy_flux_desi = np.load("DESI_spectra/galaxy_flux.npy")
snia_flux_desi = np.load("DESI_spectra/snia_flux.npy")
snib_flux_desi = np.load("DESI_spectra/snib_flux.npy")
snibc_flux_desi = np.load("DESI_spectra/snibc_flux.npy")
snic_flux_desi = np.load("DESI_spectra/snic_flux.npy")
sniin_flux_desi = np.load("DESI_spectra/sniin_flux.npy")
sniilp_flux_desi = np.load("DESI_spectra/sniilp_flux.npy")
sniip_flux_desi = np.load("DESI_spectra/sniip_flux.npy")
kn_flux = np.load("DESI_spectra/kn_flux.npy")

tde_flux_ztf = np.load("ztf_spectra/tde_flux.npy")
tde_flux_paper = np.load("tde_flux_1.npy")
snia_flux_ztf = np.load("ztf_spectra/snia_flux.npy")
snii_flux_ztf = np.load("ztf_spectra/snii_flux.npy")
snib_flux_ztf = np.load("ztf_spectra/snib_flux.npy")
snic_flux_ztf = np.load("ztf_spectra/snic_flux.npy")
galaxy_flux_ztf = np.load("ztf_spectra/gal_flux.npy")
agn_flux = np.load("ztf_spectra/agn_flux.npy")
nls_flux = np.load("ztf_spectra/nls_flux.npy")
qso_flux = np.load("ztf_spectra/qso_flux.npy")

snia_flux = np.vstack([snia_flux_desi, snia_flux_ztf])
snibc_flux = np.vstack([snib_flux_desi, snib_flux_ztf, snibc_flux_desi, snic_flux_ztf, snic_flux_desi])
snii_flux = np.vstack([sniin_flux_desi, snii_flux_ztf, sniin_flux_desi, sniilp_flux_desi, sniip_flux_desi])
galaxy_flux = np.vstack([galaxy_flux_desi, galaxy_flux_ztf, agn_flux, nls_flux, qso_flux])
tde_flux = np.vstack([tde_flux_ztf, tde_flux_paper])

minw, maxw, nbins = 3000., 8000., 150

ngalaxy, nbins  = galaxy_flux.shape
nsnia, nbins  = snia_flux.shape
nsnibc, nbins = snibc_flux.shape
nsnii, nbins = snii_flux.shape
ntde, nbins = tde_flux.shape
nkn, nbins = kn_flux.shape
ngalaxy, nsnia, nsnibc, nsnii, ntde, nkn, nbins

x = np.concatenate([galaxy_flux, 
                    snia_flux,
                    snibc_flux,
                    snii_flux,
                    tde_flux,
                    kn_flux
                   ]).reshape(-1, nbins, 1)

labels = ['Galaxy',
          'SN Ia',
          'SN Ib/c',
          'SN II',
          'TDE',
          'KN']
ntypes = len(labels)

# Convert y-label array to appropriate categorical array
from tensorflow.keras.utils import to_categorical

y = to_categorical(
        np.concatenate([np.full(ngalaxy, 0), 
                        np.full(nsnia, 1),
                        np.full(nsnibc, 2),
                        np.full(nsnii, 3),
                        np.full(ntde, 4),
                        np.full(nkn, 5)
                       ]))

n_sample = [ngalaxy, nsnia, nsnibc, nsnii, ntde, nkn]
weights = np.max(n_sample) / n_sample
class_weight = {}
for i in range(len(weights)):
    class_weight[i] = weights[i]

# Define the number of folds
num_folds = 5

batch = 30
epoch = 210

# Initialize KFold
kf = KFold(n_splits=num_folds, shuffle=True, random_state=2)

# Prepare for tracking results
fold_no = 4

# Iterate over each fold
for train_index, test_index in kf.split(x, y):
    print(f"Training fold {fold_no}...")
    
    dropout = 0.5
    model = network_new((nbins, 1), ncat=y.shape[1], dropout = dropout)

    # Split the data into train and test for this fold
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Define callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
    mc = ModelCheckpoint(
        f'kfold_model/fold_{fold_no}_b{batch}_e{epoch}_model.keras',
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )

    # Train the model
    hist = model.fit(
        x_train, y_train,
        batch_size=batch,
        epochs=epoch,
        validation_data=(x_test, y_test),
        shuffle=True,
        verbose=True,
        callbacks=[es, mc, reduce_lr],
        class_weight=class_weight
    )

    # Increment fold counter
    fold_no += 1