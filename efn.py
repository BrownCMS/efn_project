from load_data import load_data
from energyflow.archs import EFN
from keras import optimizers
import energyflow as ef
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

NUM_EPOCHS = 2

def split_data(Z, P, Y, test_prop=.1, val_prop=.1):
    length = X.shape[0]
    test_sz = int(length * test_prop)
    val_sz = int(length * val_prop)
    train_sz = length - test_sz - val_sz
    Z_train = Z[:train_sz]
    Z_val = Z[train_sz:train_sz+val_sz]
    Z_test = Z[train_sz+val_sz:]
    P_train = P[:train_sz]
    P_val = P[train_sz:train_sz+val_sz]
    P_test = P[train_sz+val_sz:]
    Y_train = Y[:train_sz]
    Y_val = Y[train_sz:train_sz+val_sz]
    Y_test = Y[train_sz+val_sz:]
    return Z_train, Z_val, Z_test, P_train, P_val, P_test, Y_train, Y_val, Y_test

def preprocess(X):
    X = X[:,:,:3]
    for x in X:
        mask = x[:,0] > 0
        weighted_avgs = np.average(x[mask,1:3], weights=x[mask,1], axis=0)

	x[:, 0] = x[:, 0] / np.sum(x[:, 0])
	x[mask, 1:3] = x[mask, 1:3] - weighted_avgs
    return X

if __name__ == '__main__':
    phi_sizes=(100,100,128)
    f_sizes=(100,100,100)

    X, Y = load_data(2000000, 'final_efn_train')
    X = preprocess(X)
    Y = ef.utils.to_categorical(Y) 
    
    (Z_train, Z_val, Z_test, P_train, 
        P_val, P_test, Y_train, Y_val, Y_test) = split_data(
            X[:,:,0], X[:,:,[1,2]], Y, test_prop=1.0/5, val_prop=1.0/5)

    #adam = optimizers.Adam(lr=.005)
    efn = EFN(input_dim=P_train.shape[-1], Phi_sizes=phi_sizes, F_sizes=f_sizes)
    efn.fit([Z_train, P_train], Y_train, epochs=NUM_EPOCHS, batch_size=500, 
        validation_data=([Z_val, P_val],Y_val), verbose=1)
    preds = efn.predict([Z_test, P_test], batch_size=1000)

    fpr, tpr, thresholds = roc_curve(Y_test[:,1], preds[:,1])
    print('AUC: ' + str(auc(fpr, tpr)))

    plt.plot(tpr, 1-fpr, '-', color='black', label='EFN')
    plt.show() 
