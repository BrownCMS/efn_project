from load_data import load_data
from energyflow.archs import PFN
from keras import optimizers
import energyflow as ef
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc

NUM_EPOCHS = 50

def particle_map(x):
    mapping = {22 : 0.0,
               211 : 0.1,
               -211 : 0.2,
               321 : 0.3,
               -321 : 0.4,
               130 : 0.5,
               2112 : 0.6,
               -2112 : 0.7,
               2212 : 0.8,
               -2212 : 0.9,
               11 : 1.0,
               -11 : 1.1,
               13 : 1.2,
               -13 : 1.3,
               1 : 1.4,
               2 : 1.5,
               0 : 0}
    return mapping[x]

def split_data(X, Y, test_prop=.1, val_prop=.1):
    length = X.shape[0]
    test_sz = int(length * test_prop)
    val_sz = int(length * val_prop)
    train_sz = length - test_sz - val_sz
    X_train = X[:train_sz]
    X_val = X[train_sz:train_sz+val_sz]
    X_test = X[train_sz+val_sz:]
    Y_train = Y[:train_sz]
    Y_val = Y[train_sz:train_sz+val_sz]
    Y_test = Y[train_sz+val_sz:]
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def preprocess(X):
    map_func = np.vectorize(particle_map)
    for x in X:
        mask = x[:,0] > 0
        weighted_avgs = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
	x[:, 0] = x[:, 0] / np.sum(x[:, 0])
	x[mask, 1:3] = x[mask, 1:3] - weighted_avgs
        x[mask, 3] = map_func(x[mask, 3])
    return X

if __name__ == '__main__':
    phi_sizes=(16,32,64,128)
    f_sizes=(128,64,32,16)

    X, Y = load_data(2000000, 'final_efn_train')
    X = preprocess(X)
    Y = ef.utils.to_categorical(Y) 
    
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(
        X, Y, test_prop=1.0/5, val_prop=1.0/5)

    adam = optimizers.Adam(lr=.0006)
    pfn = PFN(input_dim=X_train.shape[-1], Phi_sizes=phi_sizes, F_sizes=f_sizes, optimizer=adam)
    pfn.fit(X_train, Y_train, epochs=NUM_EPOCHS, batch_size=250, 
        validation_data=(X_val,Y_val), verbose=1)
    preds = pfn.predict(X_test, batch_size=1000)

    fpr, tpr, thresholds = roc_curve(Y_test[:,1], preds[:,1])
    print('AUC: ' + str(auc(fpr, tpr)))
