from load_data import load_data
from energyflow.archs import PFN
from keras import optimizers
import energyflow as ef
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc

NUM_EPOCHS = 50

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
    X = X[:,:,:3]
    for x in X:
        mask = x[:,0] > 0
        weighted_avgs = np.average(x[mask,1:3], weights=x[mask,0], axis=0)

	x[:, 0] = x[:, 0] / np.sum(x[:, 0])
	x[mask, 1:3] = x[mask, 1:3] - weighted_avgs
    return X

if __name__ == '__main__':
    phi_sizes=(100,100,128)
    f_sizes=(100,100,100)

    X, Y = load_data(200000, 'final_efn_train')
    X = preprocess(X)
    Y = ef.utils.to_categorical(Y) 
    
    p = np.random.permutation(len(X))
    X = X[p]
    Y = Y[p]

    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(
        X, Y, test_prop=1.0/5, val_prop=1.0/5)

    print(X_train[:3,:5])

    adam = optimizers.Adam(lr=.0006)
    pfn = PFN(input_dim=X_train.shape[-1], Phi_sizes=phi_sizes, F_sizes=f_sizes, optimizer=adam)
    pfn.fit(X_train, Y_train, epochs=NUM_EPOCHS, batch_size=250, 
        validation_data=(X_val,Y_val), verbose=1)
    preds = pfn.predict(X_test, batch_size=1000)

    fpr, tpr, thresholds = roc_curve(Y_test[:,1], preds[:,1])
    print('AUC: ' + str(auc(fpr, tpr)))
