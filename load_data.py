from __future__ import print_function
import h5py
import numpy as np
import os

data_path = '/storage/local/data1/gpuscratch/ccianfar/' 
expt_name = None

def get_hdf5_dir(dir, qcd):
    '''
        'PF_vars' key dimensions: (event number, kinematic variable, jet object)
         - for jet objects, index 0 is the jet itself, and indices 1-10 are jet constituents
         - kinematic variables:
           - 0 - momentum
           - 1 - energy
           - 2 - charge
           - 3 - phi position
           - 4 - eta position
    '''
    X = None
    Y = None
    for filename in os.listdir(dir):
        print('Processing ' + dir + filename, end='\r')
        f = h5py.File(dir + filename, 'r')
        length = f['jets'].shape[0]
        length = min(length, 950)
        if length == 0:
            continue
        x = f['jets'][:length]
        y = np.ones(length)
        if qcd:
            y = np.zeros(length)
        if X is None and Y is None:
            X = np.array(x)
            Y = np.array(y)
        else:
            X = np.append(X, x, axis=0)
            Y = np.append(Y, y, axis=0)
    print()
    return X, Y

def load_data(num_datapoints, name):
    #top_dir = '/uscms_data/d3/ehinkle/elise/ttbar_outputNEW/'
    #qcd_dir = '/uscms_data/d3/ehinkle/elise/qcd_outputNEW/'
    expt_name = name
    if os.path.exists(data_path + expt_name + '.hdf5'):
        f = h5py.File(data_path + expt_name + '.hdf5', 'r')
        length = f['jets'].shape[0]
        jets = f['jets'][:length]
        labels = f['labels'][:length]
        return jets[:num_datapoints], labels[:num_datapoints]

    top_dir = data_path + expt_name + '/TOP/'
    qcd_dir = data_path + expt_name + '/QCD/'

    qcdX, qcdY = get_hdf5_dir(qcd_dir, True)
    topX, topY = get_hdf5_dir(top_dir, False) 

    # topX = np.swapaxes(topX, 1, 2)
    # qcdX = np.swapaxes(qcdX, 1, 2)

    print('Top shape: ' + str(topX.shape))
    print('QCD shape: ' + str(qcdX.shape))

    X = np.concatenate((topX, qcdX))
    Y = np.concatenate((topY, qcdY))

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    return X[:num_datapoints], Y[:num_datapoints]

if __name__ == '__main__':
    X, Y = load_data(2)
    print(X)
    print('X shape: ', str(X.shape))
    print(Y)
    print('Y shape: ', str(Y.shape))
