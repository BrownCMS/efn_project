from load_data import load_data
import h5py

# name = 'efn_train_short_100.hdf5'
# name = 'efn_train_full_100.hdf5'
# name = 'efn_full.hdf5'
# name = 'pfn_id_full.hdf5'
name = 'final_efn_train'

if __name__=='__main__':
	X, Y = load_data(2000000, name)
	f = h5py.File('/storage/local/data1/gpuscratch/ccianfar/' + name + '.hdf5', 'w')
	jets = f.create_dataset('jets', data=X)
	labels = f.create_dataset('labels', data=Y)
