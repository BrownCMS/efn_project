# coding: utf-8
import uproot
import os
import h5py
import numpy as np

EVENTS_PER_FILE = 100000
MAX_JET_LEN = 321
storage_path = '/storage/local/data1/gpuscratch/ccianfar/final_efn_train/'

def write_hdf5(tree, name, events_per_file):
    tree_eta = tree['con_eta_PFCalc'].array()
    tree_energy = tree['con_energy_PFCalc'].array()
    tree_phi = tree['con_phi_PFCalc'].array()
    tree_id = tree['con_id_PFCalc'].array()
    num_events = min(len(tree_eta), events_per_file)
    jets = []
    for event in range(num_events):
        print('Processing event ' + str(event) + ' of ' + str(num_events), end='\r')
        num_jets = len(tree_eta[event])
        for jet in range(num_jets):
            size = len(tree_eta[event][jet])
            energy = tree_energy[event][jet]
            eta = tree_eta[event][jet]
            phi = tree_phi[event][jet]
            con_id = tree_id[event][jet]
            jet_arr = np.column_stack((energy, phi, eta, con_id))
            # pad jet with zeros, if necessary
            jets.append(jet_arr[:MAX_JET_LEN].tolist() + [[0,0,0,0]]*(MAX_JET_LEN - size))
    hdf5_file = h5py.File(name, 'w')
    dset = hdf5_file.create_dataset('jets', data=jets)

qcd_index_dir = "index/QCD/"
top_index_dir = "index/TOP/"
qcd_files = os.listdir(qcd_index_dir)
top_files = os.listdir(top_index_dir)

for qcd in qcd_files:
    qcd_f = open(qcd_index_dir + qcd)
    for line in qcd_f.readlines():
        url = "root://cmseos.fnal.gov//" + line.strip()
        fname = url.split('/')[-1].split('.')[0] + '.hdf5'
        name = storage_path + 'QCD/' + fname
        if url.split('/')[-1] == 'log' or os.path.exists(name):
            continue
        print('QCD file ' + line)
        root_file = uproot.open(url)
        tree = root_file['ljmet;1']['ljmet;1']
        try:
            write_hdf5(tree, name, EVENTS_PER_FILE)
        except Exception as e:
            print(e)
            print('Error writing file ' + name)

for top in top_files:
    top_f = open(top_index_dir + top)
    for line in top_f.readlines():
        url = "root://cmseos.fnal.gov//" + line.strip()
        fname = url.split('/')[-1].split('.')[0] + '.hdf5'
        name = storage_path + 'TOP/' + fname
        if name.split('/')[-1].split('.')[0] == 'log' or os.path.exists(name):
            continue
        print('Top file ' + line)
        root_file = uproot.open(url)
        tree = root_file['ljmet;1']['ljmet;1']
        try:
            write_hdf5(tree, name, EVENTS_PER_FILE)
        except Exception as e:
            print(e)
            print('Error writing file ' + name)
