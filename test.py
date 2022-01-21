import os.path
import pickle
import numpy as np
import glob
import csv
from tqdm import tqdm
from data_loader import BaseDatasetELEA

###################################################################333
def main(data_path, speech_th, w_size, w_th, w_step, use_speech, normalize_x):

    dataset_fname = 'speech_th_' + str(speech_th) + '_w_size_' + str(w_size) + '_w_th_' + str(w_th) + '_w_step_' + str(w_step)
    
    """
    # saving on disk the preprocessed data
    X_data = np.load(os.path.join(data_path,'preprocessed_data/',dataset_fname+'_X_data.npy'))
    X_data_fname = np.load(os.path.join(data_path,'preprocessed_data/',dataset_fname+'_X_data_fname.npy'))
    X_data_fname_unique = np.load(os.path.join(data_path,'preprocessed_data/',dataset_fname+'_X_data_fname_unique.npy'))
    Y_data = np.load(os.path.join(data_path,'preprocessed_data/',dataset_fname+'_Y_data.npy'))

    print(X_data.shape)
    #print(X_data[0][:5])
    print(X_data_fname.shape)
    #print(X_data_fname[:10])
    print(X_data_fname_unique.shape)
    #print(X_data_fname_unique[:10])
    print(Y_data.shape)
    #print(Y_data[:3])
"""

    dataset = BaseDatasetELEA('./data/preprocessed_data', transform=None, target_transform=None)
    print(len(dataset))
    for i in range(960, 970):#len(dataset)):
        features, target = dataset[i]
        print((features.shape, target))

    print(f"Correctly loaded!")
###################################################################




#----------------------------------------------------
if __name__== "__main__":

    speech_th = 0.01    # speech activity
    w_size = 15         # window size
    w_th = 0.75         # percentage of "window size" with speech activity higher than 'speech_th'
    w_step = 1          # step used to shift the window when acquiring a new sample from the same trajectory

    use_speech = False  # to include or not the speech activity fecture as part of the input feature fector
    normalize_x = False # normalize X data by (X-mean)/std

    main('./data', speech_th, w_size, w_th, w_step, use_speech, normalize_x)
