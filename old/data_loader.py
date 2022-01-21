import os.path
import numpy as np
import random
import math
import pickle
random.seed(123)


# loading the dataset and removing the samples of the testing individual (leave one out)
def load_train_data(data_path,dataset_fname, fname_test, fname_unique,percentage_val):

    X_train = []
    Y_train = []

    X_valid = []
    Y_valid = []

    X_test = []
    Y_test = []

    X_data = np.load(os.path.join(data_path,dataset_fname+'_X_data.npy'))
    X_data_fname = np.load(os.path.join(data_path,dataset_fname+'_X_data_fname.npy'))
    Y_data = np.load(os.path.join(data_path,dataset_fname+'_Y_data.npy'))

    # removing the test sample from the list of unique IDs
    fname_unique = np.delete(fname_unique, np.where(fname_unique == fname_test))

    # selecting N Ids (without repetition) from the train set for validation
    valid_samples = np.random.choice(fname_unique, int(len(fname_unique)*percentage_val),replace=False)

    for i, f_name in enumerate(X_data_fname):
        if f_name==fname_test:
            X_test.append(X_data[i])
            Y_test.append(Y_data[i])
        else:
            if(f_name in valid_samples):
                X_valid.append(X_data[i])
                Y_valid.append(Y_data[i])
            else:
                X_train.append(X_data[i])
                Y_train.append(Y_data[i])

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    return X_train, np.array(Y_train), np.array(X_valid), np.array(Y_valid), X_test, np.array(Y_test)





###################################################################
def main(data_path,speech_th,w_size,w_th,w_step,p_val):

    # preprocessed data 'filename' (given the input parameters)
    dataset_fname = 'speech_th_' + str(speech_th) + '_w_size_' + str(w_size) + '_w_th_' + str(w_th) + '_w_step_' + str(w_step)

    # loading the unique subject IDs (e.g., 12_K, 12_L, 12_M, etc)
    unique_IDs = np.load(os.path.join(data_path,'preprocessed_data',dataset_fname+'_X_data_fname_unique.npy'))

    # for each subject (leave one out)
    for i, fname_test in enumerate(unique_IDs):

        print(f"Case {i+1}, Leaving 1 out: {fname_test}")

        # load the respective train / test samples
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_train_data(os.path.join(data_path,'preprocessed_data'),dataset_fname, fname_test, unique_IDs, p_val)

        # shufling the samples
        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]


        """
        ## Train here
        """

        """
        ## Predict here
        ##
        ## The multiple predictions of the same subject can be aggregated using the median (illustrated below)
           predictions = model.predict(x_test)
           agg_predictions = np.median(predictions, axis=0)
        """

        """
        ## Save the best model and/or predictions here so that we can evaluate it later using the MSE, for example
        """


###################################################################



#----------------------------------------------------
if __name__== "__main__":

    speech_th = 0.01    # speech activity
    w_size = 15         # window size
    w_th = 0.75         # percentage of "window size" with speech activity higher than 'speech_th'
    w_step = 1          # step used to shift the window when acquiring a new sample from the same trajectory
    p_val = 0.2         # percentage used for validation

    main('./data/',speech_th,w_size,w_th,w_step,p_val)
