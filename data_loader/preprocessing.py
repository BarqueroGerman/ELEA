import os.path
import pickle
import numpy as np
import glob
import csv
from tqdm import tqdm


def meanfilt (x, k):
    """Apply a length-k mean filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."

    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.mean (y, axis=1)




def preprocess_X_data(traj_big5, traj_big5_norm, speech_th, w_size, w_th, w_step, gt_data, use_speech):
    traits = ['wins_a_stat_target','O','C','E','A','N','sentiment','neutral','sad','angry','happy']
    smooth_filter_size = 15
    X_data = []
    X_data_fname = []
    X_data_fname_unique = []
    Y_data = []
    Y_data_unique = []

    for n, f_name in enumerate(traj_big5_norm):
        traj_smooth = dict()
        w_aux = dict()

        # filtering the trajectories
        for i, t in enumerate(traits):
            traj_smooth[t] = meanfilt(traj_big5_norm[f_name][t],smooth_filter_size)
        traj_smooth['speech_activity'] = meanfilt(traj_big5[f_name][t],smooth_filter_size)

        # for each point of the trajectory, extract the next '15' elements
        for j in range(0,len(traj_smooth[t])-w_size,w_step):
            w_aux_t = []

            # speech activity vector of each window
            aux_s = traj_smooth['speech_activity'][j:j+w_size] # using the non-normalized version for thresholding

            # number of points with high speech activity (>th)
            high_speech_act_idx = np.argwhere(aux_s>=speech_th)

            # if 3/4 of points are 'valid'
            if(len(high_speech_act_idx) >= w_size*w_th):

                #==============================================
                # WARNING HERE --> CONSIDERING SPEECH AS PART OF THE INPUT FEAT. VECTOR
                if(use_speech==True):
                    id_zero = 0
                else:
                    id_zero = 1
                # WARNING HERE --> CONSIDERING SPEECH AS PART OF THE INPUT FEAT. VECTOR
                #==============================================

                # get the trait values
                for i, t in enumerate(traits[id_zero:]):
                    aux_t = traj_smooth[t][j:j+w_size] # trait scores
                    w_aux_t.append(aux_t)

                X_data.append(w_aux_t) # saving as ex.: (15,5) or (w_size,traits) instead of (5,15)
                X_data_fname.append(f_name)
                Y_data.append(gt_data[f_name])
        X_data_fname_unique.append(f_name)
        Y_data_unique.append(gt_data[f_name])

    #
    # WARNING HERE: RESHAPING FROM (SAMPLES, NUM_FEAT, WINDOW_SIZE) TO (SAMPLES, NUM_FEAT x WINDOW_SIZE, 1)
    #
    #print('reshaping from ', np.array(X_data).shape)
    #X_data = np.array(X_data).reshape((np.array(X_data).shape[0],np.array(X_data).shape[1]*np.array(X_data).shape[2], 1))
    #print('to ', np.array(X_data).shape)
    X_data = np.array(X_data)
    X_data = np.transpose(X_data, (0, 2, 1))

    return X_data, X_data_fname, X_data_fname_unique, Y_data, Y_data_unique



def load_and_restructure_gt_data(gt_path, traj_list,dataset_fname):
    gt_data = dict()
    with open(gt_path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        next(reader, None) # skip the header
        for row in reader:
            id = str(row[0])+'_'+str(row[2])
            if (id not in gt_data):
                gt_data[id] = [float(row[9]), float(row[11]), float(row[7]), float(row[10]), float(row[8])] # OCEAN ORDER -> 7(E) 8(N) 9(O) 10(A) 11(C)


    ###########################################################
    # NORMALIZE THE Y LABELS (SELF-REPORTED VALUES), USING THOSE '64' INDIVIDUALS ONLY
    #
    unique_64_ids = []
    for ID in traj_list:
        if(ID) not in unique_64_ids: # list of IDs (e.g., '12_K')
            unique_64_ids.append(ID)

    norm_fact = dict()

    traits = ['O','C','E','A','N']
    for t in range(0,5): # compute the mean and std value per trait (ignoring the individuals that are not in the list of those '64')
        t_aux = []
        for i in unique_64_ids:
            # storing the per-trait scores for all unique individuals
            t_aux.append(gt_data[i][t])

        # # MEAN - STD NORMALIZATION
        # m = np.array(t_aux).mean() # mean value
        # s = np.array(t_aux).std() # std value

        # MIN - MAX NORMALIZATION
        min_v = min(t_aux)
        max_v = max(t_aux)

        norm_fact[traits[t]] = [min_v, max_v]

        # # # MEAN - STD NORMALIZATION
        # # plt.hist((np.array(t_aux)-m)/s)
        # # MIN - MAX NORMALIZATION
        # plt.hist((np.array(t_aux)-min_v)/(max_v-min_v))
        # plt.title(traits[t])
        # plt.grid(True)
        # plt.savefig('./output/gt_hist/trait_n_'+traits[t]+'.png')
        # plt.clf()

        # storing the normalized value
        for i in unique_64_ids:
            # gt_data[i][t] = (gt_data[i][t]-m)/s # mean - std
            gt_data[i][t] = (gt_data[i][t]-min_v)/(max_v-min_v) # min - max
            # gt_data[i][t] = gt_data[i][t]/5.0 # max (fixed to 5)


    ###########################################################
    if not os.path.exists('./output/preprocessed_data/'):
        os.makedirs('./output/preprocessed_data/')
    with open(os.path.join('./output/preprocessed_data/',dataset_fname+'_norm_factor.pkl'), 'wb') as handle:
        pickle.dump(norm_fact, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return gt_data



def normalize_trajectories(traj_big5_sent_emot):
    # accumulating the trajectories for each 'trait' (or feature)
    acc = dict()
    for ID in traj_big5_sent_emot:
        for feat in traj_big5_sent_emot[ID].keys():
            if(feat not in acc):
                acc[feat] = traj_big5_sent_emot[ID][feat]
            else:
                acc[feat] = np.concatenate([acc[feat], traj_big5_sent_emot[ID][feat]])

    # computing the mean/std per feature
    mean_v = dict()
    std_v = dict()
    for feat in acc:
        mean_v[feat] = np.mean(np.array(acc[feat]))
        std_v[feat] = np.std(np.array(acc[feat]))

    # normalizing the data
    traj_big5_sent_emot_norm = dict()
    for ID in traj_big5_sent_emot:
        aux_feat = dict()
        for feat in traj_big5_sent_emot[ID].keys():
            aux_feat[feat] = (traj_big5_sent_emot[ID][feat]-mean_v[feat])/std_v[feat]
        traj_big5_sent_emot_norm[ID] = aux_feat

    return traj_big5_sent_emot_norm




def load_big5_sent_emo_trajectories(traj_list,data_path):
    traits = ['O','C','E','A','N'] # traits
    traj_big5_sent_emot = dict()

    # loading the Big-5 trajactories
    for ID in tqdm(traj_list):
        #print(ID)
        traits_vec = dict()
        for t in traits:
            # loading the trajectory of each trait
            pkl_in = open(os.path.join(data_path,'trajectory_15s_big5','fi_avt_linmult_'+ID+'_'+t+'.pkl'),"rb")
            pkl_f = pickle.load(pkl_in)
            # trajectory of a particular trait
            traj_t = pkl_f['wins_preds']
            #
            # WARNING: REMOVING THE FIRST N (NAN) POINTS BASED ON BIG-5 TRAJECTORY
            #
            n_ignored_points = np.count_nonzero(np.isnan(traj_t))
            # storing the trajectories using a dictionary ['filename','trait']
            traits_vec[t] = traj_t[n_ignored_points:len(traj_t),0] # removing the first 'n_ignored_points' (nan) values

        # adding speech speech_activity
        traj_sa = pkl_f['wins_a_stat_target'] # percent spoken (target person) given a time window
        traits_vec['wins_a_stat_target'] = np.array(traj_sa[n_ignored_points:len(traj_sa)]) # removing the first 'n_ignored_points' (nan) values

        # sentiment trajectory
        pkl_in = open(os.path.join(data_path,'trajectory_10sec_sentiment_final','avt_shallow_'+ID+'.pkl'),"rb")
        pkl_f = pickle.load(pkl_in)
        traj_se = pkl_f['wins_preds']
        traits_vec['sentiment'] = np.array(traj_se[n_ignored_points:len(traj_se)]) # removing the first 'n_ignored_points' (nan) values
                                                                                   # (assuming the larger number of nan points, given by the big-5 traj.)

        # emotion trajectory (distribution) # [Neutral,Sad,Angry,Happy]
        pkl_in = open(os.path.join(data_path,'trajectory_10sec_emotion','avt_shallow_'+ID+'.pkl'),"rb")
        pkl_f = pickle.load(pkl_in)
        traj_em = pkl_f['wins_preds']

        # emo_vec = dict()
        traits_vec['neutral'] = traj_em[n_ignored_points:len(traj_em),0] # Neutral
        traits_vec['sad'] = traj_em[n_ignored_points:len(traj_em),1] # Sad
        traits_vec['angry'] = traj_em[n_ignored_points:len(traj_em),2] # Angry
        traits_vec['happy'] = traj_em[n_ignored_points:len(traj_em),3] # Happy

        # generating the new data structure for each person (big-five data)
        traj_big5_sent_emot[ID] = traits_vec

    return traj_big5_sent_emot




#
# load the list of filenames from path
#
def load_jubject_IDs(dir_path):
    traj_list = []
    root_dir = os.getcwd()
    os.chdir(dir_path)
    f_list = sorted(glob.glob('*.pkl'))
    for f in f_list:
        aux = f[15:len(f)-6]
        if(aux not in traj_list):
            traj_list.append(aux)
    os.chdir(root_dir)
    return traj_list




###################################################################333
def preprocess(data_path, speech_th, w_size, w_th, w_step, use_speech, normalize_x):

    # load the list of subject IDs (e.g. [12_K, 12_L, 14_M], etc)
    traj_list = load_jubject_IDs(os.path.join(data_path,'trajectory_15s_big5'))

    # loading the big-5 trajectories for the different participants/filenames
    traj_big5_sent_emot = load_big5_sent_emo_trajectories(traj_list,data_path)

    if(normalize_x==True):
        # normalizing the trajectories --> (X-mean) / std
        traj_big5_sent_emot_norm = normalize_trajectories(traj_big5_sent_emot)
    else:
        traj_big5_sent_emot_norm = traj_big5_sent_emot

    dataset_fname = 'speech_th_' + str(speech_th) + '_w_size_' + str(w_size) + '_w_th_' + str(w_th) + '_w_step_' + str(w_step)
    # loading and restructuring the ground truth data
    gt_data = load_and_restructure_gt_data(os.path.join(data_path,'elea_gt/personality_gender_age.csv'),traj_list,dataset_fname)

    # Generating multiple samples for each trajectory,
    # given the window size, speech activity threshold, etc
    X_data, X_data_fname, X_data_fname_unique, Y_data, Y_data_unique = preprocess_X_data(traj_big5_sent_emot, traj_big5_sent_emot_norm, speech_th, w_size, w_th, w_step, gt_data, use_speech)

    print('Data size and feaure vec. length', np.array(X_data).shape, np.array(X_data_fname).shape, np.array(X_data_fname_unique).shape, np.array(Y_data).shape, np.array(Y_data_unique).shape)

    # saving on disk the preprocessed data
    np.save(os.path.join(data_path,'preprocessed_data/',dataset_fname+'_X_data.npy'), X_data)
    np.save(os.path.join(data_path,'preprocessed_data/',dataset_fname+'_X_data_fname.npy'), X_data_fname)
    np.save(os.path.join(data_path,'preprocessed_data/',dataset_fname+'_X_data_fname_unique.npy'), X_data_fname_unique)
    np.save(os.path.join(data_path,'preprocessed_data/',dataset_fname+'_Y_data.npy'), Y_data)
    np.save(os.path.join(data_path,'preprocessed_data/',dataset_fname+'_Y_data_unique.npy'), Y_data_unique)

    print(f"Correctly saved to '{os.path.join(data_path, 'preprocessed_data')}'")
###################################################################




#----------------------------------------------------
if __name__== "__main__":

    speech_th = 0.01    # speech activity
    w_size = 15         # window size
    w_th = 0.75         # percentage of "window size" with speech activity higher than 'speech_th'
    w_step = 1          # step used to shift the window when acquiring a new sample from the same trajectory

    use_speech = False  # to include or not the speech activity fecture as part of the input feature fector
    normalize_x = False # normalize X data by (X-mean)/std

    preprocess('./data', speech_th, w_size, w_th, w_step, use_speech, normalize_x)
