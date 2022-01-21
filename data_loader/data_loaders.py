from torchvision import transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import functools
from functools import partial
import torch
import pandas as pd
import hashlib 
import itertools
from data_loader.preprocessing import preprocess

class BaseDatasetELEA(Dataset):
    def __init__(self, directory, transform=None, target_transform=None, speech_th=0.01, w_size=15, w_th=0.75, w_step=1):
        super().__init__()
        self.directory = directory
        self.filename = 'speech_th_' + str(speech_th) + '_w_size_' + str(w_size) + '_w_th_' + str(w_th) + '_w_step_' + str(w_step)
        
        self.transform = transform
        self.target_transform = target_transform

        self.check_preprocessed_files(speech_th, w_size, w_th, w_step)
        self.participant2personality = self.load_participants()
        self.participant2segments = self.load_segments_per_participant()
        assert self.participant2personality.keys() == self.participant2segments.keys()
        self.participants_ids = list(self.participant2personality.keys())
        self.n_samples = np.sum([self.participant2segments[part_id].shape[0] for part_id in self.participants_ids])

    def check_preprocessed_files(self, speech_th, w_size, w_th, w_step, use_speech=False, normalize_x=False):
        ids_filename = os.path.join(self.directory, self.filename + '_X_data_fname_unique.npy')
        Ys_filename = os.path.join(self.directory, self.filename + '_Y_data_unique.npy')
        Xs_filename = os.path.join(self.directory, self.filename + '_X_data.npy')
        X_ids_filename = os.path.join(self.directory, self.filename + '_X_data_fname.npy')
        if not os.path.exists(ids_filename) or not os.path.exists(Ys_filename) or not os.path.exists(Xs_filename) or not os.path.exists(X_ids_filename):
            print(f"[INFO] 'npy' files of the dataset not found. Preprocessing dataset...")
            preprocess('./data', speech_th, w_size, w_th, w_step, use_speech, normalize_x)

    def load_participants(self):
        X_data_fname_unique = np.load(os.path.join(self.directory, self.filename + '_X_data_fname_unique.npy'))
        Y_data_unique = np.load(os.path.join(self.directory, self.filename + '_Y_data_unique.npy'))

        pers_dict = {}
        for i in range(X_data_fname_unique.shape[0]):
            participant = X_data_fname_unique[i]
            ocean = Y_data_unique[i]
            pers_dict[str(participant)] = ocean
        return pers_dict

    def load_segments_per_participant(self):
        X_data = np.load(os.path.join(self.directory, self.filename + '_X_data.npy'))
        X_data_ids = np.load(os.path.join(self.directory, self.filename + '_X_data_fname.npy'))

        features_dict = {}
        for participant_id in np.unique(X_data_ids): # we do not assume continuity for the IDs (can't slice)
            ids_participant = [feat_id for feat_id in range(X_data.shape[0]) if X_data_ids[feat_id] == participant_id]
            features_dict[participant_id] = X_data[ids_participant]
        return features_dict

    def __len__(self):
        return self.n_samples # the main entity of the dataset is a segment

    def __getitem__(self, idx):
        # we split idx into participant + segment_id
        counter = 0
        i, found = 0, False
        while i < len(self.participants_ids) and not found:
            part_id = self.participants_ids[i]
            num_segments_part = self.participant2segments[part_id].shape[0]
            if counter + num_segments_part > idx:
                found = True
                idx_part = part_id
                idx_segment = idx - counter
            else:
                counter += num_segments_part
                i += 1
        if not found:
            raise Exception(f"Error fetching segment with ID {idx}")

        #print((idx_part, idx_segment))
        # return (segment, target (personality))
        return self._transform(self.participant2segments[idx_part][idx_segment], self.participant2personality[idx_part])

    def _transform(self, segments, target):
        return self.transform(segments).squeeze() if self.transform is not None else segments, \
                self.target_transform(target).squeeze() if self.target_transform is not None else target
    


class DataLoaderELEA(BaseDataLoader):
    def __init__(self, data_dir, batch_size, speech_th=0.01, w_size=15, w_th=0.75, w_step=1, p_val=0.25, shuffle=True, num_workers=1):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        target_trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.dataset = BaseDatasetELEA(data_dir, transform=trsfm, target_transform=target_trsfm, speech_th=speech_th, w_size=w_size, w_th=w_th, w_step=w_step)
        
        super().__init__(self.dataset, batch_size, shuffle, p_val, num_workers, drop_last=True)


