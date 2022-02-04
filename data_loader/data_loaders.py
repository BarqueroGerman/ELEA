
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
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


class BaseDatasetELEA(Dataset):
    def __init__(self, directory, transform=None, target_transform=None, speech_th=0.01, w_size=15, w_th=0.75, w_step=1, log=False):
        super().__init__()
        self.directory = directory
        self.filename = 'speech_th_' + str(speech_th) + '_w_size_' + str(w_size) + '_w_th_' + str(w_th) + '_w_step_' + str(w_step)
        self.log = log
        
        self.transform = transform
        self.target_transform = target_transform

        self.check_preprocessed_files(speech_th, w_size, w_th, w_step)
        self.participant2personality = self.load_participants()
        self.participant2segments = self.load_segments_per_participant()
        assert self.participant2personality.keys() == self.participant2segments.keys()
        self.participants_ids = list(self.participant2personality.keys())
        self.n_samples = np.sum([self.participant2segments[part_id].shape[0] for part_id in self.participants_ids])

        """
        self.leftout_idx = leftout_idx
        if self.leftout_idx is not None:
            self.leftout_idx(leave_one_out_idx)
        else:
            print(f"[WARNING] Leave-one-out is disabled at the dataset level.")
        """

    # provisional. Ideally, make sampler
    """
    def leave_one_out(self, idx):
        assert idx in self.participants_ids, f"{idx} does not correspond to any participant (leave one out error)"
        segments_length = self.participant2segments[idx].shape[0]
        self.n_samples -= segments_length
        del self.participant2personality[idx]
        del self.participant2segments[idx]
        self.participants_ids.remove(idx)
    """
    def get_GT_personality(self, p_id):
        return self.participant2personality[p_id]

    def check_preprocessed_files(self, speech_th, w_size, w_th, w_step, use_speech=False, normalize_x=False):
        ids_filename = os.path.join(self.directory, self.filename + '_X_data_fname_unique.npy')
        Ys_filename = os.path.join(self.directory, self.filename + '_Y_data_unique.npy')
        Xs_filename = os.path.join(self.directory, self.filename + '_X_data.npy')
        X_ids_filename = os.path.join(self.directory, self.filename + '_X_data_fname.npy')
        if not os.path.exists(ids_filename) or not os.path.exists(Ys_filename) or not os.path.exists(Xs_filename) or not os.path.exists(X_ids_filename):
            if self.log:
                print(f"[INFO] 'npy' files of the dataset not found. Preprocessing dataset...")
            preprocess('./data', speech_th, w_size, w_th, w_step, use_speech, normalize_x)

    def load_participants(self):
        X_data_fname_unique = np.load(os.path.join(self.directory, self.filename + '_X_data_fname_unique.npy'))
        Y_data_unique = np.load(os.path.join(self.directory, self.filename + '_Y_data_unique.npy'))

        pers_dict = {}
        for i in range(X_data_fname_unique.shape[0]):
            participant = X_data_fname_unique[i]
            ocean = Y_data_unique[i]
            pers_dict[str(participant)] = ocean.astype(float)
        return pers_dict

    def load_segments_per_participant(self):
        X_data = np.load(os.path.join(self.directory, self.filename + '_X_data.npy'))
        X_data_ids = np.load(os.path.join(self.directory, self.filename + '_X_data_fname.npy'))

        features_dict = {}
        for participant_id in np.unique(X_data_ids): # we do not assume continuity for the IDs (can't slice)
            ids_participant = [feat_id for feat_id in range(X_data.shape[0]) if X_data_ids[feat_id] == participant_id]
            features_dict[participant_id] = X_data[ids_participant].astype(float)
        return features_dict

    def split_validation_set(self, p_val, leftout_idx=-1):
        assert leftout_idx == -1 or leftout_idx in self.participants_ids#< len(self.participants_ids)
        if leftout_idx != -1:
            p_leftout = leftout_idx#self.participants_ids[leftout_idx]
            ps_without_leftout = [p for p in self.participants_ids if p != p_leftout]
        else:
            ps_without_leftout = self.participants_ids

        num_valid_parts = int(np.round(len(ps_without_leftout) * p_val))
        participants_valid = np.random.choice(ps_without_leftout, num_valid_parts, replace=False)

        counter = 0
        valid_idces, train_idces, leftout_idces = [], [], []
        for part_id in self.participants_ids:
            current_segments_len = self.participant2segments[part_id].shape[0]
            if part_id in participants_valid:
                # VALIDATION
                valid_idces += list(range(counter, counter + current_segments_len))
            elif leftout_idx != -1 and part_id == p_leftout:
                # TEST (LEAVE-ONE-OUT)
                leftout_idces += list(range(counter, counter + current_segments_len))
            else:
                # TRAINING
                train_idces += list(range(counter, counter + current_segments_len))
            counter += current_segments_len

        if self.log:
            print(f"[INFO] Participants in training/validation{'/test' if leftout_idx != -1 else ''}: {num_valid_parts}/{len(ps_without_leftout)}{'/1' if leftout_idx != -1 else ''} ({100 * len(valid_idces) / self.n_samples:.1f}%{', ' + p_leftout if leftout_idx != -1 else ''})")
            print(f"[INFO] Segments in training/validation{'/test' if leftout_idx != -1 else ''}: {len(train_idces)}/{len(valid_idces)}{'/' + str(len(leftout_idces)) if leftout_idx != -1 else ''}")
        
        return train_idces, valid_idces, leftout_idces

    def __len__(self):
        return self.n_samples # the main entity of the dataset is a segment

    def get_num_participants(self):
        return len(self.participants_ids)

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
        return torch.FloatTensor(self.participant2segments[idx_part][idx_segment]), torch.FloatTensor(self.participant2personality[idx_part])
    


class DataLoaderELEA(BaseDataLoader):
    def __init__(self, data_dir, batch_size, speech_th=0.01, w_size=15, w_th=0.75, w_step=1, p_val=0.25, 
                shuffle=True, num_workers=1, leftout_idx=-1, log=False, drop_last=True):
        
        self.dataset = BaseDatasetELEA(data_dir, speech_th=speech_th, w_size=w_size, w_th=w_th, w_step=w_step, log=log)
        
        if drop_last: # if willingly set to false, then we keep that decision. If not, set to False if total length does not fit to batch size.
            drop_last = True if batch_size < len(self.dataset) * min(p_val, 1-p_val) else False
        super().__init__(self.dataset, batch_size, shuffle, p_val, num_workers, drop_last=drop_last, leftout_idx=leftout_idx)


    # we do it here because we want to split by participant and not simply idces.
    def _split_sampler(self, valid_split):
        if valid_split == 0.0:
            return None, None

        train_idces, valid_idces, _ = self.dataset.split_validation_set(valid_split)

        if self.shuffle:
            train_sampler = SubsetRandomSampler(train_idces)
        else:
            train_sampler = SequentialSampler(train_idces)

        valid_sampler = SequentialSampler(valid_idces) # valid does not benefit from shuffling

        # when using samples, shuffling becomes useless
        self.shuffle = False
        self.n_samples = len(train_idces)

        return train_sampler, valid_sampler


    # we do it here because we want to split by participant and not simply idces.
    def _split_sampler_leaveoneout(self, valid_split, leftout_idx):
        #if leftout_idx == -1:
        #    print("here")
        #    train_sampler, valid_sampler = self._split_sampler(valid_split)
        #    return train_sampler, valid_sampler, None

        train_idces, valid_idces, test_idces = self.dataset.split_validation_set(valid_split, leftout_idx=leftout_idx)
        #print(f"<<<< {(len(train_idces), len(valid_idces), len(test_idces))}>>>>")

        if self.shuffle:
            train_sampler = SubsetRandomSampler(train_idces)
        else:
            train_sampler = SequentialSampler(train_idces)

        test_sampler = SequentialSampler(test_idces) # valid does not benefit from shuffling

        # when using samples, shuffling becomes useless
        self.shuffle = False
        self.n_samples = len(train_idces)

        if valid_split == 0:
            return train_sampler, None, test_sampler

        # only if valid_split != 0
        valid_sampler = SequentialSampler(valid_idces) # valid does not benefit from shuffling

        return train_sampler, valid_sampler, test_sampler
