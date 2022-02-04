import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

SEED = 6

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, 
                    collate_fn=default_collate,    
                    drop_last=False, 
                    leftout_idx=-1):
        self.validation_split = validation_split
        self.leftout_idx = leftout_idx
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler, self.test_sampler = self._split_sampler_leaveoneout(self.validation_split, leftout_idx)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'drop_last': self.drop_last
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(SEED)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

        # we do it here because we want to split by participant and not simply idces.
    def _split_sampler_leaveoneout(self, valid_split, leftout_idx):
        raise NotImplementedError

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def split_validation_leaveoneout(self):
        if self.valid_sampler is None and self.test_sampler is None:
            # valid and test sets disabled
            return None, None
        elif self.valid_sampler is None:
            # only test set enabled
            return None, DataLoader(sampler=self.test_sampler, **self.init_kwargs)
        elif self.test_sampler is None:
            # only valid set enabled
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs), None
        else:
            # both valid and test sets enabled
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs), DataLoader(sampler=self.test_sampler, **self.init_kwargs)


    def unnormalize(self, x):
        return x * self.std + self.mean