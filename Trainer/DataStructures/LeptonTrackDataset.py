import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torch.nn as nn


class LeptonTrackDataset:

    def __init__(self, leptons_with_tracks):
        self.leptons_with_tracks = leptons_with_tracks
        self.reshuffle()

    def __len__(self):
        return len(self.leptons_with_tracks)

    def __iter__(self):
        return self

    def reshuffle(self):
        self.read_order = random.sample(
            range(len(self.leptons_with_tracks)),
            len(self.leptons_with_tracks))

    def get(self, index):
        i = self.read_order[index]
        lepton, tracks = self.leptons_with_tracks[i]
        lepton = torch.from_numpy(lepton).float()
        tracks = torch.from_numpy(np.array(tracks)).float()
        # 'truth_type': 2/6=prompt; 3/7=HF
        truth = torch.Tensor([int(lepton[12]) in [2, 6]])
        return truth, lepton, tracks


def collate(batch):
    '''pads the data with 0's'''
    length = torch.tensor([len(item[0]) for item in batch])
    max_size = int(length.max())
    data = [nn.ZeroPad2d((0, 0, 0, max_size - len(item[0])))
            (item[0]) for item in batch]
    target = torch.stack([item[-1] for item in batch])
    not_rnn_data = torch.stack([item[1] for item in batch])
    return [torch.stack(data), not_rnn_data, target]


class Torchdata(Dataset):
    """takes a list of lepton data and uses
     Lepton Track Dataset to create a callable indexable object"""

    def __init__(self, lwt):
        super(Torchdata, self).__init__()
        self.file = LeptonTrackDataset(lwt)

    def __getitem__(self, index):
        '''gets the data at a given index'''
        data = (self.file.get(index))

        return data[2], data[1], data[0]

    def __len__(self):
        return len(self.file)
