import torch
import numpy as np
import itertools as it
import random
from torch.utils.data import DataLoader, Dataset
import torch
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
        self.read_order = it.chain(random.sample(range(len(self.leptons_with_tracks)), len(self.leptons_with_tracks)))

    def __next__(self):
        try:
            i = next(self.read_order)
        except StopIteration:
            self.reshuffle()
            i = next(self.read_order)
        lepton, tracks = self.leptons_with_tracks[i]
        lepton = torch.from_numpy(lepton).float()
        tracks = torch.from_numpy(np.array(tracks)).float()
        truth = torch.LongTensor([int(lepton[12]) in [2, 6]]) # 'truth_type': 2/6=prompt; 3/7=HF
        return truth, lepton, tracks

def collate(batch):
    '''pads the data with 0's'''
    length = torch.tensor([len(item[0]) for item in batch])
    max_size = int(length.max())
    data = [nn.ZeroPad2d((0, 0, 0, max_size - len(item[0])))
            (item[0]) for item in batch]
    target = torch.stack([item[1] for item in batch])
    return [torch.stack(data), target]

class Torchdata(Dataset):
    """takes a list of lepton data and uses
     Lepton Track Dataset to create a callable indexable object"""

    def __init__(self, lwt):
        super(Torchdata, self).__init__()
        self.file = LeptonTrackDataset(lwt)

    def __getitem__(self, index, length=False):
        '''gets the data at a given index'''
        dataiter = next(self.file)
        for i in range(index - 1):
            # there is probably a better way to do this
            dataiter = next(self.file)

        if length is False:
            return dataiter[2], dataiter[0]
        else:
            return dataiter[2], len(dataiter[2])

    def __len__(self):
        return len(self.file)
