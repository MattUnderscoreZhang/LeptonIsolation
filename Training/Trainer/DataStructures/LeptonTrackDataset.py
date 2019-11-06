import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torch.nn as nn


class LeptonTrackDataset:
    """Class implementing an iterator for lepton classification data

    Attributes:
        leptons_with_tracks (list) : lepton data consisting of:
                                                            * lepton information
                                                            * track information
                                                            * truth value (for the classifier)

    Methods:
        reshuffle: shuffles the dataset passed into the constructor
        get: getter function that returns the information at a given index

    """

    def __init__(self, leptons_with_tracks):
        self.leptons_with_tracks = leptons_with_tracks
        self.reshuffle()

    def __len__(self):
        return len(self.leptons_with_tracks)

    def __iter__(self):
        return self

    def reshuffle(self):
        """Shuffles the dataset

        Args:
            None

        Returns:
            None
        """
        self.read_order = random.sample(
            range(len(self.leptons_with_tracks)), len(self.leptons_with_tracks)
        )

    def get(self, index):
        """Finds the data at a given index

       Args:
            index (int) : index of element of interest

        Returns:
            truth, leptons, tracks : tuple of data contained at the location

        """
        i = self.read_order[index]
        lepton, tracks = self.leptons_with_tracks[i]
        lepton = torch.from_numpy(lepton).float()
        tracks = torch.from_numpy(np.array(tracks)).float()
        # 'truth_type': 2/6=prompt; 3/7=HF
        truth = torch.Tensor([int(lepton[12]) in [2, 6]])
        return truth, lepton, tracks


def collate(batch):
    """Finds the length of the non zero tensor

    Args:
        track (torch.tensor): tensor containing the events padded with zeroes at the end

    Returns:
        Length (int) of the tensor were it not zero-padded

    """

    """pads the data with 0's"""
    length = torch.tensor([len(item[0]) for item in batch])
    max_size = int(length.max())
    data = [nn.ZeroPad2d((0, 0, 0, max_size - len(item[0])))
            (item[0]) for item in batch]
    target = torch.stack([item[-1] for item in batch])
    not_rnn_data = torch.stack([item[1] for item in batch])
    return [torch.stack(data), not_rnn_data, target]


class Torchdata(Dataset):
    """Class that inherits from torch's dataset that takes takes a list of lepton data and uses Lepton Track Dataset to create a callable indexable
     object

    Attributes:
        leptons_with_tracks (list) : lepton data consisting of:
                                                            * lepton information
                                                            * track information
                                                            * truth value (for the classifier)

    Methods:
        reshuffle: shuffles the dataset passed into the constructor
        get: getter function that returns the information at a given index

    """

    def __init__(self, lwt):
        super(Torchdata, self).__init__()
        self.file = LeptonTrackDataset(lwt)

    def __getitem__(self, index):
        """Finds the data at a given index

       Args:
            index (int) : index of element of interest

        Returns:
            truth, leptons, tracks : tuple of data contained at the location

        """
        data = self.file.get(index)

        return data[2], data[1], data[0]

    def __len__(self):
        return len(self.file)
