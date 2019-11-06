import torch
import numpy as np
import random
from torch.utils.data import Dataset
from ROOT import TFile


class ROOT_Dataset(Dataset):
    """Dataset class that returns track and lepton data for a given event.

    Attributes:
        data_tree (TTree): the TTree containing all lepton and surrounding-track info, sorted by lepton
        readable_event_indices (list): which events are accessible by this Dataset (since train and test Datasets are read from the same TTree)
        options (dict): global configs, including lists of features found in the data_tree
    Methods:
        shuffle_indices: shuffles the readable event indices into a random order
    """

    def __init__(self, data_filename, readable_event_indices, options):
        super().__init__()
        self.data_file = TFile(data_filename)  # keep this open to prevent segfault
        self.data_tree = getattr(self.data_file, options["tree_name"])
        self.event_order = readable_event_indices
        self.shuffle_indices()
        self.options = options

    def shuffle_indices(self):
        random.shuffle(self.event_order)

    def __getitem__(self, index):
        """Returns the data at a given index."""
        self.data_tree.GetEntry(self.event_order[index])
        truth = self.data_tree.truth_type
        lepton = []
        transposed_tracks = []
        for lep_feature in self.options["lep_features"]:
            lepton.append(getattr(self.data_tree, lep_feature))
        for trk_feature in self.options["trk_features"]:
            transposed_tracks.append(list(getattr(self.data_tree, trk_feature)))
        lepton = np.array(lepton)
        tracks = np.transpose(transposed_tracks)

        truth = torch.Tensor([int(truth) in [2, 6]])  # 'truth_type': 2/6=prompt; 3/7=HF
        lepton = torch.from_numpy(lepton).float()
        tracks = torch.from_numpy(np.array(tracks)).float()
        return tracks, lepton, truth

    def __len__(self):
        return len(self.event_order)


def collate(batch):
    """Zero-pads batches.

    Args:
        batch (list): each element of the batch is a three-Tensor tuple consisting of (tracks, lepton, truth)
    Returns:
        [tracks_batch, lepton_batch, truth_batch]: tracks_batch is a 3D Tensor, lepton_batch is 2D, and truth_batch is 1D
    """

    length = torch.tensor([len(event[0]) for event in batch])
    max_size = int(length.max())
    tracks_batch = [torch.nn.ZeroPad2d((0, 0, 0, max_size - len(event[0])))(event[0]) for event in batch]  # pads the data with 0's
    tracks_batch = torch.stack(tracks_batch)
    truth_batch = torch.stack([event[-1] for event in batch])
    lepton_batch = torch.stack([event[1] for event in batch])
    return [tracks_batch, lepton_batch, truth_batch]
