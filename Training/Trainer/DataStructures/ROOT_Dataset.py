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
        # import pdb; pdb.set_trace()
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
        tracks = np.array(transposed_tracks)

        truth = torch.Tensor([int(truth) in [2, 6]])  # 'truth_type': 2/6=prompt; 3/7=HF
        lepton = torch.from_numpy(lepton).float()
        tracks = torch.from_numpy(np.array(tracks)).float()
        return truth, lepton, tracks

    def __len__(self):
        return len(self.event_order)


def collate(batch):
    """Finds the length of the non zero tensor.

    Args:
        track (torch.tensor): tensor containing the events padded with zeroes at the end
    Returns:
        Length (int) of the tensor were it not zero-padded
    """

    """pads the data with 0's"""
    length = torch.tensor([len(item[0]) for item in batch])
    max_size = int(length.max())
    data = [torch.nn.ZeroPad2d((0, 0, 0, max_size - len(item[0])))
            (item[0]) for item in batch]
    target = torch.stack([item[-1] for item in batch])
    not_rnn_data = torch.stack([item[1] for item in batch])
    return [torch.stack(data), not_rnn_data, target]
