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

    def __init__(self, data_filename, readable_event_indices, options, shuffle_indices=True):
        super().__init__()
        self.data_file = TFile(data_filename)  # keep this open to prevent segfault
        self.data_tree_on_disk = getattr(self.data_file, options["tree_name"])
        self.event_order = readable_event_indices
        if shuffle_indices:
            random.shuffle(self.event_order)
        self.options = options
        self.data_tree = self._store_tree_in_memory(self.data_tree_on_disk, self.event_order, self.options)

    def _store_tree_in_memory(self, tree, event_order, options):
        def _sort_tracks(tracks, track_ordering, track_features):
            if track_ordering in ["high-to-low-pt", "low-to-high-pt"]:
                tracks_pT = tracks[:, track_features.index("trk_pT")]
                _, sorted_indices = torch.sort(tracks_pT, descending=True)
                if track_ordering == "low-to-high-pt":
                    sorted_indices = sorted_indices.flip(0)
                tracks = tracks[sorted_indices]
            elif track_ordering in ["near-to-far", "far-to-near"]:
                tracks_dR = tracks[:, track_features.index("trk_lep_dR")]
                _, sorted_indices = torch.sort(tracks_dR, descending=True)
                if track_ordering == "near-to-far":
                    sorted_indices = sorted_indices.flip(0)
                tracks = tracks[sorted_indices]
            return tracks

        tree_info = []
        for index in event_order:
            tree.GetEntry(index)
            lepton = [getattr(tree, lep_feature) for lep_feature in options["lep_features"]]
            transposed_tracks = [list(getattr(tree, trk_feature)) for trk_feature in options["trk_features"]]
            tracks = np.transpose(transposed_tracks)
            truth = tree.truth_type
            lepton = torch.Tensor(lepton)
            tracks = torch.Tensor(tracks)
            tracks = _sort_tracks(tracks, self.options["track_ordering"], self.options["trk_features"])
            truth = torch.Tensor([int(truth) in [2, 6]])  # 'truth_type': 2/6=prompt; 3/7=HF
            tree_info.append((lepton, tracks, truth))
        return tree_info

    def __getitem__(self, index):
        """Returns the data at a given index."""
        lepton, tracks, truth = self.data_tree[index]
        return tracks, lepton, truth, tracks.shape[0]

    def __len__(self):
        return len(self.event_order)


def collate(batch):
    """Zero-pads batches.

    Args:
        batch (list): each element of the batch is a three-Tensor tuple consisting of (tracks, lepton, truth)
    Returns:
        [tracks_batch, lepton_batch, truth_batch]: tracks_batch is a 3D Tensor, lepton_batch is 2D, and truth_batch is 1D
    """
    batch = np.array(batch)
    length = torch.from_numpy(batch[:, 3].astype(int))
    max_size = length.max()
    tracks_batch = [torch.nn.ZeroPad2d((0, 0, 0, max_size - event[3]))(event[0]) for event in batch]  # pads the data with 0's
    tracks_batch = torch.stack(tracks_batch)
    truth_batch = torch.from_numpy(batch[:, 2].astype(int))
    lepton_batch = torch.stack(batch[:, 1].tolist())
    return [tracks_batch, lepton_batch, truth_batch, length]
