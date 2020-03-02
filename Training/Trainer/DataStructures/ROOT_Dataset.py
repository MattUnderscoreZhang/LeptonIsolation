import torch
import numpy as np
import random
from torch.utils.data import Dataset
from ROOT import TFile
import pandas as pd


class ROOT_Dataset(Dataset):
    """Dataset class that returns track and lepton data for a given event.

    Attributes:
        data_tree (TTree): the TTree containing all lepton and surrounding-track info, sorted by lepton
        readable_event_indices (list): which events are accessible by this Dataset (since train and test Datasets are read from the same TTree)
        options (dict): global configs, including lists of features found in the data_tree
    Methods:
        shuffle_indices: shuffles the readable event indices into a random order
    """

    def __init__(
        self, data_filename, readable_event_indices, options, shuffle_indices=True
    ):
        super().__init__()
        self.data_file = TFile(data_filename)  # keep this open to prevent segfault
        self.data_tree_on_disk = self.data_file.Get(options["tree_name"])
        self.options = options
        if readable_event_indices is None:
            return
        self.event_order = readable_event_indices
        if shuffle_indices:
            random.shuffle(self.event_order)
        self.data_tree = self._store_tree_in_memory(
            self.data_tree_on_disk, self.event_order, self.options
        )

    def get_readable_events(self):
        tree = self.data_tree_on_disk
        event_indices = list(range(tree.GetEntries()))
        bad_indices = []
        for i, event in enumerate(tree):
            # filter events with no calo clusters
            n_calo_clusters = len(
                list(getattr(event, self.options["calo_features"][0]))
            )
            if n_calo_clusters == 0 or i in [2, 14, 52]:
                bad_indices.append(i)
        bad_indices.reverse()
        for i in bad_indices:
            event_indices.pop(i)
        return np.array(event_indices)

    def _store_tree_in_memory(self, tree, event_order, options):
        def _sort_tracks(tracks, track_ordering, track_features):
            if track_ordering in ["high-to-low-pt", "low-to-high-pt"]:
                tracks_pT = tracks[:, track_features.index("trk_pT")]
                sorted_indices = torch.argsort(tracks_pT, descending=True)
                if track_ordering == "low-to-high-pt":
                    sorted_indices = sorted_indices.flip(0)
                tracks = tracks[sorted_indices]
            elif track_ordering in ["near-to-far", "far-to-near"]:
                tracks_dR = tracks[:, track_features.index("trk_lep_dR")]
                sorted_indices = torch.argsort(tracks_dR, descending=True)
                if track_ordering == "near-to-far":
                    sorted_indices = sorted_indices.flip(0)
                tracks = tracks[sorted_indices]
            return tracks

        tree_info = []
        for index in event_order:
            tree.GetEntry(index)
            lepton = [
                getattr(tree, lep_feature) for lep_feature in options["lep_features"]
            ]
            lepton = [0 if np.isnan(value) else value for value in lepton]
            transposed_tracks = [
                list(getattr(tree, trk_feature))
                for trk_feature in options["trk_features"]
            ]
            tracks = np.transpose(transposed_tracks)
            transposed_clusters = [
                list(getattr(tree, calo_feature))
                for calo_feature in options["calo_features"]
            ]
            clusters = np.transpose(transposed_clusters)
            truth = tree.truth_type
            lepton = torch.Tensor(lepton)
            tracks = torch.Tensor(tracks)
            clusters = torch.Tensor(clusters)
            tracks = _sort_tracks(
                tracks, self.options["track_ordering"], self.options["trk_features"]
            )
            truth = torch.Tensor(
                [int(truth) in [2, 6]]
            )  # 'truth_type': 2/6=prompt; 3/7=HF
            lep_pT = torch.Tensor([getattr(tree, "ROC_slicing_lep_pT")])
            tree_info.append((lepton, tracks, clusters, truth, lep_pT))

        n_additional_features = len(options["additional_appended_features"])
        n_natural_lep_features = len(options["lep_features"]) - n_additional_features
        if n_additional_features > 0:
            additional_vars = np.array(
                [i[0][-n_additional_features:].cpu().numpy() for i in tree_info]
            )
            additional_var_means = np.append(
                np.zeros(n_natural_lep_features), np.mean(additional_vars, axis=0)
            )
            additional_var_stds = np.append(
                np.ones(n_natural_lep_features), np.std(additional_vars, axis=0)
            )
            tree_info = [
                (
                    torch.from_numpy(
                        (i.cpu().numpy() - additional_var_means) / additional_var_stds
                    ).float(),
                    j,
                    k,
                    l,
                    m,
                )
                for (i, j, k, l, m) in tree_info
            ]

        return tree_info

    def __getitem__(self, index):
        """Returns the data at a given index."""
        lepton, tracks, clusters, truth, lep_pT = self.data_tree[index]
        event = pd.Series(
            [
                tracks.cpu().numpy(),
                tracks.shape[0],
                clusters.cpu().numpy(),
                clusters.shape[0],
                lepton,
                bool(truth),
                lep_pT,
            ],
            index=[
                "track_info",
                "track_length",
                "calo_info",
                "calo_length",
                "lepton_info",
                "truth",
                "lepton_pT",
            ],
        )
        return event

    def __len__(self):
        return len(self.event_order)


def collate(batch):
    """Zero-pads batches.

    Args:
        batch (list): each element of the list is a pandas dataframe with event information
    Returns:
        dictionary with batched event information
    """

    def _zero_pad_track(event, max_track_size):
        return torch.nn.ZeroPad2d((0, 0, 0, max_track_size - event.track_length))(
            torch.from_numpy(event.track_info)
        )

    def _zero_pad_calo(event, max_calo_size):
        return torch.nn.ZeroPad2d((0, 0, 0, max_calo_size - event.calo_length))(
            torch.from_numpy(event.calo_info)
        )

    batch = pd.concat(batch, axis=1).transpose()
    max_track_size = batch.track_length.max()
    max_calo_size = batch.calo_length.max()

    collated_batch = pd.Series()
    collated_batch["track_info"] = torch.stack(
        tuple(batch.apply(_zero_pad_track, args=(max_track_size,), axis=1))
    )
    collated_batch["calo_info"] = torch.stack(
        tuple(batch.apply(_zero_pad_calo, args=(max_calo_size,), axis=1))
    )
    collated_batch["lepton_info"] = torch.stack(tuple(batch.lepton_info))
    collated_batch["track_length"] = torch.tensor(batch.track_length.values.astype(int))
    collated_batch["calo_length"] = torch.tensor(batch.calo_length.values.astype(int))
    collated_batch["truth"] = torch.tensor(tuple(batch.truth))
    collated_batch["lepton_pT"] = torch.tensor(tuple(batch.lepton_pT))

    return collated_batch.to_dict()
