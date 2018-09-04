import torch
import numpy as np
import itertools as it
import random

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
