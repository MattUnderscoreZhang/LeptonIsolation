import pdb
import torch
import numpy as np
import random
import itertools as it
from Loader.loader import load
from Architectures.RNN import RNN
from Analysis.cones import compare_ptcone_and_etcone

################################################
# Data structures for holding training results #
################################################

class historyData(list):
    def __init__(self, my_list=[]):
        super().__init__(my_list)
    def extend(self, places):
        for _ in range(places):
            self.append(historyData())
    def __getitem__(self, key): # overloads list[i] for i out of range
        if key >= len(self):
            self.extend(key+1-len(self))
        return super().__getitem__(key)
    def __setitem__(self, key, item): # overloads list[i]=x for i out of range
        if key >= len(self):
            self.extend(key+1-len(self))
        super().__setitem__(key, item)
    def __iadd__(self, key): # overloads []+=x
        return key
    def __add__(self, key): # overloads []+x
        return key

LOSS, ACC = 0, 1
TRAIN, VALIDATION, TEST = 0, 1, 2
BATCH, EPOCH = 0, 1

##################
# Train and test #
##################

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
        # i = next(self.read_order)
        try:
            i = next(self.read_order)
        except StopIteration:
            self.reshuffle()
            i = next(self.read_order)
        event = self.leptons_with_tracks[i]
        lepton = torch.from_numpy(event[0]).float()
        tracks = torch.from_numpy(np.array(event[1])).float()
        return lepton, tracks

def train_and_test(leptons_with_tracks, options):

    # split train and test
    n_events = len(leptons_with_tracks)
    n_training_events = int(options['training_split'] * n_events)
    training_events = leptons_with_tracks[:n_training_events]
    test_events = leptons_with_tracks[n_training_events:]

    # prepare the generators
    train_set = LeptonTrackDataset(training_events)
    test_set = LeptonTrackDataset(test_events)

    # set up RNN
    options['n_track_features'] = len(training_events[0][1][0])
    rnn = RNN(options)

    # training results e.g. history[CLASS_LOSS][TRAIN][EPOCH]
    history = historyData()

    # train RNN
    training_loss = 0
    training_acc = 0
    for batch_n in range(options['n_batches']):
        training_batch = []
        for i in range(options['batch_size']):
            next_event = next(train_set)
            truth = torch.LongTensor([(int(next_event[0][11]) == 3)]) # 'truth_type' - 3 = prompt; 4 = HF
            training_batch.append([truth, next_event])
        test_batch = []
        for i in range(options['batch_size']):
            next_event = next(test_set)
            truth = torch.LongTensor([(int(next_event[0][11]) == 3)]) # 'truth_type' - 3 = prompt; 4 = HF
            test_batch.append([truth, next_event])
        train_loss, train_acc = rnn.do_train(training_batch)
        test_loss, test_acc = rnn.do_eval(test_batch)
        history[LOSS][TRAIN][BATCH].append(train_loss)
        history[ACC][TRAIN][BATCH].append(train_acc)
        history[LOSS][TEST][BATCH].append(test_loss)
        history[ACC][TEST][BATCH].append(test_acc)
        print("Batch: %d, Train Loss: %0.2f, Train Acc: %0.2f, Test Loss: %0.2f, Test Acc: %0.2f" % (batch_n, train_loss, train_acc, test_loss, test_acc))

#################
# Main function #
#################

if __name__ == "__main__":

    # prepare data
    in_file = "Data/output.h5"
    save_file = "Data/lepton_track_data.pkl"
    leptons_with_tracks = load(in_file, save_file, overwrite=False)

    ## make ptcone and etcone comparison plots
    # plot_save_dir = "../Plots/"
    # compare_ptcone_and_etcone.compare_ptcone_and_etcone(leptons_with_tracks, plot_save_dir)

    # perform training
    options = {}
    options['n_hidden_neurons'] = 256
    options['learning_rate'] = 0.01
    options['training_split'] = 0.66
    options['batch_size'] = 100
    options['n_batches'] = 100
    train_and_test(leptons_with_tracks, options)
