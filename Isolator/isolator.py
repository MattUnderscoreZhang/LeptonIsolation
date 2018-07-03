import pdb
import torch
import random
import itertools as it
from Loader.loader import load
from Architectures.RNN import RNN
from Analysis.cones import compare_ptcone_and_etcone

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
        tracks = torch.from_numpy(event[1:]).float()
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
    options['n_track_features'] = len(training_events[0][1])
    rnn = RNN(options)

    # train RNN
    training_loss = 0
    training_acc = 0
    training_batch = []
    for batch_n in range(options['n_batches']):
        for i in range(options['batch_size']):
            next_event = next(train_set)
            truth = torch.LongTensor([(int(next_event[0][11]) == 3)]) # 'truth_type' - 3 = prompt; 4 = HF
            training_batch.append([truth, next_event])
        output, loss = rnn.train(training_batch)
        print(output, loss)
        _, top_i = output.data.topk(1)
        category = top_i[0][0]
        training_loss += loss
        training_acc += (category == truth.data[0])
        # if (batch_n+1) % 100 == 0:
            # print('%d%% batches trained, loss is %.4f, acc is %.4f' % batch_n, training_loss, training_acc)

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
    options['n_hidden_neurons'] = 1024
    options['learning_rate'] = 0.0000005
    options['training_split'] = 0.66
    options['batch_size'] = 20
    options['n_batches'] = 500
    train_and_test(leptons_with_tracks, options)
