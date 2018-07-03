import os
import h5py as h5
import pickle
import numpy as np
import HEP
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import compare_ptcone_and_etcone
import random
import itertools as it

#######################
# Load and group data #
#######################

# group leptons and tracks
# make a list of [lepton, track, track, ...] for each lepton
def group_leptons_and_tracks(leptons, tracks):
    leptons_with_tracks = []
    for lepton in leptons:
        if lepton['truth_type'] not in [2, 3]: continue
        # find tracks within dR of lepton i
        leptons_with_tracks_i = []
        for track in tracks:
            # see if track passes selections listed at https://twiki.cern.ch/twiki/bin/view/AtlasProtected/Run2IsolationHarmonisation
            if track['pT'] < 1000: continue
            if abs(track['z0SinTheta']) > 30: continue
            if abs(track['eta']) > 2.5: continue
            # calculate and save dR
            dR = HEP.dR(lepton['phi'], lepton['eta'], track['phi'], track['eta'])   
            dEta = HEP.dEta(lepton['eta'], track['eta'])   
            dPhi = HEP.dPhi(lepton['phi'], track['phi'])
            dd0 = abs(lepton['d0']-track['d0'])
            dz0 = abs(lepton['z0']-track['z0'])
            if dR<0.4:
                leptons_with_tracks_i.append(np.array([dR, dEta, dPhi, dd0, dz0, track['charge'], track['eta'], track['pT'], track['z0SinTheta'], track['d0'], track['z0'], track['chiSquared']], dtype=float))
        # sort by dR and remove track closest to lepton
        leptons_with_tracks_i.sort(key=lambda x: x[0])
        if len(leptons_with_tracks_i) > 0:
            leptons_with_tracks_i.pop(0)
        if len(leptons_with_tracks_i) > 0:
            leptons_with_tracks_i.insert(0, np.array([i for i in lepton]))
            leptons_with_tracks.append(np.array(leptons_with_tracks_i, dtype=float))
        # add lepton info
    return leptons_with_tracks

# load and group data
def prepare_data(in_file, save_file_name, overwrite=False):

    # open save file if it already exists
    if os.path.exists(save_file_name) and not overwrite:
        print("File exists - loading")
        with open(save_file_name, 'rb') as out_file:
            leptons_with_tracks = pickle.load(out_file)

    # else, group leptons and tracks and save the data
    else:
        if os.path.exists(save_file):
            print("File exists - overwriting")
        else:
            print("Creating save file")

        # load data and get feature index dictionaries
        print("Loading data")
        data = h5.File(in_file)
        electrons = data['electrons']
        muons = data['muons']
        tracks = data['tracks']
        n_events = electrons.shape[0]

        # group leptons with their nearby tracks
        print("Grouping leptons and tracks")
        leptons_with_tracks = []
        for event_n in range(n_events):
            if event_n%10 == 0:
                print("Event %d/%d" % (event_n, n_events))
            leptons = np.append(electrons[event_n], muons[event_n])
            leptons = np.array([i for i in leptons if ~np.isnan(i[0])])
            leptons_with_tracks += group_leptons_and_tracks(leptons, tracks[event_n])

        # # separate prompt and HF leptons
        # isolated_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==1]
        # HF_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==0]

        with open(save_file_name, 'wb') as out_file:
            pickle.dump(leptons_with_tracks, out_file)

    return leptons_with_tracks

################
# Architecture #
################

class RNN(nn.Module):

    def __init__(self, options):
        super().__init__()
        self.n_hidden_neurons = options['n_hidden_neurons']
        self.learning_rate = options['learning_rate']
        self.hidden_layer = nn.Linear(options['n_track_features'] + self.n_hidden_neurons, self.n_hidden_neurons)
        self.output_layer = nn.Linear(options['n_track_features'] + self.n_hidden_neurons, 2)
        self.softmax = nn.Softmax(dim=1)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input_values, hidden_values):
        input_values = input_values.view(1, input_values.size()[0])
        combined = torch.cat((input_values, hidden_values), 1)
        hidden = self.hidden_layer(combined)
        output = F.relu(self.output_layer(combined))
        output = self.softmax(output)
        return output, hidden

    def train(self, events):
        self.zero_grad()
        total_loss = 0
        for event in events:
            hidden = torch.zeros(1, self.n_hidden_neurons)
            truth, lep_tracks = event
            lepton, tracks = lep_tracks
            for track in tracks:
                output, hidden = self.forward(track, hidden)
            loss = self.loss_function(output, truth)
            total_loss += loss
        total_loss.backward()
        # Add parameters' gradients to their values, multiplied by learning rate
        for param in self.parameters():
            param.data.add_(-self.learning_rate, param.grad.data)
        return output, total_loss.data.item()

    def evaluate(self, truth, tracks):
        hidden = Variable(torch.zeros(1, self.n_hidden_neurons))
        for i in range(tracks.size()[0]):
            output, hidden = self.forward(tracks[i], hidden)
        loss = self.loss_function(output, truth)
        return output, loss.data[0]

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

    # test_loss = 0
    # test_acc = 0
    # for lep_n in range(len(test_data)):
        # truth = test_data[lep_n][lep_feature_dict['lepIso_lep_isolated']]
        # truth = Variable(torch.LongTensor([truth]))
        # tracks = test_data[lep_n][0]
        # output, loss = rnn.evaluate(truth, Variable(torch.FloatTensor(tracks)))
        # _, top_i = output.data.topk(1)
        # category = top_i[0][0]
        # test_loss += loss
        # test_acc += (category == truth.data[0])
    # print('Test loss is %.4f, test acc is %.4f' % (test_loss / (lep_n+1), test_acc / (lep_n+1)))

#################
# Main function #
#################

if __name__ == "__main__":

    # prepare data
    in_file = "Data/output.h5"
    save_file = "Data/lepton_track_data.pkl"
    leptons_with_tracks = prepare_data(in_file, save_file, overwrite=False)

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
