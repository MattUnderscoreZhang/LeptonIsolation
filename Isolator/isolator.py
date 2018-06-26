import h5py as h5
import numpy as np
import HEP
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import compare_ptcone_and_etcone

################
# Architecture #
################

class RNN(nn.Module):

    def __init__(self, n_track_features, n_hidden_neurons):
        super().__init__()
        self.n_hidden_neurons = n_hidden_neurons
        self.hidden_layer = nn.Linear(n_track_features + n_hidden_neurons, n_hidden_neurons)
        self.output_layer = nn.Linear(n_track_features + n_hidden_neurons, 2)
        self.softmax = nn.Softmax(dim=1)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input_values, hidden_values):
        input_values = input_values.view(1, input_values.size()[0])
        combined = torch.cat((input_values, hidden_values), 1)
        hidden = self.hidden_layer(combined)
        output = self.output_layer(combined)
        output = self.softmax(output)
        return output, hidden

    def train(self, truth, tracks):
        hidden = Variable(torch.zeros(1, self.n_hidden_neurons))
        self.zero_grad()
        for i in range(tracks.size()[0]):
            output, hidden = self.forward(tracks[i], hidden)
        loss = self.loss_function(output, truth)
        loss.backward()
        # Add parameters' gradients to their values, multiplied by learning rate
        for param in self.parameters():
            param.data.add_(-learning_rate, param.grad.data)
        return output, loss.data[0]

    def evaluate(self, truth, tracks):
        hidden = Variable(torch.zeros(1, self.n_hidden_neurons))
        for i in range(tracks.size()[0]):
            output, hidden = self.forward(tracks[i], hidden)
        loss = self.loss_function(output, truth)
        return output, loss.data[0]

##################
# Train and test #
##################

def train_and_test(data, training_split):

    data = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']] != -1] # skip leptons with unrecognized truth

    n_events = len(data)
    n_training_events = int(training_split * n_events)
    training_data = data[:n_training_events]
    test_data = data[n_training_events:]

    rnn = RNN(n_track_features, n_hidden_neurons)
    training_loss = 0
    training_acc = 0
    for lep_n in range(len(training_data)):
        truth = training_data[lep_n][lep_feature_dict['lepIso_lep_isolated']]
        truth = Variable(torch.LongTensor([truth]))
        tracks = training_data[lep_n][0]
        output, loss = rnn.train(truth, Variable(torch.FloatTensor(tracks)))
        _, top_i = output.data.topk(1)
        category = top_i[0][0]
        training_loss += loss
        training_acc += (category == truth.data[0])
        if (lep_n+1) % 100 == 0:
            print('%d%% trained, avg loss is %.4f, avg acc is %.4f' % (lep_n / len(training_data) * 100, training_loss / (lep_n+1), training_acc / (lep_n+1)))

    test_loss = 0
    test_acc = 0
    for lep_n in range(len(test_data)):
        truth = test_data[lep_n][lep_feature_dict['lepIso_lep_isolated']]
        truth = Variable(torch.LongTensor([truth]))
        tracks = test_data[lep_n][0]
        output, loss = rnn.evaluate(truth, Variable(torch.FloatTensor(tracks)))
        _, top_i = output.data.topk(1)
        category = top_i[0][0]
        test_loss += loss
        test_acc += (category == truth.data[0])
    print('Test loss is %.4f, test acc is %.4f' % (test_loss / (lep_n+1), test_acc / (lep_n+1)))

#######################
# Load and group data #
#######################

# group leptons and tracks
# make a list of [lepton, (dR, track), (dR, track), ...] for each lepton
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
            if dR<0.4:
                leptons_with_tracks_i.append((dR, track))
        # sort by dR and remove track closest to lepton
        leptons_with_tracks_i.sort(key=lambda x: x[0])
        if len(leptons_with_tracks_i) > 0:
            leptons_with_tracks_i.pop(0)
        if len(leptons_with_tracks_i) > 0:
            leptons_with_tracks_i.insert(0, lepton)
            leptons_with_tracks.append(leptons_with_tracks_i)
        # add lepton info
    return leptons_with_tracks

# calculate cones
def calculate_ptcone_and_etcone(leptons_with_tracks_i):

    max_dR = 0.4
    lepton = leptons_with_tracks_i.pop(0)
    tracks = leptons_with_tracks_i

    cones = {}
    cones['truth_ptcone20'] = lepton['ptcone20']
    cones['truth_ptcone30'] = lepton['ptcone30']
    cones['truth_ptcone40'] = lepton['ptcone40']
    cones['truth_ptvarcone20'] = lepton['ptvarcone20']
    cones['truth_ptvarcone30'] = lepton['ptvarcone30']
    cones['truth_ptvarcone40'] = lepton['ptvarcone40']
    cones['ptcone20'] = 0
    cones['ptcone30'] = 0
    cones['ptcone40'] = 0
    cones['ptvarcone20'] = 0
    cones['ptvarcone30'] = 0
    cones['ptvarcone40'] = 0

    lep_pt = lepton[1]
    for (dR, track) in tracks:
        track_pt = track[0] # pt - couldn't figure out how not to hard-code
        if dR <= 0.2:
            cones['ptcone20'] += track_pt
            # ptcone20_squared += track_pt * track_pt
            # ptcone20_dR_weighted += track_pt * 0.2 / (dR + 0.01)
        if dR <= 0.3:
            cones['ptcone30'] += track_pt
            # ptcone30_squared += track_pt * track_pt
            # ptcone30_dR_weighted += track_pt * 0.2 / (dR + 0.01)
        if dR <= 0.4:
            cones['ptcone40'] += track_pt
            # ptcone40_squared += track_pt * track_pt
            # ptcone40_dR_weighted += track_pt * 0.2 / (dR + 0.01)
        if dR <= 10 / lep_pt:
            if dR <= 0.2:
                cones['ptvarcone20'] += track_pt
            if dR <= 0.3:
                cones['ptvarcone30'] += track_pt
            if dR <= 0.4:
                cones['ptvarcone40'] += track_pt

    return cones

# load and group data
def prepare_data(in_file, plot_save_dir):

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
    # for event_n in range(n_events):
    for event_n in range(50):
        if event_n%10 == 0:
            print("Event %d/%d" % (event_n, n_events))
        leptons = np.append(electrons[event_n], muons[event_n])
        leptons = np.array([i for i in leptons if ~np.isnan(i[0])]).astype(electrons.dtype)
        leptons_with_tracks += group_leptons_and_tracks(leptons, tracks[event_n])

    # # separate prompt and HF leptons
    # isolated_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==1]
    # HF_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==0]

    # calculate ptcone
    print("Calculating ptcone variables")
    cones = {}
    for i, leptons_with_tracks_i in enumerate(leptons_with_tracks):
        cones_i = calculate_ptcone_and_etcone(leptons_with_tracks_i)
        for key in cones_i.keys():
            cones.setdefault(key, []).append(cones_i[key])

    compare_ptcone_and_etcone.compareFeatures(cones, plot_save_dir)

#################
# Main function #
#################

if __name__ == "__main__":

    in_file = "output.h5"
    plot_save_dir = "../Plots/"
    prepare_data(in_file, plot_save_dir)

    n_hidden_neurons = 128
    learning_rate = 0.005
    training_split = 0.66

    train_and_test(data, training_split)
