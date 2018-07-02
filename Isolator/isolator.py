import os
import h5py as h5
import pickle
import numpy as np
import HEP
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import compare_ptcone_and_etcone

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
            leptons = np.array([i for i in leptons if ~np.isnan(i[0])]).astype(electrons.dtype)
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
        hidden = F.relu(self.hidden_layer(combined))
        output = F.relu(self.output_layer(combined))
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
            param.data.add_(-self.learning_rate, param.grad.data)
        return output, loss.data.item()

    def evaluate(self, truth, tracks):
        hidden = Variable(torch.zeros(1, self.n_hidden_neurons))
        for i in range(tracks.size()[0]):
            output, hidden = self.forward(tracks[i], hidden)
        loss = self.loss_function(output, truth)
        return output, loss.data[0]

##################
# Train and test #
##################

class LeptonTrackDataset(data.Dataset):

    def __init__(self, leptons_with_tracks):
        self.leptons_with_tracks = leptons_with_tracks

    def __getitem__(self, index):
        return self.leptons_with_tracks[index]

    def __len__(self):
        return len(self.leptons_with_tracks)

def train_and_test(leptons_with_tracks, options):

    # split train and test
    n_events = len(leptons_with_tracks)
    n_training_events = int(options['training_split'] * n_events)
    training_events = leptons_with_tracks[:n_training_events]
    test_events = leptons_with_tracks[n_training_events:]

    # prepare the generators
    train_set = LeptonTrackDataset(training_events)
    test_set = LeptonTrackDataset(test_events)
    train_loader = data.DataLoader(dataset=train_set,batch_size=options['batch_size'],sampler=torch.utils.data.sampler.RandomSampler(train_set))
    test_loader = data.DataLoader(dataset=test_set,batch_size=options['batch_size'],sampler=torch.utils.data.sampler.RandomSampler(test_set))

    # set up RNN
    options['n_track_features'] = len(training_events[0][1][1])
    rnn = RNN(options)

    # train RNN
    training_loss = 0
    training_acc = 0
    for event in training_events:
        lepton = event.pop(0)
        track_info = event # (dR, (track))
        tracks = [i[1] for i in track_info]
        truth = torch.LongTensor([(lepton['truth_type'] == 3)]) # 3 = prompt; 4 = HF
        output, loss = rnn.train(truth, torch.FloatTensor(tracks))
        print(output, loss)

        # _, top_i = output.data.topk(1)
        # category = top_i[0][0]
        # training_loss += loss
        # training_acc += (category == truth.data[0])
        # if (lep_n+1) % 100 == 0:
            # print('%d%% trained, avg loss is %.4f, avg acc is %.4f' % (lep_n / len(training_data) * 100, training_loss / (lep_n+1), training_acc / (lep_n+1)))

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
    options['learning_rate'] = 0.00005
    options['training_split'] = 0.66
    options['batch_size'] = 20
    train_and_test(leptons_with_tracks, options)
