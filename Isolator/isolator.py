import numpy as np
import pathlib
import torch
import datetime
import Loader.loader as loader
from Architectures.RNN import RNN
# from Architectures.test_new_rnn import RNN
# from Analysis import cones
import matplotlib.pyplot as plt
# import seaborn as sns
from DataStructures.HistoryData import *
from DataStructures.LeptonTrackDataset import LeptonTrackDataset

##################
# Train and test #
##################


class RNN_Trainer:

    def __init__(self, options, leptons_with_tracks, plot_save_dir):
        self.options = options
        self.n_events = len(leptons_with_tracks)
        self.n_training_events = int(
            self.options['training_split'] * self.n_events)
        self.leptons_with_tracks = leptons_with_tracks
        self.options['n_track_features'] = len(
            self.leptons_with_tracks[0][1][0])
        self.plot_save_dir = plot_save_dir
        # training results e.g. history[CLASS_LOSS][TRAIN][EPOCH]
        self.history = HistoryData()
        self.test_truth = []
        self.test_raw_results = []

    def arch_print(self):

        print(vars(self)['options'])

    def prepare(self):
        # split train and test
        np.random.shuffle(self.leptons_with_tracks)
        self.training_events = \
            self.leptons_with_tracks[:self.n_training_events]
        self.test_events = self.leptons_with_tracks[self.n_training_events:]
        # prepare the generators
        self.train_set = LeptonTrackDataset(self.training_events)
        self.test_set = LeptonTrackDataset(self.test_events)
        # set up RNN
        self.rnn = RNN(self.options)

    def make_batch(self):
        training_batch = []
        for i in range(self.options['batch_size']):
            next_event = next(self.train_set)
            training_batch.append(next_event)
        test_batch = []
        for i in range(self.options['batch_size']):
            next_event = next(self.test_set)
            test_batch.append(next_event)
        return training_batch, test_batch

    def train(self):
        train_loss = 0
        train_acc = 0
        for batch_n in range(self.options['n_batches']):
            training_batch, test_batch = self.make_batch()
            train_loss, train_acc, _, _ = self.rnn.do_train(training_batch)
            test_loss, test_acc, _, _ = self.rnn.do_eval(test_batch)
            self.history[LOSS][TRAIN][BATCH].append(train_loss)
            self.history[ACC][TRAIN][BATCH].append(train_acc)
            self.history[LOSS][TEST][BATCH].append(test_loss)
            self.history[ACC][TEST][BATCH].append(test_acc)
            print("Batch: %d, Train Loss: %0.2f, Train Acc: %0.2f,\
             Test Loss: %0.2f, Test Acc: %0.2f" % (
                batch_n, train_loss, train_acc, test_loss, test_acc))

    def test(self):
        test_batch = []
        self.test_set.reshuffle()
        for i in range(len(self.test_events)):
            next_event = next(self.test_set)
            test_batch.append(next_event)
        _, _, self.test_raw_results, self.test_truth = self.rnn.do_eval(
            test_batch)

    def plot(self):
        '''Plots all the necessary details from the trained model'''

        # loss
        plt.plot(self.history[LOSS][TRAIN][BATCH],
                 'o-', color='g', label="Training loss")
        plt.plot(self.history[LOSS][TEST][BATCH],
                 'o-', color='r', label="Test loss")
        plt.title("Loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.grid('on', linestyle='--')
        plt.legend(loc='best')
        plt.savefig(self.plot_save_dir + "loss.png")
        plt.clf()

        # accuracy
        plt.plot(self.history[ACC][TRAIN][BATCH], 'o-',
                 color='g', label="Training accuracy")
        plt.plot(self.history[ACC][TEST][BATCH], 'o-',
                 color='r', label="Test accuracy")
        plt.title("Accuracy")
        plt.xlabel("Batch")
        plt.ylabel("Accuracy")
        plt.grid('on', linestyle='--')
        plt.legend(loc='best')
        plt.savefig(self.plot_save_dir + "accuracy.png")
        plt.clf()

        # separation
        HF_flag = [i == 0 for i in self.test_truth]
        prompt_flag = [i == 1 for i in self.test_truth]
        HF_raw_results = np.array(self.test_raw_results)[HF_flag]
        prompt_raw_results = np.array(self.test_raw_results)[prompt_flag]
        hist_bins = np.arange(0, 1, 0.01)
        plt.hist(prompt_raw_results, histtype='step', color='r',
                 label="Prompt", weights=np.ones_like(prompt_raw_results) /
                 float(len(prompt_raw_results)), bins=hist_bins)
        plt.hist(HF_raw_results, histtype='step', color='g', label="HF",
                 weights=np.ones_like(HF_raw_results) /
                 float(len(HF_raw_results)), bins=hist_bins)
        plt.title("RNN Results")
        plt.xlabel("Result")
        plt.ylabel("Percentage")
        plt.grid('on', linestyle='--')
        plt.legend(loc='best')
        plt.savefig(self.plot_save_dir + "separation.png")
        plt.clf()

    def train_and_test(self):
        self.prepare()
        self.train()
        self.test()
        self.plot()

#################
# Main function #
#################


if __name__ == "__main__":

    # set options
    # from Options.default_options import options
    options = {}
    options['n_hidden_output_neurons'] = 8
    options['n_hidden_middle_neurons'] = 8
    options['learning_rate'] = 0.01
    options['training_split'] = 0.9
    options['batch_size'] = 200
    options['n_batches'] = 50
    options['n_layers'] = 8
    options["n_size"] = [8, 8, 8]
    # prepare data
    in_file = "Data/output.h5"
    save_file = "Data/lepton_track_data.pkl"
    leptons_with_tracks = loader.create_or_load(
        in_file, save_file, overwrite=False)

    # make ptcone and etcone comparison plots
    plot_save_dir = "../Plots/"
    pathlib.Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
    lwt = list(zip(
        leptons_with_tracks['unnormed_leptons'],
        leptons_with_tracks['unnormed_tracks']))
    # cones.compare_ptcone_and_etcone(lwt, plot_save_dir)


    # perform training
    lwt = list(
        zip(leptons_with_tracks['normed_leptons'],
            leptons_with_tracks['normed_tracks']))
    RNN_trainer = RNN_Trainer(options, lwt, plot_save_dir)
    # RNN_trainer.arch_print()
    RNN_trainer.train_and_test()
