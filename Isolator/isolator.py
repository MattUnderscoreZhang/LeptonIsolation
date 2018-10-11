import numpy as np
import pathlib
import datetime
import Loader.loader as loader
from Architectures.RNN import RNN
from torch.utils.data import DataLoader
from Analysis import FeatureComparer
import matplotlib.pyplot as plt
import seaborn as sns
from DataStructures.HistoryData import *
from DataStructures.LeptonTrackDataset import Torchdata, collate
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
import pdb

###############
# Tensorboard #
###############

writer = SummaryWriter()

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

    def prepare(self):
        # split train and test
        np.random.shuffle(self.leptons_with_tracks)
        self.training_events = \
            self.leptons_with_tracks[:self.n_training_events]
        self.test_events = self.leptons_with_tracks[self.n_training_events:]

        # prepare the generators
        self.train_set = Torchdata(self.training_events)
        self.test_set = Torchdata(self.test_events)

        # set up RNN
        self.rnn = RNN(self.options)

    def make_batch(self):

        training_loader = DataLoader(
            self.train_set, batch_size=options['batch_size'],
            collate_fn=collate, shuffle=True, drop_last=True)

        testing_loader = DataLoader(
            self.test_set, batch_size=options['batch_size'],
            collate_fn=collate, shuffle=True, drop_last=True)

        return training_loader, testing_loader

    def train(self):
        train_loss = 0
        train_acc = 0
        for batch_n in range(self.options['n_batches']):
            training_batch, testing_batch = self.make_batch()
            train_loss, train_acc, _, _ = self.rnn.do_train(training_batch)
            test_loss, test_acc, _, _ = self.rnn.do_eval(testing_batch)
            self.history[LOSS][TRAIN][BATCH].append(train_loss)
            self.history[ACC][TRAIN][BATCH].append(train_acc)
            self.history[LOSS][TEST][BATCH].append(test_loss)
            self.history[ACC][TEST][BATCH].append(test_acc)
            writer.add_scalar('Accuracy/Train Accuracy', train_acc, batch_n)
            writer.add_scalar('Accuracy/Test Accuracy', test_acc, batch_n)
            writer.add_scalar('Loss/Train Loss', train_loss, batch_n)
            writer.add_scalar('Loss/Test Loss', test_loss, batch_n)
            print("Batch: %d, Train Loss: %0.4f, Train Acc: %0.4f, "
                  "Test Loss: %0.4f, Test Acc: %0.4f" % (
                      batch_n, train_loss, train_acc, test_loss, test_acc))

    def test(self):
        test_batch = []
        self.test_set.file.reshuffle()
        testing_loader = DataLoader(
            self.test_set, batch_size=options['batch_size'],
            collate_fn=collate, shuffle=True)
        _, _, self.test_raw_results, self.test_truth = self.rnn.do_eval(
            testing_loader)

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

        # # separation
        # self.test_truth = [i[0] for i in self.test_truth]
        # HF_flag = [i == 0 for i in self.test_truth]
        # prompt_flag = [i == 1 for i in self.test_truth]
        # HF_raw_results = np.array(self.test_raw_results)[HF_flag]
        # prompt_raw_results = np.array(self.test_raw_results)[prompt_flag]
        # hist_bins = np.arange(0, 1, 0.01)
        # pdb.set_trace()
        # plt.hist(prompt_raw_results, histtype='step', color='r',
             # label="Prompt", weights=np.ones_like(prompt_raw_results) /
             # float(len(prompt_raw_results)), bins=hist_bins)
        # plt.hist(HF_raw_results, histtype='step', color='g', label="HF",
             # weights=np.ones_like(HF_raw_results) /
             # float(len(HF_raw_results)), bins=hist_bins)
        # plt.title("RNN Results")
        # plt.xlabel("Result")
        # plt.ylabel("Percentage")
        # plt.grid('on', linestyle='--')
        # plt.legend(loc='best')
        # plt.savefig(self.plot_save_dir + "separation.png")
        # plt.clf()

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
    from Options.default_options import options

    # prepare data
    in_file = "../Data/output.h5"
    save_file = "../Data/lepton_track_data.pkl"
    leptons_with_tracks = loader.create_or_load(
        in_file, save_file, overwrite=False, pseudodata=False)
    options['lepton_size'] = len(leptons_with_tracks['lepton_labels'])
    options['track_size'] = len(leptons_with_tracks['track_labels'])

    # make ptcone and etcone comparison plots - normed
    # plot_save_dir = "../Plots_normed/"
    # pathlib.Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
    # lwt = list(zip(
    #     leptons_with_tracks['normed_leptons'],
    #     leptons_with_tracks['normed_tracks']))
    # labels = [leptons_with_tracks['lepton_labels'],
    #     leptons_with_tracks['track_labels']]
    # FeatureComparer.compare_ptcone_and_etcone(lwt, labels, plot_save_dir)

    # make ptcone and etcone comparison plots - unnormed
    # plot_save_dir = "../Plots_unnormed/"
    # pathlib.Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
    # lwt = list(zip(
    #     leptons_with_tracks['unnormed_leptons'],
    #     leptons_with_tracks['unnormed_tracks']))
    # labels = [leptons_with_tracks['lepton_labels'],
    #     leptons_with_tracks['track_labels']]
    # FeatureComparer.compare_ptcone_and_etcone(lwt, labels, plot_save_dir)

    # perform training

    plot_save_dir = "../Plots/"
    lwt = list(
        zip(leptons_with_tracks['normed_leptons'],
            leptons_with_tracks['normed_tracks']))
    RNN_trainer = RNN_Trainer(options, lwt, plot_save_dir)
    RNN_trainer.train_and_test()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
