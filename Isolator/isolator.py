import pdb
import numpy as np
import Loader.loader as loader
from Architectures.RNN import RNN
from Analysis import cones
import matplotlib.pyplot as plt
import seaborn as sns
from DataStructures.HistoryData import *
from DataStructures.LeptonTrackDataset import LeptonTrackDataset

##################
# Train and test #
##################

def train_and_test(leptons_with_tracks, options, plot_save_dir):

    # split train and test
    n_events = len(leptons_with_tracks)
    np.random.shuffle(leptons_with_tracks)
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
    history = HistoryData()

    # train RNN
    training_loss = 0
    training_acc = 0
    for batch_n in range(options['n_batches']):
        training_batch = []
        for i in range(options['batch_size']):
            next_event = next(train_set)
            training_batch.append(next_event)
        test_batch = []
        for i in range(options['batch_size']):
            next_event = next(test_set)
            test_batch.append(next_event)
        train_loss, train_acc, _, _ = rnn.do_train(training_batch)
        test_loss, test_acc, _, _ = rnn.do_eval(test_batch)
        history[LOSS][TRAIN][BATCH].append(train_loss)
        history[ACC][TRAIN][BATCH].append(train_acc)
        history[LOSS][TEST][BATCH].append(test_loss)
        history[ACC][TEST][BATCH].append(test_acc)
        print("Batch: %d, Train Loss: %0.2f, Train Acc: %0.2f, Test Loss: %0.2f, Test Acc: %0.2f" % (batch_n, train_loss, train_acc, test_loss, test_acc))

    # evaluate complete test set
    test_batch = []
    test_set.reshuffle()
    for i in range(len(test_events)):
        next_event = next(test_set)
        test_batch.append(next_event)
    _, _, test_raw_results, test_truth = rnn.do_eval(test_batch)

    # make plots
    plt.plot(history[LOSS][TRAIN][BATCH], 'o-', color='g', label="Training loss")
    plt.plot(history[LOSS][TEST][BATCH], 'o-', color='r', label="Test loss")
    plt.title("Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid('on', linestyle='--')
    plt.legend(loc='best')
    plt.savefig(plot_save_dir + "loss.png")
    plt.clf()

    plt.plot(history[ACC][TRAIN][BATCH], 'o-', color='g', label="Training accuracy")
    plt.plot(history[ACC][TEST][BATCH], 'o-', color='r', label="Test accuracy")
    plt.title("Accuracy")
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.grid('on', linestyle='--')
    plt.legend(loc='best')
    plt.savefig(plot_save_dir + "accuracy.png")
    plt.clf()

    HF_flag = [i==0 for i in test_truth]
    prompt_flag = [i==1 for i in test_truth]
    HF_raw_results = np.array(test_raw_results)[HF_flag]
    prompt_raw_results = np.array(test_raw_results)[prompt_flag]
    hist_bins = np.arange(0, 1, 0.01)
    plt.hist(prompt_raw_results, histtype='step', color='r', label="Prompt", weights=np.ones_like(prompt_raw_results)/float(len(prompt_raw_results)), bins=hist_bins)
    plt.hist(HF_raw_results, histtype='step', color='g', label="HF", weights=np.ones_like(HF_raw_results)/float(len(HF_raw_results)), bins=hist_bins)
    plt.title("RNN Results")
    plt.xlabel("Result")
    plt.ylabel("Percentage")
    plt.grid('on', linestyle='--')
    plt.legend(loc='best')
    plt.savefig(plot_save_dir + "separation.png")
    plt.clf()

#################
# Main function #
#################

if __name__ == "__main__":

    # prepare data
    in_file = "Data/output.h5"
    save_file = "Data/lepton_track_data.pkl"
    leptons_with_tracks = loader.create_or_load(in_file, save_file, overwrite=False)

    # make ptcone and etcone comparison plots
    plot_save_dir = "../Plots/"
    lwt = list(zip(leptons_with_tracks['unnormed_leptons'], leptons_with_tracks['unnormed_tracks']))
    # cones.compare_ptcone_and_etcone(lwt, plot_save_dir)

    # perform training
    options = {}
    options['n_hidden_output_neurons'] = 8
    options['n_hidden_middle_neurons'] = 8
    options['learning_rate'] = 0.01
    options['training_split'] = 0.9
    options['batch_size'] = 200
    options['n_batches'] = 5000
    lwt = list(zip(leptons_with_tracks['normed_leptons'], leptons_with_tracks['normed_tracks']))
    train_and_test(lwt, options, plot_save_dir)
