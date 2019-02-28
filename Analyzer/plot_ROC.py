import matplotlib.pyplot as plt
from sklearn import metrics
import pickle as pkl
import numpy as np


def plot_history(history, plot_save_dir):
    '''Plots all the necessary details from the trained model'''
    # # loss
    # plt.plot(history[LOSS][TRAIN][BATCH],
             # 'o-', color='g', label="Training loss")
    # plt.plot(history[LOSS][TEST][BATCH],
             # 'o-', color='r', label="Test loss")
    # plt.title("Loss")
    # plt.xlabel("Batch")
    # plt.ylabel("Loss")
    # plt.grid('on', linestyle='--')
    # plt.legend(loc='best')
    # plt.savefig(plot_save_dir + "loss.png")
    # plt.clf()

    # # accuracy
    # plt.plot(history[ACC][TRAIN][BATCH], 'o-',
             # color='g', label="Training accuracy")
    # plt.plot(history[ACC][TEST][BATCH], 'o-',
             # color='r', label="Test accuracy")
    # plt.title("Accuracy")
    # plt.xlabel("Batch")
    # plt.ylabel("Accuracy")
    # plt.grid('on', linestyle='--')
    # plt.legend(loc='best')
    # plt.savefig(plot_save_dir + "accuracy.png")
    # plt.clf()

    # # separation
    # test_truth = [i[0] for i in test_truth]
    # HF_flag = [i == 0 for i in test_truth]
    # prompt_flag = [i == 1 for i in test_truth]
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
    # plt.savefig(plot_save_dir + "separation.png")
    # plt.clf()


def plot_ROC(data_filename, test_raw_results, test_truth):

    # open file
    with open(data_filename, 'rb') as data_file:
        leptons_with_tracks = pkl.load(data_file, encoding='latin1')

    # extract ptcone info
    leptons = leptons_with_tracks['unnormed_leptons']
    lepton_keys = leptons_with_tracks['lepton_labels']
    isolated = [int(lepton[lepton_keys.index('truth_type')] in [2, 6])
                for lepton in leptons]
    cones = {}
    pt_keys = ['ptcone20', 'ptcone30', 'ptcone40',
               'ptvarcone20', 'ptvarcone30', 'ptvarcone40']
    for key in pt_keys:
        cones[key] = [lepton[lepton_keys.index(key)] for lepton in leptons]
        max_key = max(cones[key])
        min_key = min(cones[key])
        range_key = max_key - min_key
        cones[key] = [(i - min_key) / range_key for i in cones[key]]

    # get rid of events with ptcone=0
    good_leptons = [lepton[lepton_keys.index(
        'ptcone20')] > 0 for lepton in leptons]
    leptons = np.array(leptons)[good_leptons]
    isolated = np.array(isolated)[good_leptons]
    for key in pt_keys:
        cones[key] = np.array(cones[key])[good_leptons]

    # make ROC comparison plots
    for key in pt_keys:
        fpr, tpr, thresholds = metrics.roc_curve(isolated, cones[key])
        # roc_auc = metrics.auc(fpr, tpr)
        plt.plot(tpr, fpr, lw=2, label=key)

    fpr, tpr, thresholds = metrics.roc_curve(test_truth, test_raw_results)
    # roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label='RNN')

    # plot style
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid('on', linestyle='--')
    plt.title('ROC Curves for Classification')
    plt.legend(loc="lower right")
    # plt.show()
    plot_save_dir = "Plots/"
    plt.savefig(plot_save_dir + "compare_ROC.png")
    plt.clf()
