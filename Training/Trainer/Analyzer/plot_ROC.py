import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle as pkl
import numpy as np


def plot_ROC(data_filename, test_raw_results, test_truth):

    # pdb.set_trace()
    fig = plt.figure()
    plt.clf()
    # open file
    with open(data_filename, 'rb') as data_file:
        leptons_with_tracks = pkl.load(data_file, encoding='latin1')

    # extract ptcone info
    leptons = leptons_with_tracks['unnormed_leptons']
    lepton_keys = leptons_with_tracks['lepton_labels']
    isolated = [int(lepton[lepton_keys.index('truth_type')] in [2, 6])
                for lepton in leptons]
    baselines = {}
    baseline_keys = ['ptcone20', 'ptcone30', 'ptcone40',
                     'ptvarcone20', 'ptvarcone30', 'ptvarcone40', 'PLT']
    for key in baseline_keys:
        baselines[key] = [lepton[lepton_keys.index(key)] for lepton in leptons]
        max_key = max(baselines[key])
        min_key = min(baselines[key])
        range_key = max_key - min_key
        baselines[key] = [1 - ((i - min_key) / range_key) for i in baselines[key]]  # larger value = less isolated

    # get rid of events with ptcone=0
    good_leptons = [lepton[lepton_keys.index('ptcone20')] > 0 for lepton in leptons]
    leptons = np.array(leptons)[good_leptons]
    isolated = np.array(isolated)[good_leptons]
    for key in baseline_keys:
        baselines[key] = np.array(baselines[key])[good_leptons]

    # make ROC comparison plots
    for key in baseline_keys:
        fpr, tpr, thresholds = metrics.roc_curve(isolated, baselines[key])
        # roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=key)

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

    return fig
