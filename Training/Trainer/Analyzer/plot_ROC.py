import matplotlib
matplotlib.use("Agg")  # NOQA
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle as pkl
import numpy as np
from ROOT import TFile


def plot_ROC(options, test_raw_results, test_truth):

    fig = plt.figure()
    plt.clf()
    # open file
    data_filename = options["input_data"]
    with open(data_filename, "rb") as data_file:
        leptons_with_tracks = pkl.load(data_file, encoding="latin1")

    # extract ptcone info
    leptons = leptons_with_tracks["unnormed_leptons"]
    lepton_keys = leptons_with_tracks["lepton_labels"]
    isolated = [
        int(lepton[lepton_keys.index("truth_type")] in [2, 6]) for lepton in leptons
    ]
    baselines = {}
    for key in options["baseline_features"]:
        baselines[key] = [lepton[lepton_keys.index(key)] for lepton in leptons]
        max_key = max(baselines[key])
        min_key = min(baselines[key])
        range_key = max_key - min_key
        baselines[key] = [
            1 - ((i - min_key) / range_key) for i in baselines[key]
        ]  # larger value = less isolated

    # get rid of events with ptcone=0
    good_leptons = [lepton[lepton_keys.index("ptcone20")] > 0 for lepton in leptons]
    leptons = np.array(leptons)[good_leptons]
    isolated = np.array(isolated)[good_leptons]
    for key in options["baselines"]:
        baselines[key] = np.array(baselines[key])[good_leptons]

    # make ROC comparison plots
    for key in options["baselines"]:
        fpr, tpr, thresholds = metrics.roc_curve(isolated, baselines[key])
        # roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=key)

    fpr, tpr, thresholds = metrics.roc_curve(test_truth, test_raw_results)
    # roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label="RNN")

    # plot style
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid("on", linestyle="--")
    plt.title("ROC Curves for Classification")
    plt.legend(loc="lower right")

    return fig


def plot_ROOT_ROC(options, test_raw_results, test_truth):

    fig = plt.figure()
    plt.clf()

    # open file
    data_filename = options["input_data"]
    data_file = TFile(data_filename)  # keep this open to prevent segfault
    data_tree = getattr(data_file, options["tree_name"])

    # get truth info for all events
    truth_type = []
    for i in range(data_tree.GetEntries()):
        data_tree.GetEntry(i)
        truth_type.append(data_tree.truth_type)
    isolated = [int(int(i) in [2, 6]) for i in truth_type]

    # get baseline features for all events
    baselines = {}
    for key in options["baseline_features"]:
        baseline_array = []
        for i in range(data_tree.GetEntries()):
            data_tree.GetEntry(i)
            baseline_array.append(getattr(data_tree, key))
        baselines[key] = baseline_array

    # normalize baseline features to be between 0 and 1
    for key in options["baseline_features"]:
        max_key = max(baselines[key])
        min_key = min(baselines[key])
        range_key = max_key - min_key
        baselines[key] = [1 - ((i - min_key) / range_key) for i in baselines[key]]  # larger value = less isolated

    # # get rid of events with ptcone=0
    # good_leptons = [lepton[lepton_keys.index("ptcone20")] > 0 for lepton in leptons]
    # leptons = np.array(leptons)[good_leptons]
    # isolated = np.array(isolated)[good_leptons]
    # for key in options["baselines"]:
        # baselines[key] = np.array(baselines[key])[good_leptons]

    # make ROC comparison plots
    for key in options["baseline_features"]:
        fpr, tpr, thresholds = metrics.roc_curve(isolated, baselines[key])
        # roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=key)

    # make ROC curve for RNN on test data
    fpr, tpr, thresholds = metrics.roc_curve(test_truth, test_raw_results)
    # roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label="RNN")

    # add plot style and labels
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid("on", linestyle="--")
    plt.title("ROC Curves for Classification")
    plt.legend(loc="lower right")

    return fig
