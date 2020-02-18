import matplotlib

matplotlib.use("Agg")  # NOQA
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from ROOT import TFile
from collections import namedtuple


class ROCPlot(object):
    """docstring for ROCPlot"""

    def __init__(self, options, test_raw_results, test_truth, test_lep_pT):
        super(ROCPlot, self).__init__()
        self.options = options
        self.test_truth = np.array(test_truth)
        self.test_lep_pT = test_lep_pT
        self.test_raw_results = np.array(test_raw_results)
        (
            self.truth_type,
            self.slicing_pT,
            self.isolated,
            self.baselines,
        ) = self._get_data()
        self._remove_nans()
        self._normalize()
        self.fig = plt.figure()
        plt.clf()
        self.ROC_Fig = namedtuple("ROC_Fig", ["label", "image"])
        self.figs = []

    def _normalize(self):
        for key in self.options["baseline_features"]:
            max_key = max(self.baselines[key])
            min_key = min(self.baselines[key])
            range_key = max_key - min_key
            self.baselines[key] = [
                1 - ((i - min_key) / range_key) for i in self.baselines[key]
            ]  # larger value = less isolated

    def _get_data(self):
        # open file
        data_filename = self.options["input_data"]
        data_file = TFile(data_filename)  # keep this open to prevent segfault
        data_tree = getattr(data_file, self.options["tree_name"])
        truth_type = []
        slicing_pT = []
        baselines = {}
        for i in range(data_tree.GetEntries()):
            data_tree.GetEntry(i)
            truth_type.append(data_tree.truth_type)
            slicing_pT.append(data_tree.ROC_slicing_lep_pT)
        isolated = [int(int(i) in [2, 6]) for i in truth_type]

        for key in self.options["baseline_features"]:
            baseline_array = []
            for i in range(data_tree.GetEntries()):
                data_tree.GetEntry(i)
                baseline_array.append(getattr(data_tree, key))
            baselines[key] = baseline_array

        return truth_type, slicing_pT, np.array(isolated), baselines

    def _remove_nans(self):
        good_features = []
        for key in self.options["baseline_features"]:
            if not np.isnan(self.baselines[key]).any():
                good_features.append(key)
        self.options["baseline_features"] = good_features

    def ComparisionPlots(self):
        # make ROC comparison plots
        for key in self.options["baseline_features"]:
            fpr, tpr, thresholds = metrics.roc_curve(self.isolated, self.baselines[key])
            plt.plot(fpr, tpr, lw=2, label=key)

        fpr, tpr, thresholds = metrics.roc_curve(self.test_truth, self.test_raw_results)
        plt.plot(fpr, tpr, lw=2, label="RNN")

        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid("on", linestyle="--")
        plt.title("ROC Curves for Classification")
        plt.legend(loc=7)

        self.figs.append(self.ROC_Fig("ROC", self.fig))

    def pTsplitPlots(self):
        # split ROC curves by lepton pT
        lep_pT_boundaries = np.array([0, 10, 15, 20, 1000]) * 1000

        for i in range(len(lep_pT_boundaries) - 1):
            fig = plt.figure()
            plt.clf()

            low_pT = lep_pT_boundaries[i]
            high_pT = lep_pT_boundaries[i + 1]

            pT_slice = np.array([i > low_pT and i < high_pT for i in self.slicing_pT])
            isolated_pT_slice = self.isolated[pT_slice]
            for key in self.options["baseline_features"]:
                baselines_pT_slice = np.array(self.baselines[key])[pT_slice]
                fpr, tpr, thresholds = metrics.roc_curve(
                    isolated_pT_slice, baselines_pT_slice
                )
                plt.plot(fpr, tpr, lw=2, label=key)

            test_pT_slice = np.array(
                [i > low_pT and i < high_pT for i in self.test_lep_pT]
            )
            test_truth_pT_slice = self.test_truth[test_pT_slice]
            test_raw_results_pT_slice = self.test_raw_results[test_pT_slice]
            fpr, tpr, thresholds = metrics.roc_curve(
                test_truth_pT_slice, test_raw_results_pT_slice
            )
            plt.plot(fpr, tpr, lw=2, label="RNN")

            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.grid("on", linestyle="--")
            plt.title("ROC Curves for Classification")
            plt.legend(loc="lower right")

            self.figs.append(
                self.ROC_Fig(
                    "ROC slice - pT "
                    + str(int(low_pT / 1000))
                    + " to "
                    + str(int(high_pT / 1000)),
                    fig,
                )
            )

    def run(self):
        self.ComparisionPlots()
        self.pTsplitPlots()
        return self.figs
