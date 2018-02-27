import sys
sys.dont_write_bytecode=True
import pickle
import features
import matplotlib.pyplot as plt
import numpy as np

###################
# Comparison code #
###################

def compareFeatures(inFile, saveDir):

    # associate feature names with their positions in the list
    lep_feature = {}
    for feature in features.lep_features:
        lep_feature[feature[0]] = len(lep_feature)
    track_feature = {}
    for feature in features.track_features:
        track_feature[feature[0]] = len(track_feature)

    # load data and separate lepton types
    data = open(inFile)
    data = pickle.load(data)

    isolated_leptons = [lepton for lepton in data if lepton[lep_feature['lep_truthType']] in features.isolated_types]
    HF_leptons = [lepton for lepton in data if lepton[lep_feature['lep_truthType']] in features.HF_types]

    # plot comparisons for all lepton variables
    for feature, _ in features.lep_features:
        isolated_feature_values = [lepton[lep_feature[feature]] for lepton in isolated_leptons]
        HF_feature_values = [lepton[lep_feature[feature]] for lepton in HF_leptons]
        data = isolated_feature_values + HF_feature_values
        bins = np.linspace(min(data), max(data), 30)
        plt.hist([isolated_feature_values, HF_feature_values], normed=True, bins=bins, histtype='step')
        plt.title(feature)
        plt.legend(['isolated', 'HF'])
        plt.savefig(saveDir + feature + ".png", bbox_inches='tight')

#################
# Main function #
#################

if __name__ == "__main__":
    inFile = "/afs/cern.ch/work/m/mazhang/LepIso/Pkl/393407.pkl"
    saveDir = "/afs/cern.ch/user/m/mazhang/Projects/LepIso/LeptonIsolation/Plots/"
    compareFeatures(inFile, saveDir)
