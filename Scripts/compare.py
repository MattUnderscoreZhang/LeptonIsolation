import pickle
import matplotlib.pyplot as plt
import numpy as np

###################
# Comparison code #
###################

def compareFeatures(inFile, saveDir):

    # load data and get feature index dictionaries
    data = open(inFile)
    data = pickle.load(data)
    [lep_feature, track_feature] = data[0]
    data.pop(0)

    # separate lepton types
    isolated_leptons = [lepton for lepton in data if lepton[lep_feature['lep_isolated']]==1]
    HF_leptons = [lepton for lepton in data if lepton[lep_feature['lep_isolated']]==0]

    # plot comparisons for all lepton variables
    for feature, index in lep_feature.items():
        if feature == 'lep_associated_tracks':
            continue
        isolated_feature_values = [lepton[index] for lepton in isolated_leptons]
        HF_feature_values = [lepton[index] for lepton in HF_leptons]
        data = isolated_feature_values + HF_feature_values
        bins = np.linspace(min(data), max(data), 30)
        plt.hist([isolated_feature_values, HF_feature_values], normed=True, bins=bins, histtype='step')
        plt.title(feature)
        plt.legend(['isolated', 'HF'])
        plt.savefig(saveDir + feature + ".png", bbox_inches='tight')
        plt.clf()

#################
# Main function #
#################

if __name__ == "__main__":
    inFile = "/afs/cern.ch/work/m/mazhang/LepIso/Pkl/393407.pkl"
    saveDir = "/afs/cern.ch/user/m/mazhang/Projects/LepIso/LeptonIsolation/Plots/"
    compareFeatures(inFile, saveDir)
