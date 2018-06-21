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
    [lep_feature_dict, track_feature_dict] = data[0]
    data.pop(0)

    # separate lepton types
    isolated_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==1]
    HF_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==0]

    # plot comparisons for all lepton features
    for feature, index in lep_feature_dict.items():
        if feature == 'lepIso_lep_associated_tracks':
            continue
        isolated_feature_values = [lepton[index] for lepton in isolated_leptons]
        HF_feature_values = [lepton[index] for lepton in HF_leptons]
        all_feature_values = isolated_feature_values + HF_feature_values
        bins = np.linspace(min(all_feature_values), max(all_feature_values), 30)
        plt.hist([isolated_feature_values, HF_feature_values], normed=True, bins=bins, histtype='step')
        plt.title(feature)
        plt.legend(['lepIso_isolated', 'HF'])
        plt.savefig(saveDir + feature + ".png", bbox_inches='tight')
        plt.clf()

    # plot comparisons for calculated and stored ptcone features
    ptcone_features = [
        ('lepIso_lep_ptcone20', 'lepIso_lep_calculated_ptcone20'),
        ('lepIso_lep_ptcone30', 'lepIso_lep_calculated_ptcone30'),
        ('lepIso_lep_ptcone40', 'lepIso_lep_calculated_ptcone40'),
        # ('lepIso_lep_topoetcone20', 'lepIso_lep_calculated_topoetcone20'),
        # ('lepIso_lep_topoetcone30', 'lepIso_lep_calculated_topoetcone30'),
        # ('lepIso_lep_topoetcone40', 'lepIso_lep_calculated_topoetcone40'),
        ('lepIso_lep_ptvarcone20', 'lepIso_lep_calculated_ptvarcone20'),
        ('lepIso_lep_ptvarcone30', 'lepIso_lep_calculated_ptvarcone30'),
        ('lepIso_lep_ptvarcone40', 'lepIso_lep_calculated_ptvarcone40')]
    for feature, calc_feature in ptcone_features:
        lepton_feature_values = [lepton[lep_feature_dict[feature]] for lepton in data]
        lepton_calc_feature_values = [lepton[lep_feature_dict[calc_feature]] for lepton in data]
        plt.scatter(lepton_feature_values, lepton_calc_feature_values, s=1)
        # heatmap, xedges, yedges = np.histogram2d(lepton_feature_values, lepton_calc_feature_values, bins=500)
        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        # plt.clf()
        # plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.title(feature + ' vs. ' + calc_feature)
        plt.xlabel(feature)
        plt.ylabel(calc_feature)
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.savefig(saveDir + feature + "_vs_" + calc_feature + ".png", bbox_inches='tight')
        plt.clf()

#################
# Main function #
#################

if __name__ == "__main__":
    inFile = "/eos/user/m/mazhang/LepIso/Pkl/410501_small.pkl"
    saveDir = "/afs/cern.ch/user/m/mazhang/Projects/LepIso/LeptonIsolation/Plots/410501_small/"
    compareFeatures(inFile, saveDir)
