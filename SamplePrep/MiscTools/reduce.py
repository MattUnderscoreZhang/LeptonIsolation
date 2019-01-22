import pickle
import matplotlib.pyplot as plt
import numpy as np

########################################################################
# Reduces number of background events to match number of signal events #
########################################################################

def reduce(inFile, saveFile):

    # load data and get feature index dictionaries
    data = open(inFile)
    data = pickle.load(data)
    [lep_feature_dict, track_feature_dict] = data[0]
    data.pop(0)

    # separate lepton types
    isolated_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==1]
    HF_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==0]

    # reduce number of leptons of each type so the numbers match
    n_leptons = min(len(isolated_leptons), len(HF_leptons))
    n_leptons = 10
    isolated_leptons = isolated_leptons[:n_leptons]
    HF_leptons = HF_leptons[:n_leptons]

    # save data
    data = [[lep_feature_dict, track_feature_dict], isolated_leptons + HF_leptons]
    with open(saveFile, 'w') as f:
        pickle.dump(data, f)

#################
# Main function #
#################

if __name__ == "__main__":
    inFile = "/eos/user/m/mazhang/LepIso/Pkl/410501_small.pkl"
    saveFile = "/eos/user/m/mazhang/LepIso/Pkl/410501_small_reduced.pkl"
    reduce(inFile, saveFile)
