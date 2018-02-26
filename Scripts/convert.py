import sys
sys.dont_write_bytecode=True
import numpy as np
import sys, ast
import ROOT
import pickle
from features import *

############
# Features #
############

class FeaturesList(object):

    def __init__(self):
        self.features = {}

    def add(self, featureName, feature):
        self.features.setdefault(featureName, []).append(feature)

    def keys(self):
        return self.features.keys()

    def get(self, featureName):
        return self.features[featureName][0]

############################
# File reading and writing #
############################

def convertFile(inFile, outFile):

    input_file = ROOT.TFile.Open(str(inFile), "read")

    max_dR = 0.4

    all_data = []
    # [event][lepton]
    # each lepton has [lep_var1, lep_var2, ... [track container]]
    # each track has [track_var1, track_var2, ...]
    for event in input_file.tree_NoSys:
        # convert ROOT vectors into arrays
        event_features = FeaturesList()
        for (feature_name, _) in all_features:
            exec("event_feature_rootvec = event." + feature_name)
            event_feature = []
            for i in range(event_feature_rootvec.size()):
                event_feature.append(event_feature_rootvec[i])
            event_features.add(feature_name, event_feature)
        # get info for each lepton in event
        lep_etas = event_features.get('lep_eta')
        lep_phis = event_features.get('lep_phi')
        track_etas = event_features.get('track_eta')
        track_phis = event_features.get('track_phi')
        event_data = []
        for i, (lep_eta, lep_phi) in enumerate(zip(lep_etas, lep_phis)):
            lepton_data = []
            for (feature_name, _) in lep_features:
                lepton_data.append(event_features.get(feature_name)[i])
            associated_tracks = []
            # get associated tracks for each lepton
            for j, (track_eta, track_phi) in enumerate(zip(track_etas, track_phis)):
                dEta = pow(lep_eta - track_eta, 2)
                dPhi = abs(lep_phi - track_phi) % (2*np.pi)
                if dPhi > np.pi:
                    dPhi = (2*np.pi) - dPhi
                dPhi = pow(dPhi, 2)
                dR = np.sqrt(dEta*dEta + dPhi*dPhi)
                if dR <= max_dR:
                    associated_track = []
                    for (feature_name, _) in track_features:
                        associated_track.append(event_features.get(feature_name)[j])
                    associated_tracks.append(associated_track)
            if len(associated_tracks) > 0:
                lepton_data.append(associated_tracks)
                event_data.append(lepton_data)
        if len(event_data) > 0:
            all_data.append(event_data)

    # Save features to a pickle file
    with open(outFile, 'w') as f:
        # should save as dictionary with lepton info too
        pickle.dump(all_data, f)
    
#################
# Main function #
#################

if __name__ == "__main__":
    inFile = "/afs/cern.ch/work/m/mazhang/LepIso/Ntuples/393407.root"
    # inFiles = [
        # "/afs/cern.ch/work/m/mazhang/LepIso/Ntuples/393407.root",
        # "/afs/cern.ch/work/m/mazhang/LepIso/Ntuples/410472.root"
        # ]
    outFile = "/afs/cern.ch/work/m/mazhang/LepIso/Pkl/393407.pkl"
    print "Converting file"
    convertFile(inFile, outFile)
    print "Finished"
