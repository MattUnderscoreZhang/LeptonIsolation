import numpy as np
import sys, ast
import ROOT
import pickle

############
# Features #
############

lep_features = [
    ("lep_q", "int"),
    ("lep_pt", "float"),
    ("lep_eta", "float"),
    ("lep_phi", "float"),
    ("lep_m", "float"),
    ("lep_d0", "float"),
    ("lep_z0", "float"),
    ("lep_d0Err", "float"),
    ("lep_z0Err", "float"),
    ("lep_pTErr", "float"),
    ("lep_ptcone20", "float"),
    ("lep_ptcone30", "float"),
    ("lep_ptcone40", "float"),
    ("lep_topoetcone20", "float"),
    ("lep_topoetcone30", "float"),
    ("lep_topoetcone40", "float"),
    ("lep_ptvarcone20", "float"),
    ("lep_ptvarcone30", "float"),
    ("lep_ptvarcone40", "float"),
    ("lep_truthType", "int")]

track_features = [
    ("track_q", "float"),
    ("track_pt", "float"),
    ("track_eta", "float"),
    ("track_phi", "float"),
    ("track_m", "float"),
    ("track_fitQuality", "float"),
    ("track_d0", "float"),
    ("track_z0", "float"),
    ("track_d0Err", "float"),
    ("track_z0Err", "float"),
    ("track_nIBLHits", "int"),
    ("track_nPixHits", "int"),
    ("track_nPixHoles", "int"),
    ("track_nPixOutliers", "int"),
    ("track_nSCTHits", "int"),
    ("track_nTRTHits", "int")]

all_features = lep_features + track_features

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

    # associate feature names with their positions in the list
    lep_feature = {'lep_associated_tracks': 0}
    for feature in lep_features:
        lep_feature[feature[0]] = len(lep_feature)
    track_feature = {}
    for feature in track_features:
        track_feature[feature[0]] = len(track_feature)
    all_data = [[lep_feature, track_feature]]

    # save features for each lepton, and save features for associated tracks
    # each lepton has [[track container], lep_var1, lep_var2, ...]
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
                lepton_data.insert(0, associated_tracks)
                all_data.append(lepton_data)

    # add new lepton features
    isolated_types = [2, 6] # e, mu
    HF_types = [3, 7] # e, mu
    all_data[0][0]['lep_isolated'] = len(all_data[0][0])
    for i, lepton in enumerate(all_data):
        if i==0: continue
        if lepton[lep_feature['lep_truthType']] in isolated_types:
            all_data[i].append(1)
        elif lepton[lep_feature['lep_truthType']] in HF_types:
            all_data[i].append(0)
        else:
            all_data[i].append(-1) # not a recognized type

    # save features to a pickle file
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
