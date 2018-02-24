import numpy as np
import sys, ast
import ROOT
import pickle

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
        ("lep_truthAuthor", "int"),
        ("lep_truthType", "int"),
        ("lep_truthOrigin", "int")]

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

    max_dR = 0.4

    lepton_track_associations = []
    for event in input_file.tree_NoSys:
        # convert ROOT vectors into arrays
        event_features = FeaturesList()
        for (feature_name, _) in all_features:
            exec("event_feature_rootvec = event." + feature_name)
            event_feature = []
            for i in range(event_feature_rootvec.size()):
                event_feature.append(event_feature_rootvec[i])
            event_features.add(feature_name, event_feature)
        # get associated tracks for each lepton
        lep_etas = event_features.get('lep_eta')
        lep_phis = event_features.get('lep_phi')
        track_etas = event_features.get('track_eta')
        track_phis = event_features.get('track_phi')
        event_lepton_track_associations = [] # [event][lepton][track][feature]
        for lep_data in zip(lep_etas, lep_phis):
            lep_eta = lep_data[0]
            lep_phi = lep_data[1]
            associated_tracks = []
            for j, track_data in enumerate(zip(track_etas, track_phis)):
                track_eta = track_data[0]
                track_phi = track_data[1]
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
                event_lepton_track_associations.append(associated_tracks)
        if len(event_lepton_track_associations) > 0:
            lepton_track_associations.append(event_lepton_track_associations)

    # Save features to a pickle file
    with open(outFile, 'w') as f:
        pickle.dump(lepton_track_associations, f)
    
#################
# Main function #
#################

if __name__ == "__main__":
    inFile = "/afs/cern.ch/work/m/mazhang/LepIso/Ntuples/393407.root"
    outFile = "/afs/cern.ch/work/m/mazhang/LepIso/H5/393407.pkl"
    print "Converting file"
    convertFile(inFile, outFile)
    print "Finished"
