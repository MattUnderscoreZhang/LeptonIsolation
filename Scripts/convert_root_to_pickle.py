# Converting ROOT files to pickle files
# We use pickle because we need each lepton to be associated to a variable number of tracks, and this is not suitable for the H5 format.

from __future__ import division
import numpy as np
import sys, ast
import ROOT
import pickle
import pdb

############
# Features #
############

# dictionary with improved data updating functionality
class FeaturesList(object):

    def __init__(self):
        self.features = {}

    def add(self, featureName, feature):
        self.features.setdefault(featureName, []).append(feature)

    def keys(self):
        return self.features.keys()

    def get(self, featureName):
        return self.features[featureName][0]

# needs its own function because exec() has weird rules
def getFeatureFromROOT(event, feature_name):

    exec("event_feature_rootvec = event." + feature_name)
    return event_feature_rootvec

############################
# File reading and writing #
############################

def convertFile(in_file_name, tree_name, lep_features, track_features, out_file_name):

    input_file = ROOT.TFile.Open(str(in_file_name), "read")
    exec("input_tree = input_file." + tree_name)
    nEvents = input_tree.GetEntries()
    print "Associating leptons to tracks in", nEvents, "events"

    # dictionaries associating {feature_name: feature_index} e.g. {'TruthParticlesAux.pdgId': 1}
    lep_feature_dict = {'lepIso_lep_associated_tracks': 0}
    for feature, _ in lep_features:
        lep_feature_dict[feature] = len(lep_feature_dict)
    track_feature_dict = {}
    for feature, _ in track_features:
        track_feature_dict[feature] = len(track_feature_dict)
    all_features = lep_features + track_features
    all_data = [[lep_feature_dict, track_feature_dict]]
    # also calculate and store lepton-track dR
    track_feature_dict['lep_track_dR'] = len(track_feature_dict)

    # save features for each lepton, and save features for associated tracks
    # each lepton has [[track container], lep_var1, lep_var2, ...]
    # each track has [track_var1, track_var2, ...]
    for eventN, event in enumerate(input_tree):
        if (eventN % 100 == 0): print eventN, "out of", nEvents, "events"
        # store features from ROOT event in dictionary
        event_features = FeaturesList()
        for (feature_name, _) in all_features:
            event_feature_rootvec = getFeatureFromROOT(event, feature_name)
            event_feature = []
            for event_feature_i in event_feature_rootvec.d0:
                event_feature.append(event_feature_i)
            event_features.add(feature_name, event_feature)
        pdb.set_trace()
        # get info for each lepton in event
        lep_etas = event_features.get('lepIso_lep_eta')
        lep_phis = event_features.get('lepIso_lep_phi')
        track_etas = event_features.get('lepIso_track_eta')
        track_phis = event_features.get('lepIso_track_phi')
        for lep_i, (lep_eta, lep_phi) in enumerate(zip(lep_etas, lep_phis)):
            lepton_data = []
            for (feature_name, _) in lep_features:
                lepton_data.append(event_features.get(feature_name)[lep_i])
            associated_tracks = []
            # get associated tracks for each lepton
            for track_i, (track_eta, track_phi) in enumerate(zip(track_etas, track_phis)):
                dEta = lep_eta - track_eta
                dPhi = abs(lep_phi - track_phi) % (2*np.pi)
                if dPhi > np.pi:
                    dPhi = (2*np.pi) - dPhi
                dR = np.sqrt(dEta*dEta + dPhi*dPhi)
                if dR <= max_dR:
                    associated_track = []
                    for (feature_name, _) in track_features:
                        associated_track.append(event_features.get(feature_name)[track_i])
                    associated_track.append(dR)
                    associated_tracks.append(associated_track)
            if len(associated_tracks) > 0:
                lepton_data.insert(0, associated_tracks)
                all_data.append(lepton_data)

    # save features to a pickle file
    print "Saving events"
    with open(out_file_name, 'w') as f:
        # should save as dictionary with lepton info too
        pickle.dump(all_data, f)

    # reduce number of leptons of each type so the numbers match
    print "Reducing number of events"
    all_data.pop(0)
    isolated_leptons = [lepton for lepton in all_data if lepton[lep_feature_dict['lepIso_lep_isolated']]==1]
    HF_leptons = [lepton for lepton in all_data if lepton[lep_feature_dict['lepIso_lep_isolated']]==0]
    n_leptons = min(len(isolated_leptons), len(HF_leptons))
    isolated_leptons = isolated_leptons[:n_leptons]
    HF_leptons = HF_leptons[:n_leptons]

    # save reduced features to a pickle file
    print "Saving reduced events"
    all_data = [lep_feature_dict, track_feature_dict] + isolated_leptons + HF_leptons
    with open(out_file_name + "_reduced", 'w') as f:
        # should save as dictionary with lepton info too
        pickle.dump(all_data, f)
    
#################
# Main function #
#################

if __name__ == "__main__":

    lep_features = [
        ('TruthParticlesAux', 'int')]
        # ('TruthParticlesAux.pdgId', 'int'),
        # ('TruthParticlesAuxDyn.d0', 'float'),
        # ('TruthParticlesAuxDyn.z0', 'float'),
        # ('TruthParticlesAuxDyn.phi', 'float'),
        # ('TruthParticlesAuxDyn.theta', 'float'),
        # ('TruthParticlesAuxDyn.qOverP', 'float')]

    track_features = [
        ('InDetTrackParticlesAux', 'float')]
        # ('InDetTrackParticlesAux.d0', 'float'),
        # ('InDetTrackParticlesAux.z0', 'float'),
        # ('InDetTrackParticlesAux.phi', 'float'),
        # ('InDetTrackParticlesAux.theta', 'float'),
        # ('InDetTrackParticlesAux.qOverP', 'float'),
        # ('InDetTrackParticlesAux.chiSquared', 'float'),
        # ('InDetTrackParticlesAuxDyn.truthOrigin', 'int'),
        # ('InDetTrackParticlesAuxDyn.truthType', 'int'),
        # ('InDetTrackParticlesAuxDyn.truthParticleLink', 'int')]

    in_file_name = "/eos/user/m/mazhang/LepIso/MC/IDTIDE/mc16_13TeV.410000.PowhegPythiaEvtGen_P2012_ttbar_hdamp172p5_nonallhad.recon.DAOD_IDTRKVALID.e3698_s2995_r9639/DAOD_IDTRKVALID.11500650._000001.pool.root.1"
    tree_name = "CollectionTree"
    out_file_name = "410000.pkl"

    print "Converting file"
    convertFile(in_file_name, tree_name, lep_features, track_features, out_file_name)
    print "Finished"
