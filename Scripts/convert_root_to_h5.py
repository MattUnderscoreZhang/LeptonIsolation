from __future__ import division
import numpy as np
import sys, ast
import ROOT
import pickle

############
# Features #
############

lep_features = [
    ('TruthParticlesAux.pdgId', 'int'),
    ('TruthParticlesAuxDyn.d0', 'float'),
    ('TruthParticlesAuxDyn.z0', 'float'),
    ('TruthParticlesAuxDyn.phi', 'float'),
    ('TruthParticlesAuxDyn.theta', 'float'),
    ('TruthParticlesAuxDyn.qOverP', 'float')]

track_features = [
    ('InDetTrackParticlesAux.d0', 'float'),
    ('InDetTrackParticlesAux.z0', 'float'),
    ('InDetTrackParticlesAux.phi', 'float'),
    ('InDetTrackParticlesAux.theta', 'float'),
    ('InDetTrackParticlesAux.qOverP', 'float'),
    ('InDetTrackParticlesAux.chiSquared', 'float'),
    ('InDetTrackParticlesAuxDyn.truthOrigin', 'int'),
    ('InDetTrackParticlesAuxDyn.truthType', 'int'),
    ('InDetTrackParticlesAuxDyn.truthParticleLink', 'int')]

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

# needs its own function because exec() has weird rules
def getFeature(event, feature_name):

    exec("event_feature_rootvec = event." + feature_name)
    return event_feature_rootvec

def convertFile(inFile, outFile, outFileReduced):

    input_file = ROOT.TFile.Open(str(inFile), "read")

    max_dR = 0.4

    # associate feature names with their positions in the list
    lep_feature_dict = {'lepIso_lep_associated_tracks': 0}
    for feature in lep_features:
        lep_feature_dict[feature[0]] = len(lep_feature_dict)
    track_feature_dict = {}
    for feature in track_features:
        track_feature_dict[feature[0]] = len(track_feature_dict)
    all_data = [[lep_feature_dict, track_feature_dict]]

    # save features for each lepton, and save features for associated tracks
    # each lepton has [[track container], lep_var1, lep_var2, ...]
    # each track has [track_var1, track_var2, ...]
    # also calculate and store lepton-track dR
    nEvents = input_file.tree_NoSys.GetEntries()
    print "Associating leptons to tracks in", nEvents, "events"
    eventN = 0
    track_feature_dict['lepIso_track_lep_dR'] = len(track_feature_dict)
    for event in input_file.tree_NoSys:
        eventN += 1
        if (eventN%100==0): print eventN, "out of", nEvents, "events"
        # convert ROOT vectors into arrays
        event_features = FeaturesList()
        for (feature_name, _) in all_features:
            event_feature_rootvec = getFeature(event, feature_name)
            event_feature = []
            for i in range(event_feature_rootvec.size()):
                event_feature.append(event_feature_rootvec[i])
            event_features.add(feature_name, event_feature)
        # get info for each lepton in event
        lep_etas = event_features.get('lepIso_lep_eta')
        lep_phis = event_features.get('lepIso_lep_phi')
        track_etas = event_features.get('lepIso_track_eta')
        track_phis = event_features.get('lepIso_track_phi')
        for i, (lep_eta, lep_phi) in enumerate(zip(lep_etas, lep_phis)):
            lepton_data = []
            for (feature_name, _) in lep_features:
                lepton_data.append(event_features.get(feature_name)[i])
            associated_tracks = []
            # get associated tracks for each lepton
            for j, (track_eta, track_phi) in enumerate(zip(track_etas, track_phis)):
                dEta = lep_eta - track_eta
                dPhi = abs(lep_phi - track_phi) % (2*np.pi)
                if dPhi > np.pi:
                    dPhi = (2*np.pi) - dPhi
                dR = np.sqrt(dEta*dEta + dPhi*dPhi)
                if dR <= max_dR:
                    associated_track = []
                    for (feature_name, _) in track_features:
                        associated_track.append(event_features.get(feature_name)[j])
                    associated_track.append(dR)
                    associated_tracks.append(associated_track)
            if len(associated_tracks) > 0:
                lepton_data.insert(0, associated_tracks)
                all_data.append(lepton_data)

    # calculate lepton isolation
    print "Classifying lepton isolation"
    isolated_types = [2, 6] # e, mu
    HF_types = [3, 7] # e, mu
    lep_feature_dict['lepIso_lep_isolated'] = len(lep_feature_dict)
    for i, lepton in enumerate(all_data):
        if i==0: continue
        if lepton[lep_feature_dict['lepIso_lep_truthType']] in isolated_types:
            all_data[i].append(1)
        elif lepton[lep_feature_dict['lepIso_lep_truthType']] in HF_types:
            all_data[i].append(0)
        else:
            all_data[i].append(-1) # not a recognized type

    print "Calculating ptcone variables"
    # calculate ptconeX, ptvarconeX, and topoetconeX, where X is 20, 30, 40
    # also calculate new ptcone features
    lep_feature_dict['lepIso_lep_calculated_ptcone20'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_calculated_ptcone30'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_calculated_ptcone40'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_calculated_ptvarcone20'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_calculated_ptvarcone30'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_calculated_ptvarcone40'] = len(lep_feature_dict)
    # lep_feature_dict['lepIso_lep_calculated_topoetcone20'] = len(lep_feature_dict)
    # lep_feature_dict['lepIso_lep_calculated_topoetcone30'] = len(lep_feature_dict)
    # lep_feature_dict['lepIso_lep_calculated_topoetcone40'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptcone20_squared'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptcone30_squared'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptcone40_squared'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptvarcone20_squared'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptvarcone30_squared'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptvarcone40_squared'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptcone20_dR_weighted'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptcone30_dR_weighted'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptcone40_dR_weighted'] = len(lep_feature_dict)

    for i, lepton in enumerate(all_data):
        if i==0: continue
        if i%100==0: print i, "out of", len(all_data)-1, "leptons"
        associated_tracks = lepton[lep_feature_dict['lepIso_lep_associated_tracks']]
        lep_pt = lepton[lep_feature_dict['lepIso_lep_pt']]
        ptcone20 = 0
        ptcone30 = 0
        ptcone40 = 0
        ptvarcone20 = 0
        ptvarcone30 = 0
        ptvarcone40 = 0
        # topoetcone20 = 0
        # topoetcone30 = 0
        # topoetcone40 = 0
        ptcone20_squared = 0
        ptcone30_squared = 0
        ptcone40_squared = 0
        ptvarcone20_squared = 0
        ptvarcone30_squared = 0
        ptvarcone40_squared = 0
        ptcone20_dR_weighted = 0
        ptcone30_dR_weighted = 0
        ptcone40_dR_weighted = 0

        # sorted(associated_tracks, key=lambda l:l[track_feature_dict['lepIso_track_lep_dR']])
        for j, track in enumerate(associated_tracks):

            # if j==0:
                # continue # skip track closest to lepton (its own track)

            # track selection criteria already applied in TrackObject.cxx in SusySkimMaker
            # the stuff below is if I want to calculate these things myself

                # if track[track_feature_dict['lepIso_track_pt']] < 1: continue
                # float eta = track[track_feature_dict['lepIso_track_eta']];
                # float theta = arctan(exp(-eta)) * 2;
                # if abs(track[track_feature_dict['lepIso_track_z0']] * sin(theta)) > 3 : continue
                # # Loose track critera from https://twiki.cern.ch/twiki/bin/view/AtlasProtected/TrackingCPRecsEarly2018
                # if track[track_feature_dict['lepIso_track_pt']] < 0.5: continue
                # if abs(track[track_feature_dict['lepIso_track_eta']]) > 2.5: continue
                # if track[track_feature_dict['lepIso_track_nSCTHits']] + track[track_feature_dict['lepIso_track_nPixHits']] < 7: continue
                # if track[track_feature_dict['lepIso_track_nSharedPixHits']] + track[track_feature_dict['lepIso_track_nSharedSCTHits']]/2 > 1: continue
                # if track[track_feature_dict['lepIso_track_nPixHoles']] + track[track_feature_dict['lepIso_track_nSCTHoles']] > 2: continue
                # if track[track_feature_dict['lepIso_track_nPixHoles']] > 1: continue

            dR = track[track_feature_dict['lepIso_track_lep_dR']]
            track_pt = track[track_feature_dict['lepIso_track_pt']]
            if dR <= 0.2:
                ptcone20 += track_pt
                ptcone20_squared += track_pt * track_pt
                ptcone20_dR_weighted += track_pt * 0.2 / (dR + 0.01)
            if dR <= 0.3:
                ptcone30 += track_pt
                ptcone30_squared += track_pt * track_pt
                ptcone30_dR_weighted += track_pt * 0.2 / (dR + 0.01)
            if dR <= 0.4:
                ptcone40 += track_pt
                ptcone40_squared += track_pt * track_pt
                ptcone40_dR_weighted += track_pt * 0.2 / (dR + 0.01)
            if dR <= 10 / lep_pt:
                if dR <= 0.2:
                    ptvarcone20 += track_pt
                    ptvarcone20_squared += track_pt * track_pt
                if dR <= 0.3:
                    ptvarcone30 += track_pt
                    ptvarcone30_squared += track_pt * track_pt
                if dR <= 0.4:
                    ptvarcone40 += track_pt
                    ptvarcone40_squared += track_pt * track_pt

        all_data[i].append(ptcone20)
        all_data[i].append(ptcone30)
        all_data[i].append(ptcone40)
        all_data[i].append(ptvarcone20)
        all_data[i].append(ptvarcone30)
        all_data[i].append(ptvarcone40)
        # all_data[i].append(topoetcone20)
        # all_data[i].append(topoetcone30)
        # all_data[i].append(topoetcone40)
        all_data[i].append(ptcone20_squared)
        all_data[i].append(ptcone30_squared)
        all_data[i].append(ptcone40_squared)
        all_data[i].append(ptvarcone20_squared)
        all_data[i].append(ptvarcone30_squared)
        all_data[i].append(ptvarcone40_squared)
        all_data[i].append(ptcone20_dR_weighted)
        all_data[i].append(ptcone30_dR_weighted)
        all_data[i].append(ptcone40_dR_weighted)

    # save features to a pickle file
    print "Saving events"
    with open(outFile, 'w') as f:
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
    with open(outFileReduced, 'w') as f:
        # should save as dictionary with lepton info too
        pickle.dump(all_data, f)
    
#################
# Main function #
#################

if __name__ == "__main__":
    # inFiles = [
        # "/afs/cern.ch/work/m/mazhang/LepIso/Ntuples/393407.root",
        # "/afs/cern.ch/work/m/mazhang/LepIso/Ntuples/410472.root"
        # ]
    inFile = "/Users/mattzhang/Dropbox/Projects/Data/LepIso/IDTIDE/410000.root"
    outFile = "/Users/mattzhang/Dropbox/Projects/Data/LepIso/IDTIDE/410000.h5"
    outFileReduced = "/Users/mattzhang/Dropbox/Projects/Data/LepIso/IDTIDE/410000_small.h5"
    print "Converting file"
    convertFile(inFile, outFile, outFileReduced)
    print "Finished"
