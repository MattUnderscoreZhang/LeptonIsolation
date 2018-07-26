import os
import h5py as h5
import pickle
import HEP.HEP as HEP
import numpy as np
import pdb

############################
# Group leptons and tracks #
############################

# Not stored in code (unfortunately):
# Lepton keys: ['pdgID', 'pT', 'eta', 'phi', 'd0', 'z0', 'ptcone20', 'ptcone30', 'ptcone40', 'ptvarcone20', 'ptvarcone30', 'ptvarcone40', 'truth_type']
# Track keys: ['dR', 'dEta', 'dPhi', 'dd0', 'dz0', 'charge', 'eta', 'pT', 'theta', 'd0', 'z0', 'chiSquared']

def group_leptons_and_tracks(leptons, tracks):

    grouped_leptons = []
    grouped_tracks = []

    # from https://gitlab.cern.ch/atlas/athena/blob/master/PhysicsAnalysis/MCTruthClassifier/MCTruthClassifier/MCTruthClassifierDefs.h
    HF_lep_types = [i in [3, 7] for i in leptons['truth_type']] # 2/3 (6/7) is iso/non-iso electron (muon)
    prompt_lep_types = [i in [2, 6] for i in leptons['truth_type']]
    # good_pt = leptons['pT'] < 10000
    good_HF_leptons = leptons[HF_lep_types]
    good_prompt_leptons = leptons[prompt_lep_types]
    # n_each_type = min(len(good_HF_leptons), len(good_prompt_leptons))
    # print("Event has", len(leptons), ", good:", n_each_type*2)
    # if n_each_type == 0:
        # return [], []
    # good_leptons = np.concatenate((good_HF_leptons[:n_each_type], good_prompt_leptons[:n_each_type]))
    good_leptons = np.concatenate((good_HF_leptons, good_prompt_leptons))

    # see if track passes selections listed at https://twiki.cern.ch/twiki/bin/view/AtlasProtected/Run2IsolationHarmonisation and https://twiki.cern.ch/twiki/bin/view/AtlasProtected/TrackingCPRecsEarly2018
    good_track_pt = tracks['pT'] > 500 # 500 MeV
    good_track_eta = abs(tracks['eta']) < 2.5
    good_track_hits = [i+j >= 7 for i,j in zip(tracks['nSCTHits'], tracks['nPixHits'])] and (tracks['nIBLHits'] > 0)
    good_track_holes = [i+j <= 2 for i,j in zip(tracks['nPixHoles'], tracks['nSCTHoles'])] and (tracks['nPixHoles'] <= 1)
    good_tracks = tracks[good_track_pt & good_track_eta & good_track_hits & good_track_holes]

    for lepton in good_leptons:

        nearby_tracks = []

        # find tracks within dR of lepton i
        for track in good_tracks:

            if abs((lepton['z0']-track['z0']) * np.sin(track['theta'])) > 3: continue
            # calculate and save dR
            dR = HEP.dR(lepton['phi'], lepton['eta'], track['phi'], track['eta'])   
            dEta = HEP.dEta(lepton['eta'], track['eta'])   
            dPhi = HEP.dPhi(lepton['phi'], track['phi'])
            dd0 = abs(lepton['d0']-track['d0'])
            dz0 = abs(lepton['z0']-track['z0'])
            if dR<0.5:
                nearby_tracks.append(np.array([dR, dEta, dPhi, dd0, dz0, track['charge'], track['eta'], track['pT'], track['theta'], track['d0'], track['z0'], track['chiSquared']], dtype=float))

        # sort by dR and remove tracks associated to lepton
        nearby_tracks.sort(key=lambda x: x[0])
        if len(nearby_tracks) > 0:
            nearby_tracks.pop(0)

        # add lepton and tracks to return data
        if len(nearby_tracks) > 0:
            grouped_leptons.append(np.array([i for i in lepton]))
            grouped_tracks.append(np.array(nearby_tracks, dtype=float))

    return grouped_leptons, grouped_tracks

#####################
# Save or load data #
#####################

def load(in_file, save_file_name, overwrite=False):

    # open save file if it already exists
    if os.path.exists(save_file_name) and not overwrite:
        print("File exists - loading")
        with open(save_file_name, 'rb') as out_file:
            leptons_with_tracks = pickle.load(out_file)

    # else, group leptons and tracks and save the data
    else:
        if os.path.exists(save_file_name):
            print("File exists - overwriting")
        else:
            print("Creating save file")

        # load data and get feature index dictionaries
        print("Loading data")
        data = h5.File(in_file)
        electrons = data['electrons'][()]
        muons = data['muons'][()]
        tracks = data['tracks'][()]
        n_events = electrons.shape[0]

        # group leptons with their nearby tracks
        print("Grouping leptons and tracks")
        unnormed_leptons = []
        unnormed_tracks = []
        for event_n in range(n_events):
            if event_n%1000 == 0:
                print("Event %d/%d" % (event_n, n_events))
            leptons = np.append(electrons[event_n], muons[event_n])
            leptons = np.array([i for i in leptons if ~np.isnan(i[1])])
            if len(leptons)==0: continue
            grouped_leptons, grouped_tracks = group_leptons_and_tracks(leptons, tracks[event_n])
            unnormed_leptons += grouped_leptons
            unnormed_tracks += grouped_tracks

        HF_lep_types = [i[12] in [3, 7] for i in unnormed_leptons]
        prompt_lep_types = [i[12] in [2, 6] for i in unnormed_leptons]
        good_HF_leptons = np.array(unnormed_leptons)[HF_lep_types]
        good_prompt_leptons = np.array(unnormed_leptons)[prompt_lep_types]
        n_each_type = min(len(good_HF_leptons), len(good_prompt_leptons))
        unnormed_leptons = list(good_HF_leptons)[:n_each_type] + list(good_prompt_leptons)[:n_each_type]

        # normalize and create final data structure
        unfolded_leptons = np.array(unnormed_leptons)
        unfolded_tracks = np.array([i for lep_tracks in unnormed_tracks for i in lep_tracks])
        lepton_means = unfolded_leptons.mean(axis=0)
        lepton_stds = unfolded_leptons.std(axis=0)
        track_means = unfolded_tracks.mean(axis=0)
        track_stds = unfolded_tracks.std(axis=0)
        for i in [0, 12]: # ignore pdgID and truth_type
            lepton_means[i] = 0
            lepton_stds[i] = 1
        normed_leptons = [(i-lepton_means)/lepton_stds for i in unnormed_leptons]
        normed_tracks = [[(j-track_means)/track_stds for j in i] for i in unnormed_tracks]
        leptons_with_tracks = {'unnormed_leptons': unnormed_leptons, 'normed_leptons': normed_leptons, 'unnormed_tracks': unnormed_tracks, 'normed_tracks': normed_tracks}

        # # separate prompt and HF leptons
        # isolated_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==1]
        # HF_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==0]

        with open(save_file_name, 'wb') as out_file:
            pickle.dump(leptons_with_tracks, out_file)

    return leptons_with_tracks
