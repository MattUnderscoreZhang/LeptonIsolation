import os
import h5py as h5
import pickle
import HEP.HEP as HEP
import numpy as np
import pdb

############################
# Group leptons and tracks #
############################

def group_leptons_and_tracks(leptons, tracks):

    grouped_leptons = []
    grouped_tracks = []

    for lepton in leptons:

        if lepton['truth_type'] not in [2, 3]: continue
        nearby_tracks = []

        # find tracks within dR of lepton i
        for track in tracks:
            # see if track passes selections listed at https://twiki.cern.ch/twiki/bin/view/AtlasProtected/Run2IsolationHarmonisation
            if track['pT'] < 1000: continue
            if abs(track['z0SinTheta']) > 30: continue
            if abs(track['eta']) > 2.5: continue
            # calculate and save dR
            dR = HEP.dR(lepton['phi'], lepton['eta'], track['phi'], track['eta'])   
            dEta = HEP.dEta(lepton['eta'], track['eta'])   
            dPhi = HEP.dPhi(lepton['phi'], track['phi'])
            dd0 = abs(lepton['d0']-track['d0'])
            dz0 = abs(lepton['z0']-track['z0'])
            if dR<0.4:
                nearby_tracks.append(np.array([dR, dEta, dPhi, dd0, dz0, track['charge'], track['eta'], track['pT'], track['z0SinTheta'], track['d0'], track['z0'], track['chiSquared']], dtype=float))

        # sort by dR and remove track closest to lepton
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
        electrons = data['electrons']
        muons = data['muons']
        tracks = data['tracks']
        n_events = electrons.shape[0]

        # group leptons with their nearby tracks
        print("Grouping leptons and tracks")
        unnormed_leptons = []
        unnormed_tracks = []
        for event_n in range(n_events):
            if event_n%10 == 0:
                print("Event %d/%d" % (event_n, n_events))
            leptons = np.append(electrons[event_n], muons[event_n])
            leptons = np.array([i for i in leptons if ~np.isnan(i[0])])
            grouped_leptons, grouped_tracks = group_leptons_and_tracks(leptons, tracks[event_n])
            unnormed_leptons += grouped_leptons
            unnormed_tracks += grouped_tracks

        # normalize and create final data structure
        unfolded_leptons = np.array(unnormed_leptons)
        unfolded_tracks = np.array([i for lep_tracks in unnormed_tracks for i in lep_tracks])
        lepton_means = unfolded_leptons.mean(axis=0)
        lepton_stds = unfolded_leptons.std(axis=0)
        track_means = unfolded_tracks.mean(axis=0)
        track_stds = unfolded_tracks.std(axis=0)
        leptons = [(i-lepton_means)/lepton_stds for i in unnormed_leptons]
        tracks = [[(j-track_means)/track_stds for j in i] for i in unnormed_tracks]
        leptons_with_tracks = [[lepton, track] for lepton, track in zip(leptons, tracks)]

        # # separate prompt and HF leptons
        # isolated_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==1]
        # HF_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==0]

        with open(save_file_name, 'wb') as out_file:
            pickle.dump(leptons_with_tracks, out_file)

    return leptons_with_tracks
