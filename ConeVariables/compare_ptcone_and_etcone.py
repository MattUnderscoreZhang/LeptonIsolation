import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import HEP
import pdb

############################
# Group leptons and tracks #
############################

# make a list of [lepton, (dR, track), (dR, track), ...] for each lepton
def group_leptons_and_tracks(leptons, tracks):
    leptons_with_tracks = []
    for lepton in leptons:
        # find tracks within dR of lepton i
        leptons_with_tracks_i = []
        for track in tracks:
            dR = HEP.dR(lepton['phi'], lepton['eta'], track['phi'], track['eta'])   
            if dR<0.4:
                leptons_with_tracks_i.append((dR, track))
        # sort by dR and remove track closest to lepton
        leptons_with_tracks_i.sort(key=lambda x: x[0])
        if len(leptons_with_tracks_i) > 0:
            leptons_with_tracks_i.pop(0)
        if len(leptons_with_tracks_i) > 0:
            leptons_with_tracks_i.insert(0, lepton)
            leptons_with_tracks.append(leptons_with_tracks_i)
        # add lepton info
    return leptons_with_tracks

###################
# Calculate cones #
###################

def calculate_ptcone_and_etcone(leptons_with_tracks_i):

    max_dR = 0.4
    lepton = leptons_with_tracks_i.pop(0)
    tracks = leptons_with_tracks_i

    cones = {}
    cones['truth_ptcone20'] = lepton['ptcone20']
    cones['truth_ptcone30'] = lepton['ptcone30']
    cones['truth_ptcone40'] = lepton['ptcone40']
    cones['truth_ptvarcone20'] = lepton['ptvarcone20']
    cones['truth_ptvarcone30'] = lepton['ptvarcone30']
    cones['truth_ptvarcone40'] = lepton['ptvarcone40']
    cones['ptcone20'] = 0
    cones['ptcone30'] = 0
    cones['ptcone40'] = 0
    cones['ptvarcone20'] = 0
    cones['ptvarcone30'] = 0
    cones['ptvarcone40'] = 0

    lep_pt = lepton[1]
    for (dR, track) in tracks:
        track_pt = track[0] # pt - couldn't figure out how not to hard-code
        if dR <= 0.2:
            cones['ptcone20'] += track_pt
            # ptcone20_squared += track_pt * track_pt
            # ptcone20_dR_weighted += track_pt * 0.2 / (dR + 0.01)
        if dR <= 0.3:
            cones['ptcone30'] += track_pt
            # ptcone30_squared += track_pt * track_pt
            # ptcone30_dR_weighted += track_pt * 0.2 / (dR + 0.01)
        if dR <= 0.4:
            cones['ptcone40'] += track_pt
            # ptcone40_squared += track_pt * track_pt
            # ptcone40_dR_weighted += track_pt * 0.2 / (dR + 0.01)
        if dR <= 10 / lep_pt:
            if dR <= 0.2:
                cones['ptvarcone20'] += track_pt
            if dR <= 0.3:
                cones['ptvarcone30'] += track_pt
            if dR <= 0.4:
                cones['ptvarcone40'] += track_pt

    return cones

###################
# Comparison code #
###################

def compareFeatures(inFile, saveDir):

    # load data and get feature index dictionaries
    print("Loading data")
    data = h5.File(inFile)
    electrons = data['electrons']
    muons = data['muons']
    tracks = data['tracks']
    n_events = electrons.shape[0]

    # # separate prompt and HF leptons
    # isolated_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==1]
    # HF_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==0]

    # group leptons with their nearby tracks
    print("Grouping leptons and tracks")
    leptons_with_tracks = []
    # for event_n in range(n_events):
    for event_n in range(100):
        if event_n%10 == 0:
            print("Event %d/%d" % (event_n, n_events))
        leptons = np.append(electrons[event_n], muons[event_n])
        leptons = np.array([i for i in leptons if ~np.isnan(i[0])]).astype(electrons.dtype)
        leptons_with_tracks += group_leptons_and_tracks(leptons, tracks[event_n])

    # calculate ptcone
    print("Calculating ptcone variables")
    cones = {}
    for i, leptons_with_tracks_i in enumerate(leptons_with_tracks):
        if i%100 == 0:
            print("Lepton %d/%d" % (i, len(leptons_with_tracks)))
        cones_i = calculate_ptcone_and_etcone(leptons_with_tracks_i)
        for key in cones_i.keys():
            cones.setdefault(key, []).append(cones_i[key])

    # plot comparisons for calculated and stored ptcone features
    print("Producing plots")
    cone_features = ['ptcone20', 'ptcone30', 'ptcone40', 'ptvarcone20', 'ptvarcone30', 'ptvarcone40']
    for feature in cone_features:
        plt.scatter(cones[feature], cones["truth_" + feature], s=1)
        # heatmap, xedges, yedges = np.histogram2d(lepton_feature_values, lepton_calc_feature_values, bins=500)
        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        # plt.clf()
        # plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.title(feature)
        plt.xlabel(feature)
        plt.ylabel("Truth " + feature)
        # plt.xlim(0, 50)
        # plt.ylim(0, 50)
        plt.savefig(saveDir + feature + ".png", bbox_inches='tight')
        plt.clf()

    # # plot comparisons for all lepton features
    # for feature, index in lep_feature_dict.items():
        # if feature == 'lepIso_lep_leptons_with_tracks':
            # continue
        # isolated_feature_values = [lepton[index] for lepton in isolated_leptons]
        # HF_feature_values = [lepton[index] for lepton in HF_leptons]
        # all_feature_values = isolated_feature_values + HF_feature_values
        # bins = np.linspace(min(all_feature_values), max(all_feature_values), 30)
        # plt.hist([isolated_feature_values, HF_feature_values], normed=True, bins=bins, histtype='step')
        # plt.title(feature)
        # plt.legend(['lepIso_isolated', 'HF'])
        # plt.savefig(saveDir + feature + ".png", bbox_inches='tight')
        # plt.clf()

#################
# Main function #
#################

if __name__ == "__main__":
    inFile = "output.h5"
    saveDir = "./"
    compareFeatures(inFile, saveDir)
