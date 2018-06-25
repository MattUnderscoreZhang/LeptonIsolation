import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import HEP
import pdb

############################
# Group leptons and tracks #
############################

# make a list of [(dR, track)] for each lepton
def group_leptons_and_tracks(leptons, tracks):
    associated_tracks = []
    for i, lepton in enumerate(leptons):
        if i%1 == 0:
            print("%d/%d" % (i, len(leptons)))
        associated_tracks_i = []
        for track in tracks:
            dR = HEP.dR(lepton['phi'], lepton['eta'], track[3], track[2])   
            if dR<0.4:
                associated_tracks_i.append((dR, track))
        # sort by dR and remove track closest to lepton
        associated_tracks_i.sort(key=lambda x: x[0])
        if len(associated_tracks_i) > 0:
            associated_tracks_i.pop(0)
        associated_tracks.append(associated_tracks_i)
    return associated_tracks

###################
# Calculate cones #
###################

def calculate_ptcone_and_etcone(lepton, associated_tracks):

    max_dR = 0.4
    lep_pt = lepton[1]

    ptcone20 = 0
    ptcone30 = 0
    ptcone40 = 0
    ptvarcone20 = 0
    ptvarcone30 = 0
    ptvarcone40 = 0

    for j, (dR, track) in enumerate(associated_tracks):

        track_pt = track[1]
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
            if dR <= 0.3:
                ptvarcone30 += track_pt
            if dR <= 0.4:
                ptvarcone40 += track_pt

    cones = []
    cones.append(ptcone20)
    cones.append(ptcone30)
    cones.append(ptcone40)
    cones.append(ptvarcone20)
    cones.append(ptvarcone30)
    cones.append(ptvarcone40)

    return cones

###################
# Comparison code #
###################

def compareFeatures(inFile, saveDir):

    # load data and get feature index dictionaries
    print("Loading data")
    data = h5.File(inFile)
    leptons = np.append(data['electrons'].value, data['muons'].value)[:10]
    tracks = data['tracks']

    # # separate prompt and HF leptons
    # isolated_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==1]
    # HF_leptons = [lepton for lepton in data if lepton[lep_feature_dict['lepIso_lep_isolated']]==0]

    # group leptons with their nearby tracks
    print("Grouping leptons and tracks")
    for eventN in np.unique(leptons['eventN']):
        event_leptons = [i for i in leptons if i[0]==eventN]
        event_tracks = [i for i in tracks if i[0]==eventN]
        associated_tracks = group_leptons_and_tracks(event_leptons, event_tracks)
        pdb.set_trace()

    # calculate ptcone
    print("Calculating ptcone variables")
    cones = []
    for i, (lepton, tracks) in enumerate(zip(leptons, associated_tracks)):
        if i%100 == 0:
            print("%d/%d" % (i, len(leptons)))
        stored_cones = [0] * len(calculated_cones)
        calculated_cones = calculate_ptcone_and_etcone(lepton, tracks)
        cones.append(zip(stored_cones, calculated_cones))
    pdb.set_trace()

    # plot comparisons for calculated and stored ptcone features
    Print("Producing plots")
    for stored_feature, calc_feature in cones:
        lepton_feature_values = [lepton[lep_feature_dict[stored_feature]] for lepton in data]
        lepton_calc_feature_values = [lepton[lep_feature_dict[calc_feature]] for lepton in data]
        plt.scatter(lepton_feature_values, lepton_calc_feature_values, s=1)
        # heatmap, xedges, yedges = np.histogram2d(lepton_feature_values, lepton_calc_feature_values, bins=500)
        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        # plt.clf()
        # plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.title(stored_feature + ' vs. ' + calc_feature)
        plt.xlabel(stored_feature)
        plt.ylabel(calc_feature)
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.savefig(saveDir + stored_feature + "_vs_" + calc_feature + ".png", bbox_inches='tight')
        plt.clf()
    pdb.set_trace()

    # # plot comparisons for all lepton features
    # for feature, index in lep_feature_dict.items():
        # if feature == 'lepIso_lep_associated_tracks':
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
    saveDir = "."
    compareFeatures(inFile, saveDir)
