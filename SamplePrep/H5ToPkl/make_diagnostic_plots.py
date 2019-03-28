import sys
import matplotlib
matplotlib.use('Agg') # NOQA
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle as pkl


def calculate_ptcone_and_etcone(leptons_with_tracks_i, labels):

    lepton = leptons_with_tracks_i[0]
    tracks = leptons_with_tracks_i[1]

    lepton_keys = labels[0]
    track_keys = labels[1]

    cones = {}
    cones['truth_ptcone20'] = lepton[lepton_keys.index('ptcone20')]
    cones['truth_ptcone30'] = lepton[lepton_keys.index('ptcone30')]
    cones['truth_ptcone40'] = lepton[lepton_keys.index('ptcone40')]
    cones['truth_ptvarcone20'] = lepton[lepton_keys.index('ptvarcone20')]
    cones['truth_ptvarcone30'] = lepton[lepton_keys.index('ptvarcone30')]
    cones['truth_ptvarcone40'] = lepton[lepton_keys.index('ptvarcone40')]
    cones['ptcone20'] = 0
    cones['ptcone30'] = 0
    cones['ptcone40'] = 0
    cones['ptvarcone20'] = 0
    cones['ptvarcone30'] = 0
    cones['ptvarcone40'] = 0

    lep_pt = lepton[lepton_keys.index('pT')] / 1000
    for track in tracks:
        dR = track[track_keys.index('dR')]
        track_pt = track[track_keys.index('pT')]
        # if track_pt < 1000: continue # 1 GeV tracks
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


def compare_ptcone_and_etcone(leptons_with_tracks, labels, plot_save_dir, normed=False):

    # labels
    lepton_keys = labels[0]
    track_keys = labels[1]

    # separate flavors
    # electrons_with_tracks = [lwt for lwt in leptons_with_tracks if lwt[0][lepton_keys.index('pdgID')] == 11]
    # muons_with_tracks = [lwt for lwt in leptons_with_tracks if lwt[0][lepton_keys.index('pdgID')] == 13]

    # calculate ptcone (just muons for now until I figure out electrons)
    print("Calculating ptcone variables")
    cones = {}
    # for i, leptons_with_tracks_i in enumerate(electrons_with_tracks):
    # for i, leptons_with_tracks_i in enumerate(muons_with_tracks):
    for i, leptons_with_tracks_i in enumerate(leptons_with_tracks):
        cones_i = calculate_ptcone_and_etcone(leptons_with_tracks_i, labels)
        for key in cones_i.keys():
            cones.setdefault(key, []).append(cones_i[key])

    # plot comparisons for calculated and stored ptcone features
    print("Producing plots")
    cone_features = ['ptcone20', 'ptcone30', 'ptcone40',
                     'ptvarcone20', 'ptvarcone30', 'ptvarcone40']
    for feature in cone_features:
        plt.scatter(cones[feature], cones["truth_" + feature], s=1)
        # heatmap, xedges, yedges = np.histogram2d(lepton_feature_values, lepton_calc_feature_values, bins=500)
        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        # plt.clf()
        # plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.title(feature)
        plt.xlabel(feature)
        plt.ylabel("Truth " + feature)
        # plt.xlim(0, 200000)
        # plt.ylim(0, 200000)
        plt.xlim(0, 50000)
        plt.ylim(0, 50000)
        if normed:
            plt.xlim(0, 20)
            plt.ylim(0, 20)
        plt.savefig(plot_save_dir + feature +
                    "_scatter.eps", bbox_inches='tight')
        plt.clf()

    # plot comparisons for all lepton features
    isolated_leptons = [lwt[0] for lwt in leptons_with_tracks if lwt[0]
                        [lepton_keys.index('truth_type')] in [2, 6]]
    HF_leptons = [lwt[0] for lwt in leptons_with_tracks if lwt[0]
                  [lepton_keys.index('truth_type')] in [3, 7]]
    for feature in lepton_keys:
        isolated_feature_values = [
            lepton[lepton_keys.index(feature)] for lepton in isolated_leptons]
        HF_feature_values = [
            lepton[lepton_keys.index(feature)] for lepton in HF_leptons]
        all_feature_values = isolated_feature_values + HF_feature_values
        bins = np.linspace(min(all_feature_values),
                           max(all_feature_values), 30)
        # bins = np.linspace(min(all_feature_values), 2*np.median(all_feature_values), 30)
        plt.hist([isolated_feature_values, HF_feature_values],
                 normed=True, bins=bins, histtype='step', linewidth=1)
        plt.title(feature)
        plt.legend(['HF', 'isolated'])  # yes, I think this order is correct
        plt.savefig(plot_save_dir + "lepton_" +
                    feature + ".eps", bbox_inches='tight')
        plt.clf()

    # plot nTracks
    isolated_tracks = [lwt[1] for lwt in leptons_with_tracks if lwt[0]
                       [lepton_keys.index('truth_type')] in [2, 6]]
    HF_tracks = [lwt[1] for lwt in leptons_with_tracks if lwt[0]
                 [lepton_keys.index('truth_type')] in [3, 7]]
    isolated_feature_values = [len(tracks) for tracks in isolated_tracks]
    HF_feature_values = [len(tracks) for tracks in HF_tracks]
    # print(sorted(isolated_feature_values))
    # print(sorted(HF_feature_values))
    all_feature_values = isolated_feature_values + HF_feature_values
    bins = np.linspace(0, 15, 15)
    plt.hist([isolated_feature_values, HF_feature_values],
             normed=True, bins=bins, histtype='step', linewidth=1)
    plt.title('ntracks')
    plt.legend(['HF', 'isolated'])
    plt.savefig(plot_save_dir + "ntracks.eps", bbox_inches='tight')
    plt.clf()

    # plot comparisons for all track features
    isolated_tracks = [track for event in isolated_tracks for track in event]
    HF_tracks = [track for event in HF_tracks for track in event]
    for feature in track_keys:
        isolated_feature_values = [
            track[track_keys.index(feature)] for track in isolated_tracks]
        HF_feature_values = [
            track[track_keys.index(feature)] for track in HF_tracks]
        all_feature_values = isolated_feature_values + HF_feature_values
        bins = np.linspace(min(all_feature_values),
                           max(all_feature_values), 30)
        # bins = np.linspace(min(all_feature_values), 2*np.median(all_feature_values), 30)
        plt.hist([isolated_feature_values, HF_feature_values],
                 normed=True, bins=bins, histtype='step', linewidth=1)
        plt.title(feature)
        plt.legend(['HF', 'isolated'])  # yes, I think this order is correct
        plt.savefig(plot_save_dir + "track_" +
                    feature + ".eps", bbox_inches='tight')
        plt.clf()


def compare(options):

    # read data
    data_file = options['input_folder'] + "/pkl/lepton_track_data.pkl"
    leptons_with_tracks = pkl.load(open(data_file, 'rb'), encoding='latin1')

    # make ptcone and etcone comparison plots - normed
    plot_save_dir = options['output_folder'] + "/Normed/"
    pathlib.Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
    lwt = list(zip(
        leptons_with_tracks['normed_leptons'],
        leptons_with_tracks['normed_tracks']))
    labels = [leptons_with_tracks['lepton_labels'],
              leptons_with_tracks['track_labels']]
    compare_ptcone_and_etcone(lwt, labels, plot_save_dir, normed=True)

    # make ptcone and etcone comparison plots - unnormed
    plot_save_dir = options['output_folder'] + "/Unnormed/"
    pathlib.Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
    lwt = list(zip(
        leptons_with_tracks['unnormed_leptons'],
        leptons_with_tracks['unnormed_tracks']))
    labels = [leptons_with_tracks['lepton_labels'],
              leptons_with_tracks['track_labels']]
    compare_ptcone_and_etcone(lwt, labels, plot_save_dir)


if __name__ == "__main__":

    options = {}
    options['input_folder'] = sys.argv[1]
    options['output_folder'] = sys.argv[2]
    compare(options)
