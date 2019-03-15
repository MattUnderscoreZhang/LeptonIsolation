import os
import sys
import h5py as h5
import pickle as pkl
import HEP
import numpy as np


def prepare_data(h5_file):

    '''Either generate pseudodata or convert H5 data to pkl form. Returns lepton-track bunches.'''

    if h5_file is None:
        leptons_with_tracks = generate_pseudodata()
    else:
        leptons_with_tracks = prepare_h5_data(h5_file)
    return leptons_with_tracks


def generate_pseudodata(n_lep_each_type = 10000):

    '''Generate track-lepton pseudodata. This function is incomplete.'''
    sys.exit("generate_pseudodata is not in working condition.")

    unnormed_leptons = []
    unnormed_tracks = []
    for _ in range(n_lep_each_type):
        # create a random HF lepton and tracks
        new_lepton = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 3]  # only truth_type matters here
        new_tracks = []
        n_tracks = 2
        for _ in range(n_tracks):
            new_track = [0, 0]
            new_tracks.append(new_track)
        unnormed_leptons.append(np.array(new_lepton))
        unnormed_tracks.append(np.array(new_tracks))
        # create a random isolated lepton and tracks
        new_lepton = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 2]  # only truth_type matters here
        new_tracks = []
        n_tracks = 2
        for _ in range(n_tracks):
            new_track = [1, 1]
            new_tracks.append(new_track)
        unnormed_leptons.append(np.array(new_lepton))
        unnormed_tracks.append(np.array(new_tracks))

    # normalize and create final data structure
    normed_leptons, normed_tracks = normalize_leptons_and_tracks(
        unnormed_leptons, unnormed_tracks)
    leptons_with_tracks = {'unnormed_leptons': unnormed_leptons,
                           'normed_leptons': normed_leptons,
                           'unnormed_tracks': unnormed_tracks,
                           'normed_tracks': normed_tracks}

    return leptons_with_tracks


def prepare_h5_data(h5_file):

    '''Create lepton-track groups using data from an H5 file.'''

    print("Loading data")
    electrons = h5_file['electrons'][()]
    muons = h5_file['muons'][()]
    tracks = h5_file['tracks'][()]
    n_events = electrons.shape[0]

    print("Grouping leptons and tracks")
    unnormed_leptons = []
    unnormed_tracks = []
    for event_n in range(n_events):
        if event_n % 1000 == 0:
            print("Event %d/%d" % (event_n, n_events))
        leptons = np.append(electrons[event_n], muons[event_n])
        leptons = np.array([i for i in leptons if ~np.isnan(i[1])])
        if len(leptons) == 0:
            continue
        grouped_leptons, grouped_tracks = group_leptons_and_tracks(leptons, tracks[event_n])
        unnormed_leptons += grouped_leptons
        unnormed_tracks += grouped_tracks

    normed_leptons, normed_tracks = normalize_leptons_and_tracks(
        unnormed_leptons, unnormed_tracks)

    leptons_with_tracks = {'unnormed_leptons': unnormed_leptons,
                           'normed_leptons': normed_leptons,
                           'unnormed_tracks': unnormed_tracks,
                           'normed_tracks': normed_tracks}

    return leptons_with_tracks


def group_leptons_and_tracks(leptons, tracks):

    '''Group leptons with nearby tracks (except tracks associated to lepton itself).'''

    good_leptons = filter_leptons(leptons)
    good_tracks = filter_tracks(tracks)

    grouped_leptons = []
    grouped_tracks = []

    for lepton in good_leptons:

        nearby_tracks = []

        # find tracks within dR of lepton i
        for track in good_tracks:

            if abs((lepton['z0'] - track['z0']) * np.sin(track['theta'])) > 3:
                continue
            # calculate and save dR
            dR = HEP.dR(lepton['phi'], lepton['eta'],
                        track['phi'], track['eta'])
            dEta = HEP.dEta(lepton['eta'], track['eta'])
            dPhi = HEP.dPhi(lepton['phi'], track['phi'])
            dd0 = abs(lepton['d0'] - track['d0'])
            dz0 = abs(lepton['z0'] - track['z0'])
            if dR < 0.5:
                nearby_tracks.append(np.array([dR, dEta, dPhi, dd0, dz0,
                                               track['charge'], track['eta'],
                                               track['pT'], track['theta'],
                                               track['d0'], track['z0'],
                                               track['chiSquared']],
                                               dtype=float))

        # remove tracks associated to lepton
        nearby_tracks = remove_lepton_associated_tracks(nearby_tracks)

        # add lepton and tracks to return data
        if len(nearby_tracks) > 0:
            grouped_leptons.append(np.array(list(lepton)))
            grouped_tracks.append(np.array(nearby_tracks, dtype=float))

    return grouped_leptons, grouped_tracks


def filter_leptons(leptons):

    '''from https://gitlab.cern.ch/atlas/athena/blob/master/
    PhysicsAnalysis/MCTruthClassifier/MCTruthClassifier/
    MCTruthClassifierDefs.h'''

    # 2/3 (6/7) is iso/non-iso electron (muon)
    HF_lep_types = [i in [3, 7] for i in leptons['truth_type']]
    prompt_lep_types = [i in [2, 6] for i in leptons['truth_type']]
    # good_pt = leptons['pT'] < 10000
    good_HF_leptons = leptons[HF_lep_types]
    good_prompt_leptons = leptons[prompt_lep_types]
    # n_each_type = min(len(good_HF_leptons), len(good_prompt_leptons))
    # print("Event has", len(leptons), ", good:", n_each_type*2)
    # if n_each_type == 0:
    # return [], []
    # good_leptons = np.concatenate((good_HF_leptons[:n_each_type],\
    # good_prompt_leptons[:n_each_type]))
    good_leptons = np.concatenate((good_HF_leptons, good_prompt_leptons))

    return good_leptons


def filter_tracks(tracks):

    """see if track passes selections listed at
    https://twiki.cern.ch/twiki/bin/view/AtlasProtected/
    Run2IsolationHarmonisation
    and
    https://twiki.cern.ch/twiki/bin/view/AtlasProtected/
    TrackingCPRecsEarly2018"""

    good_track_pt = tracks['pT'] > 500  # 500 MeV
    good_track_eta = abs(tracks['eta']) < 2.5
    good_track_hits = [i + j >= 7 for i, j in zip(
        tracks['nSCTHits'], tracks['nPixHits'])] and\
        (tracks['nIBLHits'] > 0)
    good_track_holes = [i + j <= 2 for i, j in zip(
        tracks['nPixHoles'], tracks['nSCTHoles'])] and\
        (tracks['nPixHoles'] <= 1)
    good_tracks = tracks[good_track_pt & good_track_eta &
                         good_track_hits & good_track_holes]

    return good_tracks


def remove_lepton_associated_tracks(tracks):
    
    '''Sort tracks by dR to lepton and remove the closest one.'''

    tracks.sort(key=lambda x: x[0])
    if len(tracks) > 0:
        tracks.pop(0)
    return tracks


def normalize_leptons_and_tracks(unnormed_leptons, unnormed_tracks):

    '''Normalize all lepton and track features except for pdgID and truth_type.'''

    unfolded_leptons = np.array(unnormed_leptons)
    unfolded_tracks = np.array([i for lep_tracks in unnormed_tracks for i in lep_tracks])
    lepton_means = unfolded_leptons.mean(axis=0)
    lepton_stds = unfolded_leptons.std(axis=0)
    track_means = unfolded_tracks.mean(axis=0)
    track_stds = unfolded_tracks.std(axis=0)
    for i in [0, 12]:  # ignore pdgID and truth_type
        lepton_means[i] = 0
        lepton_stds[i] = 1
    normed_leptons = [(i - lepton_means) / lepton_stds for i in unnormed_leptons]
    normed_tracks = [[(j - track_means) / track_stds for j in i] for i in unnormed_tracks]

    return normed_leptons, normed_tracks


def balance_classes(data):

    '''Reduces number of background events to match number of signal events.'''

    is_HF_lepton = [i[12] in [3, 7] for i in data['unnormed_leptons']]
    is_prompt_lepton = [i[12] in [2, 6] for i in data['unnormed_leptons']]
    n_each_type = min(sum(is_HF_lepton), sum(is_prompt_lepton))

    def balance(data):
        good_HF_data = np.array(data)[is_HF_lepton][:n_each_type]
        good_prompt_data = np.array(data)[is_prompt_lepton][:n_each_type]
        return np.concatenate([good_HF_data, good_prompt_data])

    for key in ['unnormed_tracks', 'normed_leptons', 'normed_tracks', 'unnormed_leptons']:
        data[key] = balance(data[key])
    return data


if __name__ == "__main__":

    # prepare all data
    sample_path = sys.argv[1]
    lepton_track_data = []
    count = 0
    for filename in os.listdir(sample_path + "/H5/"):
        print("Working on file " + filename)
        with h5.File(sample_path + "/H5/" + filename) as h5_file:
            data = prepare_data(h5_file)
            lepton_track_data.append(data)

    # merge pkl files
    all_data = {}
    all_keys = ['unnormed_tracks', 'normed_leptons', 'normed_tracks', 'unnormed_leptons']
    for data in lepton_track_data:
        for key in all_keys:
            if key not in all_data.keys():
                all_data[key] = data[key]
            else:
                all_data[key] += data[key]

    # add data labels
    all_data["lepton_labels"] = ['pdgID', 'pT', 'eta', 'phi', 'd0', 'z0',
                     'ptcone20', 'ptcone30', 'ptcone40',
                     'ptvarcone20', 'ptvarcone30',
                     'ptvarcone40', 'truth_type', 'PLT']
    all_data["track_labels"] = ['dR', 'dEta', 'dPhi', 'dd0', 'dz0',
                    'charge', 'eta', 'pT', 'theta', 'd0', 'z0', 'chiSquared']

    # balance classes
    all_data = balance_classes(all_data)

    # save merged file
    if not os.path.exists(sample_path + "/pkl/"):
        os.mkdir(sample_path + "/pkl/")
    with open(sample_path + "/pkl/lepton_track_data.pkl", 'wb') as out_file:
        pkl.dump(all_data, out_file)
