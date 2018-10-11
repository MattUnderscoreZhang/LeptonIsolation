import Loader.loader as loader
import pickle as pkl
import pdb

# create pkl files
for i in range(1, 11):
    in_file = "../Data/output_" + str(i) + ".h5"
    save_file = "../Data/lepton_track_data_" + str(i) + ".pkl"
    leptons_with_tracks = loader.create_or_load(
        in_file, save_file, overwrite=False, pseudodata=False)

# merge pkl files
all_data = {}
all_keys = ['unnormed_tracks', 'normed_leptons', 'normed_tracks', 'unnormed_leptons']

data_file = "../Data/lepton_track_data_1.pkl"
data = pkl.load(open(data_file, 'rb'))
for key in all_keys:
    all_data[key] = data[key]
all_data['lepton_labels'] = data['lepton_labels']
all_data['track_labels'] = data['track_labels']

for i in range(2, 11):
    data_file = "../Data/lepton_track_data_" + str(i) + ".pkl"
    data = pkl.load(open(data_file, 'rb'))
    for key in all_keys:
        all_data[key] += data[key]

# save merged file
with open("../Data/lepton_track_data.pkl", 'wb') as out_file:
    pkl.dump(all_data, out_file)
