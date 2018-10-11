import Loader.loader as loader

for i in range(1, 11):
    in_file = "../Data/output_" + str(i) + ".h5"
    save_file = "../Data/lepton_track_data_" + str(i) + ".pkl"
    leptons_with_tracks = loader.create_or_load(
        in_file, save_file, overwrite=False, pseudodata=False)
