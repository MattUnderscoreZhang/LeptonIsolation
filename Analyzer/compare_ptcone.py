from Analysis import FeatureComparer

if __name__ == "__main__":

    make ptcone and etcone comparison plots - normed
    plot_save_dir = "../Plots_normed/"
    pathlib.Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
    lwt = list(zip(
        leptons_with_tracks['normed_leptons'],
        leptons_with_tracks['normed_tracks']))
    labels = [leptons_with_tracks['lepton_labels'],
        leptons_with_tracks['track_labels']]
    FeatureComparer.compare_ptcone_and_etcone(lwt, labels, plot_save_dir)

    make ptcone and etcone comparison plots - unnormed
    plot_save_dir = "../Plots_unnormed/"
    pathlib.Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
    lwt = list(zip(
        leptons_with_tracks['unnormed_leptons'],
        leptons_with_tracks['unnormed_tracks']))
    labels = [leptons_with_tracks['lepton_labels'],
        leptons_with_tracks['track_labels']]
    FeatureComparer.compare_ptcone_and_etcone(lwt, labels, plot_save_dir)
