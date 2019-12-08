from ROOT import TFile


def make_plots(ROOT_filename, treename):
    """Make comparison histograms of lepton and track features for HF vs. isolated leptons."""

    print("Loading data")
    data_file = TFile(ROOT_filename)
    data_tree = getattr(data_file, treename)
    n_events = data_tree.GetEntries()
    print(n_events)
    data_file.Close()


if __name__ == "__main__":
    make_plots("/public/data/RNN/data.root", "UnnormedTree")
