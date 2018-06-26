import matplotlib.pyplot as plt

###################
# Comparison code #
###################

def compareFeatures(cones, plot_save_dir):

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
        plt.xlim(0, 200000)
        plt.ylim(0, 200000)
        plt.savefig(plot_save_dir + feature + ".png", bbox_inches='tight')
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
        # plt.savefig(plot_save_dir + feature + ".png", bbox_inches='tight')
        # plt.clf()
