import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

data = h5.File("output.h5")

plot_names = ["ptcone20", "ptcone30", "ptcone40", "ptvarcone20", "ptvarcone30", "ptvarcone40"]
for n, plot_name in enumerate(plot_names):
    ref_ptcone = [i[8+n] for j in data['muons'] for i in j if not np.isnan(i[8+n])]
    calc_ptcone = [i[14+n] for j in data['muons'] for i in j if not np.isnan(i[14+n])]
    plt.scatter(ref_ptcone, calc_ptcone, s=10, alpha=0.35)
    plt.title("Muon Comparison for " + plot_name)
    plt.xlabel("Reference " + plot_name)
    plt.ylabel("Calculated " + plot_name)
    plt.savefig("muon_" + plot_name + "_comparison.png")
    plt.clf()

plot_names = ["ptcone20", "ptvarcone20", "ptvarcone30", "ptvarcone40"]
for n, plot_name in enumerate(plot_names):
    ref_ptcone = [i[8+n] for j in data['electrons'] for i in j if not np.isnan(i[8+n])]
    calc_ptcone = [i[12+n] for j in data['electrons'] for i in j if not np.isnan(i[12+n])]
    plt.scatter(ref_ptcone, calc_ptcone, s=10, alpha=0.35)
    plt.title("Electron Comparison for " + plot_name)
    plt.xlabel("Reference " + plot_name)
    plt.ylabel("Calculated " + plot_name)
    plt.savefig("electron_" + plot_name + "_comparison.png")
    plt.clf()
