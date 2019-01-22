import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns
sns.set()
import pickle
import pdb

# open file
data_filename = "../../Data/lepton_track_data.pkl"
with open(data_filename, 'rb') as data_file:
    leptons_with_tracks = pickle.load(data_file)

# extract ptcone info
leptons = leptons_with_tracks['unnormed_leptons']
lepton_keys = leptons_with_tracks['lepton_labels']
isolated = [int(lepton[lepton_keys.index('truth_type')] in [2, 6])
            for lepton in leptons]
cones = {}
pt_keys = ['ptcone20', 'ptcone30', 'ptcone40',
           'ptvarcone20', 'ptvarcone30', 'ptvarcone40']
for key in pt_keys:
    cones[key] = [lepton[lepton_keys.index(key)] for lepton in leptons]
    max_key = max(cones[key])
    min_key = min(cones[key])
    range_key = max_key - min_key
    cones[key] = [(i - min_key) / range_key for i in cones[key]]

# get rid of events with ptcone=0
good_leptons = [lepton[lepton_keys.index(
    'ptcone20')] > 0 for lepton in leptons]
leptons = np.array(leptons)[good_leptons]
isolated = np.array(isolated)[good_leptons]
for key in pt_keys:
    cones[key] = np.array(cones[key])[good_leptons]

# make ROC comparison plots
for key in pt_keys:
    fpr, tpr, thresholds = metrics.roc_curve(isolated, cones[key])
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(tpr, fpr, lw=2, label=key)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid('on', linestyle='--')
plt.title('ROC Curve for Classification')
plt.legend(loc="lower right")
# plt.savefig("compare_ROC.png")
plt.show()
