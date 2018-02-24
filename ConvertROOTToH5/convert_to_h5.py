import numpy as np
import sys, ast
import h5py as h5
import ROOT

############
# Features #
############

class FeaturesList(object):

    def __init__(self):
        self.features = {}

    def add(self, featureName, feature):
        self.features.setdefault(featureName, []).append(feature)

    def keys(self):
        return self.features.keys()

    def get(self, featureName):
        return self.features[featureName]

############################
# File reading and writing #
############################

def convertFile(inFile, outFile):

    input_file = ROOT.TFile.Open(str(inFile), "read")
    myFeatures = FeaturesList()

    all_features = [
        ("lep_q", "int"),
        ("lep_pt", "float"),
        ("lep_eta", "float"),
        ("lep_phi", "float"),
        ("lep_m", "float"),
        ("lep_d0", "float"),
        ("lep_z0", "float"),
        ("lep_d0Err", "float"),
        ("lep_z0Err", "float"),
        ("lep_pTErr", "float"),
        ("lep_ptcone20", "float"),
        ("lep_ptcone30", "float"),
        ("lep_ptcone40", "float"),
        ("lep_topoetcone20", "float"),
        ("lep_topoetcone30", "float"),
        ("lep_topoetcone40", "float"),
        ("lep_ptvarcone20", "float"),
        ("lep_ptvarcone30", "float"),
        ("lep_ptvarcone40", "float"),
        ("lep_truthAuthor", "int"),
        ("lep_truthType", "int"),
        ("lep_truthOrigin", "int"),
        ("track_q", "float"),
        ("track_pt", "float"),
        ("track_eta", "float"),
        ("track_phi", "float"),
        ("track_m", "float"),
        ("track_fitQuality", "float"),
        ("track_d0", "float"),
        ("track_z0", "float"),
        ("track_d0Err", "float"),
        ("track_z0Err", "float"),
        ("track_nIBLHits", "int"),
        ("track_nPixHits", "int"),
        ("track_nPixHoles", "int"),
        ("track_nPixOutliers", "int"),
        ("track_nSCTHits", "int"),
        ("track_nTRTHits", "int")]

    for event in input_file.tree_NoSys:
        for (feature, _) in all_features:
            exec("event_feature = event." + feature)
            value = []
            for i in range(event_feature.size()):
                value.append(event_feature[i])
            myFeatures.add(feature, value)

    # Save features to an h5 file
    f = h5.File(outFile, "w")
    for (feature, data_type) in all_features:
        feature_data = myFeatures.get(feature)
        if data_type == "int":
            dt = h5.special_dtype(vlen=np.dtype('int32'))
        elif data_type == "float":
            dt = h5.special_dtype(vlen=np.dtype('float32'))
        new_data = f.create_dataset(feature, (len(feature_data),), dtype=dt)
        for i, data in enumerate(feature_data):
            new_data[i] = data
    f.close()
    
#################
# Main function #
#################

if __name__ == "__main__":
    inFile = "/afs/cern.ch/work/m/mazhang/LepIso/Ntuples/393407.root"
    outFile = "/afs/cern.ch/work/m/mazhang/LepIso/H5/393407.h5"
    print "Converting file"
    convertFile(inFile, outFile)
    print "Finished"
