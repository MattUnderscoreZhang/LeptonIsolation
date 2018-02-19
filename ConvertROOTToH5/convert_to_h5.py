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

    for event in input_file.tree_NoSys:
        myFeatures.add("q", event.q)
        myFeatures.add("pt", event.pt)
        myFeatures.add("eta", event.eta)
        myFeatures.add("phi", event.phi)
        myFeatures.add("m", event.m)
        myFeatures.add("fitQuality", event.fitQuality)
        myFeatures.add("d0", event.d0)
        myFeatures.add("z0Err", event.z0Err)
        myFeatures.add("nIBLHits", event.nIBLHits)
        myFeatures.add("nPixHits", event.nPixHits)
        myFeatures.add("nPixHoles", event.nPixHoles)
        myFeatures.add("nPixOutliers", event.nPixOutliers)
        myFeatures.add("nSCTHits", event.nSCTHits)
        myFeatures.add("nTRTHits", event.nTRTHits)
        myFeatures.add("truthType", event.truthType)
        myFeatures.add("truthOrigin", event.truthOrigin)

    # Save features to an h5 file
    f = h5.File(outFile, "w")
    for key in myFeatures.keys():
        f.create_dataset(key, data=np.array(myFeatures.get(key)).squeeze(),compression='gzip')
    f.close()
    
#################
# Main function #
#################

if __name__ == "__main__":
    inFile = "/afs/cern.ch/work/m/mazhang/LepIso/Ntuples/393407/393407.root"
    outFile = "/afs/cern.ch/work/m/mazhang/LepIso/H5/393407.h5"
    print "Converting file"
    convertFile(inFile, outFile)
    print "Finished"
