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
        myFeatures.add("lep_q", event.lep_q)
        myFeatures.add("lep_pt", event.lep_pt)
        myFeatures.add("lep_eta", event.lep_eta)
        myFeatures.add("lep_phi", event.lep_phi)
        myFeatures.add("lep_m", event.lep_m)
        myFeatures.add("lep_d0", event.lep_d0)
        myFeatures.add("lep_z0", event.lep_z0)
        myFeatures.add("lep_d0Err", event.lep_d0Err)
        myFeatures.add("lep_z0Err", event.lep_z0Err)
        myFeatures.add("lep_pTErr", event.lep_pTErr)
        myFeatures.add("lep_ptcone20", event.lep_ptcone20)
        myFeatures.add("lep_ptcone30", event.lep_ptcone30)
        myFeatures.add("lep_ptcone40", event.lep_ptcone40)
        myFeatures.add("lep_topoetcone20", event.lep_topoetcone20)
        myFeatures.add("lep_topoetcone30", event.lep_topoetcone30)
        myFeatures.add("lep_topoetcone40", event.lep_topoetcone40)
        myFeatures.add("lep_ptvarcone20", event.lep_ptvarcone20)
        myFeatures.add("lep_ptvarcone30", event.lep_ptvarcone30)
        myFeatures.add("lep_ptvarcone40", event.lep_ptvarcone40)
        myFeatures.add("lep_truthAuthor", event.lep_truthAuthor)
        myFeatures.add("lep_truthType", event.lep_truthType)
        myFeatures.add("lep_truthOrigin", event.lep_truthOrigin)
        myFeatures.add("track_q", event.track_q)
        myFeatures.add("track_pt", event.track_pt)
        myFeatures.add("track_eta", event.track_eta)
        myFeatures.add("track_phi", event.track_phi)
        myFeatures.add("track_m", event.track_m)
        myFeatures.add("track_fitQuality", event.track_fitQuality)
        myFeatures.add("track_d0", event.track_d0)
        myFeatures.add("track_z0", event.track_z0)
        myFeatures.add("track_d0Err", event.track_d0Err)
        myFeatures.add("track_z0Err", event.track_z0Err)
        myFeatures.add("track_nIBLHits", event.track_nIBLHits)
        myFeatures.add("track_nPixHits", event.track_nPixHits)
        myFeatures.add("track_nPixHoles", event.track_nPixHoles)
        myFeatures.add("track_nPixOutliers", event.track_nPixOutliers)
        myFeatures.add("track_nSCTHits", event.track_nSCTHits)
        myFeatures.add("track_nTRTHits", event.track_nTRTHits)

    # Save features to an h5 file
    f = h5.File(outFile, "w")
    for key in myFeatures.keys():
        f.create_dataset(key, data=np.array(myFeatures.get(key)).squeeze(),compression='gzip')
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
