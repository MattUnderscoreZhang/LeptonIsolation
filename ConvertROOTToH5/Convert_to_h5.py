import numpy as np
import sys, ast, os
import h5py as h5

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

    myFeatures = FeaturesList()

    with open(inFile) as myfile:
        for index,line in enumerate(myfile):

            my_event_string = line.replace('\n', '')
            my_event_string = my_event_string.replace(' ', '')
            my_event_string = my_event_string.replace('}{','} {')
            my_event = ast.literal_eval(my_event_string)

            if index%200 == 0:
                print "Event", index

            myFeatures.add("HCAL", HCAL_window)
            myFeatures.add("energy", my_event['E']/1000.) # convert MeV to GeV
            myFeatures.add("pdgID", my_event['pdgID'])
            myFeatures.add("conversion", my_event['conversion'])
            myFeatures.add("openingAngle", my_event['openingAngle'])

    # Save features to an h5 file
    f = h5.File(outFile, "w")
    for key in myFeatures.keys():
        f.create_dataset(key, data=np.array(myFeatures.get(key)).squeeze(),compression='gzip')
    f.close()
    
#################
# Main function #
#################

if __name__ == "__main__":
    inFile = "/afs/cern.ch/work/m/mazhang/LepIso/MC/393407/DAOD_SUSY16.12999070._000001.pool.root.1"
    outFile = sys.argv[2]
    print "Converting file"
    convertFile(inFile, outFile+"_temp")
    print "Calculating features"
    addFeatures.convertFile(outFile+"_temp", outFile)
    os.remove(outFile+"_temp")
