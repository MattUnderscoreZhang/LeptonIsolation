// local tools
#include "Root/TrackWriter.h"
#include "Root/ElectronWriter.h"
#include "Root/MuonWriter.h"

// EDM things
#include "xAODTracking/TrackParticleContainer.h"
#include "xAODEgamma/ElectronContainer.h"
#include "xAODMuon/MuonContainer.h"

// AnalysisBase tool include(s):
#include "xAODRootAccess/Init.h"
#include "xAODRootAccess/TEvent.h"
#include "xAODRootAccess/tools/ReturnCheck.h"

// 3rd party includes
#include "TFile.h"
#include "H5Cpp.h"

// stl includes
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <cassert>

///////////////////////////
// simple options struct //
///////////////////////////
struct Options
{
    std::vector<std::string> files;
    std::string nn_file;
};
// simple options parser
Options get_options(int argc, char *argv[]);

////////////////////////////////////
// baseline selection for leptons //
////////////////////////////////////
bool ObjectTools::passBaselineSelection(const xAOD::IParticle* p)
{

  // Baseline skimming for all objects
  if( m_filterBaseline && !cacc_baseline(*p)  ) return false;

  // OR skimming for all objects by Taus
  if ( p->type() != xAOD::Type::Tau && cacc_passOR.isAvailable(*p) ){
    if( m_filterOR && !cacc_passOR(*p)  ) return false;
  }

  // Keep track of number of leptons in the event
  if ( p->type() == xAOD::Type::Muon || p->type() == xAOD::Type::Electron ){
    if( cacc_baseline(*p) ) m_evtProperties->nBaseLeptons++;
    if( cacc_signal(*p)   ) m_evtProperties->nSignalLeptons++;
  }

  // Keep track of the number of photons in the event
  if ( p->type() == xAOD::Type::Photon ){
    if( cacc_baseline(*p) ) m_evtProperties->nBasePhotons++;
    if( cacc_signal(*p)   ) m_evtProperties->nSignalPhotons++;
  }

  // Desired object!
  return true;

}

//////////////////
// main routine //
//////////////////
int main (int argc, char *argv[])
{
    const char* ALG = argv[0];
    Options opts = get_options(argc, argv);

    // set up xAOD basics
    RETURN_CHECK(ALG, xAOD::Init());
    xAOD::TEvent event(xAOD::TEvent::kClassAccess);

    // set up output file
    H5::H5File output("output.h5", H5F_ACC_TRUNC);
    TrackWriter track_writer(output);
    ElectronWriter electron_writer(output);
    MuonWriter muon_writer(output);

    // Loop over the specified files:
    for (std::string file_name: opts.files) {

        // Open the file:
        std::unique_ptr<TFile> ifile(TFile::Open(file_name.c_str(), "READ"));
        if ( ! ifile.get() || ifile->IsZombie()) {
            throw std::logic_error("Couldn't open file: " + file_name);
            return 1;
        }
        std::cout << "Opened file: " << file_name << std::endl;

        // Connect the event object to it:
        RETURN_CHECK(ALG, event.readFrom(ifile.get()));

        // Loop over its events:
        const unsigned long long entries = event.getEntries();
        std::cout << "got " << entries << " entries" << std::endl;
        for (unsigned long long entry = 0; entry < entries; ++entry) {

            // Print some status
            if ( ! (entry % 500)) {
                std::cout << "Processing " << entry << "/" << entries << "\n";
            }

            // Load the event
            bool ok = event.getEntry(entry) >= 0;
            if (!ok) throw std::logic_error("getEntry failed");

            // Write track info
            const xAOD::TrackParticleContainer *tracks = 0;
            RETURN_CHECK(ALG, event.retrieve(tracks, "InDetTrackParticles"));
            for (const xAOD::TrackParticle *track : *tracks) {
                track_writer.write(*track);
            }

            // Write lepton info
            const xAOD::ElectronContainer *electrons = 0;
            RETURN_CHECK(ALG, event.retrieve(electrons, "Electrons"));
            for (const xAOD::Electron *electron : *electrons) {
                electron_writer.write(*electron);
            }
            const xAOD::MuonContainer *muons = 0;
            RETURN_CHECK(ALG, event.retrieve(muons, "Muons"));
            for (const xAOD::Muon *muon : *muons) {
                muon_writer.write(*muon);
            }

        } // end event loop
    } // end file loop


    return 0;
}

/////////////////////////////////
// command line options parser //
/////////////////////////////////
void usage(std::string name) {
    std::cout << "usage: " << name << " [-h] [--nn-file NN_FILE] <AOD>..."
        << std::endl;
}

Options get_options(int argc, char *argv[]) {
    Options opts;
    for (int argn = 1; argn < argc; argn++) {
        std::string arg(argv[argn]);
        if (arg == "--nn-file") {
            argn++;
            opts.nn_file = argv[argn];
        } else if (arg == "-h") {
            usage(argv[0]);
            exit(1);
        } else {
            opts.files.push_back(arg);
        }
    }
    if (opts.files.size() == 0) {
        usage(argv[0]);
        exit(1);
    }
    return opts;
}
