// local tools
#include "../headers/TrackFilter.h"
#include "../headers/ElectronFilter.h"
#include "../headers/MuonFilter.h"
#include "../headers/TrackWriter.h"
#include "../headers/ElectronWriter.h"
#include "../headers/MuonWriter.h"

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

    // object filters
    TrackFilter track_filter;
    ElectronFilter electron_filter;
    MuonFilter muon_filter;

    // object writers
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

            // Get tracks and leptons
            const xAOD::TrackParticleContainer *tracks = 0;
            RETURN_CHECK(ALG, event.retrieve(tracks, "InDetTrackParticles"));
            const xAOD::VertexContainer *primary_vertices = 0;
            RETURN_CHECK(ALG, event.retrieve(primary_vertices, "PrimaryVertices"));
            const xAOD::ElectronContainer *electrons = 0;
            RETURN_CHECK(ALG, event.retrieve(electrons, "Electrons"));
            const xAOD::MuonContainer *muons = 0;
            RETURN_CHECK(ALG, event.retrieve(muons, "Muons"));

            // Filter objects
            std::vector<const xAOD::TrackParticle*> filtered_tracks = track_filter.filter_tracks(*tracks);
            std::vector<const xAOD::Muon*> filtered_muons = muon_filter.filter_muons(*muons);
            std::vector<const xAOD::Electron*> filtered_electrons = electron_filter.filter_electrons(*electrons);

            // Write event
            track_writer.write(filtered_tracks);
            electron_writer.write(filtered_electrons, *primary_vertices);
            muon_writer.write(filtered_muons, *primary_vertices);

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
