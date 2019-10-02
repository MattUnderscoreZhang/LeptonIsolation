// local tools
#include "ObjectFilters.cxx"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "xAODEgamma/ElectronContainer.h"
#include "xAODEgamma/Electron.h"
#include "xAODMuon/MuonContainer.h"
#include "xAODMuon/Muon.h"
#include "xAODTracking/TrackParticleContainer.h"
#include "xAODTracking/TrackParticle.h"
#include "xAODEgamma/EgammaxAODHelpers.h"
#include "xAODTracking/TrackParticlexAODHelpers.h"
#include "InDetTrackSelectionTool/InDetTrackSelectionTool.h"

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

using namespace std;

int main (int argc, char *argv[]) {

    // parse input
    const char* ALG = argv[0];
    string inputFilename = argv[1];

    // object filters
    ObjectFilters object_filters;

    // Open the file:
    unique_ptr<TFile> ifile(TFile::Open(inputFilename.c_str(), "READ"));
    if ( ! ifile.get() || ifile->IsZombie()) {
        throw logic_error("Couldn't open file: " + inputFilename);
        return 1;
    }
    cout << "Opened file: " << inputFilename << endl;

    // Connect the event object to it:
    RETURN_CHECK(ALG, xAOD::Init());
    xAOD::TEvent event(xAOD::TEvent::kClassAccess);
    RETURN_CHECK(ALG, event.readFrom(ifile.get()));

    // Leptons
    TFile outputFile("output.root", "recreate");
    TTree* outputTree = new TTree("BaselineTree", "baseline tree");

    int entry_n; outputTree->Branch("event_n", &entry_n, "event_n/I");
    int pdgID; outputTree->Branch("pdgID", &pdgID, "pdgID/I");
    float lep_pT; outputTree->Branch("lep_pT", &lep_pT, "lep_pT/F");
    float lep_eta; outputTree->Branch("lep_eta", &lep_eta, "lep_eta/F");
    float lep_theta; outputTree->Branch("lep_theta", &lep_theta, "lep_theta/F");
    float lep_phi; outputTree->Branch("lep_phi", &lep_phi, "lep_phi/F");
    float lep_d0; outputTree->Branch("lep_d0", &lep_d0, "lep_d0/F");
    float lep_d0_over_sigd0; outputTree->Branch("lep_d0_over_sigd0", &lep_d0_over_sigd0, "lep_d0_over_sigd0/F");
    float lep_z0; outputTree->Branch("lep_z0", &lep_z0, "lep_z0/F");
    float lep_dz0; outputTree->Branch("lep_dz0", &lep_dz0, "lep_dz0/F");
    float ptcone20; outputTree->Branch("ptcone20", &ptcone20, "ptcone20/F");
    float ptcone30; outputTree->Branch("ptcone30", &ptcone30, "ptcone30/F");
    float ptcone40; outputTree->Branch("ptcone40", &ptcone40, "ptcone40/F");
    float ptvarcone20; outputTree->Branch("ptvarcone20", &ptvarcone20, "ptvarcone20/F");
    float ptvarcone30; outputTree->Branch("ptvarcone30", &ptvarcone30, "ptvarcone30/F");
    float ptvarcone40; outputTree->Branch("ptvarcone40", &ptvarcone40, "ptvarcone40/F");
    float topoetcone20; outputTree->Branch("topoetcone20", &topoetcone20, "topoetcone20/F");
    float topoetcone30; outputTree->Branch("topoetcone30", &topoetcone30, "topoetcone30/F");
    float topoetcone40; outputTree->Branch("topoetcone40", &topoetcone40, "topoetcone40/F");
    float eflowcone20; outputTree->Branch("eflowcone20", &eflowcone20, "eflowcone20/F");
    float PLT; outputTree->Branch("PLT", &PLT, "PLT/F");
    int truth_type; outputTree->Branch("truth_type", &truth_type, "truth_type/I");

    vector<float>* trk_lep_dR = new vector<float>; outputTree->Branch("trk_lep_dR", "vector<float>", &trk_lep_dR);
    vector<float>* trk_pT = new vector<float>; outputTree->Branch("trk_pT", "vector<float>", &trk_pT);
    vector<float>* trk_eta = new vector<float>; outputTree->Branch("trk_eta", "vector<float>", &trk_eta);
    vector<float>* trk_phi = new vector<float>; outputTree->Branch("trk_phi", "vector<float>", &trk_phi);
    vector<float>* trk_d0 = new vector<float>; outputTree->Branch("trk_d0", "vector<float>", &trk_d0);
    vector<float>* trk_z0 = new vector<float>; outputTree->Branch("trk_z0", "vector<float>", &trk_z0);
    vector<float>* trk_lep_dEta = new vector<float>; outputTree->Branch("trk_lep_dEta", "vector<float>", &trk_lep_dEta);
    vector<float>* trk_lep_dPhi = new vector<float>; outputTree->Branch("trk_lep_dPhi", "vector<float>", &trk_lep_dPhi);
    vector<float>* trk_lep_dD0 = new vector<float>; outputTree->Branch("trk_lep_dD0", "vector<float>", &trk_lep_dD0);
    vector<float>* trk_lep_dZ0 = new vector<float>; outputTree->Branch("trk_lep_dZ0", "vector<float>", &trk_lep_dZ0);
    vector<int>* trk_charge = new vector<int>; outputTree->Branch("trk_charge", "vector<int>", &trk_charge);
    vector<float>* chiSquared = new vector<float>; outputTree->Branch("chiSquared", "vector<float>", &chiSquared);
    vector<int>* nIBLHits = new vector<int>; outputTree->Branch("nIBLHits", "vector<int>", &nIBLHits);
    vector<int>* nPixHits = new vector<int>; outputTree->Branch("nPixHits", "vector<int>", &nPixHits);
    vector<int>* nPixHoles = new vector<int>; outputTree->Branch("nPixHoles", "vector<int>", &nPixHoles);
    vector<int>* nPixOutliers = new vector<int>; outputTree->Branch("nPixOutliers", "vector<int>", &nPixOutliers);
    vector<int>* nSCTHits = new vector<int>; outputTree->Branch("nSCTHits", "vector<int>", &nSCTHits);
    vector<int>* nSCTHoles = new vector<int>; outputTree->Branch("nSCTHoles", "vector<int>", &nSCTHoles);
    vector<int>* nTRTHits = new vector<int>; outputTree->Branch("nTRTHits", "vector<int>", &nTRTHits);

    // Event objects
    const xAOD::TrackParticleContainer *tracks;
    const xAOD::VertexContainer *primary_vertices;
    const xAOD::Vertex *primary_vertex;
    const xAOD::ElectronContainer *electrons;
    const xAOD::MuonContainer *muons;
    const xAOD::CaloClusterContainer *calo_clusters;
    vector<const xAOD::TrackParticle*> filtered_tracks;
    vector<pair<const xAOD::Electron*, int>> filtered_electrons;
    vector<pair<const xAOD::Muon*, int>> filtered_muons;

    // Fill branches for one lepton
    SG::AuxElement::ConstAccessor<float> accessPromptVar("PromptLeptonVeto");

    auto process_lepton = [&] (const xAOD::IParticle* lepton, const xAOD::TrackParticle* track_particle, bool is_electron) {

        // retrieve ptcone and etcone variables
        auto process_electron_cones = [&] (const xAOD::Electron* electron) {
            electron->isolation(ptcone20,xAOD::Iso::ptcone20_TightTTVA_pt1000);
            ptcone30 = numeric_limits<float>::quiet_NaN();
            ptcone40 = numeric_limits<float>::quiet_NaN();
            electron->isolation(ptvarcone20,xAOD::Iso::ptvarcone20);
            electron->isolation(ptvarcone30,xAOD::Iso::ptvarcone30_TightTTVA_pt1000);
            electron->isolation(ptvarcone40,xAOD::Iso::ptvarcone40);
            electron->isolation(topoetcone20,xAOD::Iso::topoetcone20);
            topoetcone30 = numeric_limits<float>::quiet_NaN();
            electron->isolation(topoetcone40,xAOD::Iso::topoetcone40);
            electron->isolation(eflowcone20,xAOD::Iso::neflowisol20);
            // topocluster stuff - using https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/IsolationManualCalculation
        };

        auto process_muon_cones = [&] (const xAOD::Muon* muon) {
            muon->isolation(ptcone20,xAOD::Iso::ptcone20);
            muon->isolation(ptcone30,xAOD::Iso::ptcone30);
            muon->isolation(ptcone40,xAOD::Iso::ptcone40);
            muon->isolation(ptvarcone20,xAOD::Iso::ptvarcone20);
            muon->isolation(ptvarcone30,xAOD::Iso::ptvarcone30);
            muon->isolation(ptvarcone40,xAOD::Iso::ptvarcone40);
            muon->isolation(topoetcone20,xAOD::Iso::topoetcone20);
            muon->isolation(topoetcone30,xAOD::Iso::topoetcone30);
            muon->isolation(topoetcone40,xAOD::Iso::topoetcone40);
            muon->isolation(eflowcone20,xAOD::Iso::neflowisol20);
        };

        // retrieve all relevant lepton variables
        lep_pT = lepton->pt();
        lep_eta = lepton->eta();
        lep_theta = 2 * atan(exp(-lep_eta));
        lep_phi = lepton->phi();
        lep_d0 = track_particle->d0();
        lep_d0_over_sigd0 = xAOD::TrackingHelpers::d0significance(track_particle);
        lep_z0 = track_particle->z0();
        lep_dz0 = track_particle->z0() - primary_vertex->z();
        PLT = accessPromptVar(*lepton);
        if (is_electron) process_electron_cones((const xAOD::Electron*)lepton);
        else process_muon_cones((const xAOD::Muon*)lepton);

        // check if lepton passes cuts
        bool dz0_cut = fabs(lep_dz0 * sin(lep_theta)) < 0.5;
        bool d0_over_sigd0_cut = (is_electron and (fabs(lep_d0_over_sigd0) < 5)) or (!is_electron and (d0_over_sigd0_cut = fabs(lep_d0_over_sigd0) < 3));
        bool passes_cuts = (dz0_cut and d0_over_sigd0_cut);
        if (!passes_cuts) return false;

        // get tracks associated to lepton
        auto get_electron_own_tracks = [&] (const xAOD::Electron* electron) {
            set<const xAOD::TrackParticle*> electron_tracks = xAOD::EgammaHelpers::getTrackParticles((const xAOD::Egamma*)electron, true);
            return electron_tracks;
        };

        auto get_muon_own_tracks = [&] (const xAOD::Muon* muon) {
            xAOD::Muon::TrackParticleType type = xAOD::Muon::TrackParticleType::InnerDetectorTrackParticle;
            auto own_track = muon->trackParticle(type);
            set<const xAOD::TrackParticle*> muon_tracks {own_track};
            return muon_tracks;
        };

        // store tracks in dR cone of 0.5
        trk_lep_dR->clear(); trk_pT->clear(); trk_eta->clear(); trk_phi->clear();
        trk_d0->clear(); trk_z0->clear(); trk_charge->clear(); chiSquared->clear();
        trk_lep_dEta->clear(); trk_lep_dPhi->clear(); trk_lep_dD0->clear(); trk_lep_dZ0->clear();
        nIBLHits->clear(); nPixHits->clear(); nPixHoles->clear(); nPixOutliers->clear();
        nSCTHits->clear(); nSCTHoles->clear(); nTRTHits->clear();
        set<const xAOD::TrackParticle*> own_tracks;
        if (is_electron) own_tracks = get_electron_own_tracks((const xAOD::Electron*)lepton);
        else own_tracks = get_muon_own_tracks((const xAOD::Muon*)lepton);

        for (auto track : filtered_tracks) {
            if (!track->vertex() or track->vertex()!=primary_vertex) continue;
            bool matches_own_track = false;
            for (auto own_track : own_tracks)
                if (track == own_track) matches_own_track = true;
            if (matches_own_track) continue;
            float dR = track->p4().DeltaR(lepton->p4());
            if (dR > 0.5) continue; 

            trk_lep_dR->push_back(dR);
            trk_pT->push_back(track->pt());
            trk_eta->push_back(track->eta());
            trk_phi->push_back(track->phi());
            trk_d0->push_back(track->d0());
            trk_z0->push_back(track->z0());
            trk_charge->push_back(track->charge());
            chiSquared->push_back(track->chiSquared());

            trk_lep_dEta->push_back(track->eta() - lep_eta);
            trk_lep_dPhi->push_back(track->phi() - lep_phi);
            trk_lep_dD0->push_back(track->d0() - lep_d0);
            trk_lep_dZ0->push_back(track->z0() - lep_z0);

            uint8_t placeholder;
            track->summaryValue(placeholder, xAOD::numberOfInnermostPixelLayerHits); nIBLHits->push_back(placeholder);
            track->summaryValue(placeholder, xAOD::numberOfPixelHits); nPixHits->push_back(placeholder);
            track->summaryValue(placeholder, xAOD::numberOfPixelHoles); nPixHoles->push_back(placeholder);
            track->summaryValue(placeholder, xAOD::numberOfPixelOutliers); nPixOutliers->push_back(placeholder);
            track->summaryValue(placeholder, xAOD::numberOfSCTHits); nSCTHits->push_back(placeholder);
            track->summaryValue(placeholder, xAOD::numberOfSCTHoles); nSCTHoles->push_back(placeholder);
            track->summaryValue(placeholder, xAOD::numberOfTRTHits); nTRTHits->push_back(placeholder);
        }

        return true;
    };

    // Cutflow table [electron/muon/HF/isolated][truth_type/medium/impact_params]
    int cutflow_table[4][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

    auto update_cutflow = [&] (vector<pair<const xAOD::Electron*, int>> electrons, vector<pair<const xAOD::Muon*, int>> muons, int stage) {
        cutflow_table[0][stage] += electrons.size();
        cutflow_table[1][stage] += muons.size();
        for (auto electron_info : electrons) {
            truth_type = electron_info.second;
            int is_isolated = (truth_type == 2);
            cutflow_table[2+is_isolated][stage]++;
        }
        for (auto muon_info : muons) {
            truth_type = muon_info.second;
            int is_isolated = (truth_type == 3);
            cutflow_table[2+is_isolated][stage]++;
        }
    };

    auto print_cutflow = [&] () {
        cout << "Printing cutflow table:" << endl;
        for (int i=0; i<4; i++) {
            cout << cutflow_table[i][0] << " " << cutflow_table[i][1] << " " << cutflow_table[i][2] << endl;
        }
    };

    // Loop over entries
    int entries = event.getEntries();
    cout << "Retrieved " << entries << " events" << endl;
    //entries = 10000;
    cout << "\nProcessing leptons" << endl;
    for (entry_n = 0; entry_n < entries; ++entry_n) {

        // Print some status
        if ( ! (entry_n % 500)) {
            cout << "Processing event " << entry_n << "/" << entries << "\n";
        }

        // Load the event
        bool ok = event.getEntry(entry_n) >= 0;
        if (!ok) throw logic_error("getEntry failed");

        // Get event objects
        RETURN_CHECK(ALG, event.retrieve(tracks, "InDetTrackParticles"));
        RETURN_CHECK(ALG, event.retrieve(primary_vertices, "PrimaryVertices"));
        primary_vertex = primary_vertices->at(0);
        RETURN_CHECK(ALG, event.retrieve(electrons, "Electrons"));
        RETURN_CHECK(ALG, event.retrieve(muons, "Muons"));
        RETURN_CHECK(ALG, event.retrieve(calo_clusters, "CaloCalTopoClusters"));

        // Filter objects
        filtered_tracks = object_filters.filter_tracks(tracks, primary_vertex);
        filtered_electrons = object_filters.filter_electrons_truth_type(electrons);
        filtered_muons = object_filters.filter_muons_truth_type(muons);
        update_cutflow(filtered_electrons, filtered_muons, 0);
        filtered_electrons = object_filters.filter_electrons_medium(filtered_electrons);
        filtered_muons = object_filters.filter_muons_medium(filtered_muons);
        update_cutflow(filtered_electrons, filtered_muons, 1);

        // Write event
        vector<pair<const xAOD::Electron*, int>> new_filtered_electrons;
        vector<pair<const xAOD::Muon*, int>> new_filtered_muons;
        for (auto electron_info : filtered_electrons) {
            const xAOD::Electron* electron = electron_info.first;
            truth_type = electron_info.second;
            pdgID = 11;
            if (!process_lepton(electron, electron->trackParticle(), true)) continue;
            new_filtered_electrons.push_back(electron_info);
            outputTree->Fill();
        }
        for (auto muon_info : filtered_muons) {
            const xAOD::Muon* muon = muon_info.first;
            truth_type = muon_info.second;
            pdgID = 13;
            if (!process_lepton(muon, muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle), false)) continue;
            new_filtered_muons.push_back(muon_info);
            outputTree->Fill();
        }
        update_cutflow(new_filtered_electrons, new_filtered_muons, 2);
    }

    // Print # leptons passing each step
    print_cutflow();

    outputTree->Write();
    outputFile.Close();

    return 0;
}
