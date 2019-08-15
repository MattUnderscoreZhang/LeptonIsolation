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
#include "xAODTruth/xAODTruthHelpers.h"
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

int main (int argc, char *argv[]) {

    // parse input
    const char* ALG = argv[0];
    std::string inputFilename = argv[1];

    // object filters
    ObjectFilters object_filters;

    // Open the file:
    std::unique_ptr<TFile> ifile(TFile::Open(inputFilename.c_str(), "READ"));
    if ( ! ifile.get() || ifile->IsZombie()) {
        throw std::logic_error("Couldn't open file: " + inputFilename);
        return 1;
    }
    std::cout << "Opened file: " << inputFilename << std::endl;

    // Connect the event object to it:
    RETURN_CHECK(ALG, xAOD::Init());
    xAOD::TEvent event(xAOD::TEvent::kClassAccess);
    RETURN_CHECK(ALG, event.readFrom(ifile.get()));

    // output ROOT file
    TString outputFilename = "output.root";
    TFile outputFile(outputFilename, "recreate");
    TTree* outputTree = new TTree("BaselineTree", "baseline tree");

    int entry_n; outputTree->Branch("event_n", &entry_n, "event_n/I");
    int pdgID; outputTree->Branch("pdgID", &pdgID, "pdgID/I");
    float pT; outputTree->Branch("pT", &pT, "pT/F");
    float eta; outputTree->Branch("eta", &eta, "eta/F");
    float phi; outputTree->Branch("phi", &phi, "phi/F");
    float d0; outputTree->Branch("d0", &d0, "d0/F");
    float d0_over_sigd0; outputTree->Branch("d0_over_sigd0", &d0_over_sigd0, "d0_over_sigd0/F");
    float z0; outputTree->Branch("z0", &z0, "z0/F");
    float dz0; outputTree->Branch("dz0", &dz0, "dz0/F");
    float ref_ptcone20; outputTree->Branch("ref_ptcone20", &ref_ptcone20, "ref_ptcone20/F");
    float ref_ptcone30; outputTree->Branch("ref_ptcone30", &ref_ptcone30, "ref_ptcone30/F");
    float ref_ptcone40; outputTree->Branch("ref_ptcone40", &ref_ptcone40, "ref_ptcone40/F");
    float ref_ptvarcone20; outputTree->Branch("ref_ptvarcone20", &ref_ptvarcone20, "ref_ptvarcone20/F");
    float ref_ptvarcone30; outputTree->Branch("ref_ptvarcone30", &ref_ptvarcone30, "ref_ptvarcone30/F");
    float ref_ptvarcone40; outputTree->Branch("ref_ptvarcone40", &ref_ptvarcone40, "ref_ptvarcone40/F");
    float ref_topoetcone20; outputTree->Branch("ref_topoetcone20", &ref_topoetcone20, "ref_topoetcone20/F");
    float ref_topoetcone30; outputTree->Branch("ref_topoetcone30", &ref_topoetcone30, "ref_topoetcone30/F");
    float ref_topoetcone40; outputTree->Branch("ref_topoetcone40", &ref_topoetcone40, "ref_topoetcone40/F");
    float ref_eflowcone20; outputTree->Branch("ref_eflowcone20", &ref_eflowcone20, "ref_eflowcone20/F");
    float PLT; outputTree->Branch("PLT", &PLT, "PLT/F");
    int truth_type; outputTree->Branch("truth_type", &truth_type, "truth_type/I");

    // Loop over input events:
    int entries = event.getEntries();
    std::cout << "got " << entries << " entries" << std::endl;
    for (entry_n = 0; entry_n < entries; ++entry_n) {

        // Print some status
        if ( ! (entry_n % 500)) {
            std::cout << "Processing " << entry_n << "/" << entries << "\n";
        }

        // Load the event
        bool ok = event.getEntry(entry_n) >= 0;
        if (!ok) throw std::logic_error("getEntry failed");

        // Get tracks and leptons
        const xAOD::TrackParticleContainer *tracks = 0;
        RETURN_CHECK(ALG, event.retrieve(tracks, "InDetTrackParticles"));
        const xAOD::VertexContainer *primary_vertices = 0;
        RETURN_CHECK(ALG, event.retrieve(primary_vertices, "PrimaryVertices"));
        const xAOD::Vertex *primary_vertex = primary_vertices->at(0);
        const xAOD::ElectronContainer *electrons = 0;
        RETURN_CHECK(ALG, event.retrieve(electrons, "Electrons"));
        const xAOD::MuonContainer *muons = 0;
        RETURN_CHECK(ALG, event.retrieve(muons, "Muons"));
        const xAOD::CaloClusterContainer *calo_clusters;
        RETURN_CHECK(ALG, event.retrieve(calo_clusters, "CaloCalTopoClusters"));

        // Filter objects
        std::vector<const xAOD::TrackParticle*> filtered_tracks = object_filters.filter_tracks(tracks, primary_vertex);
        std::vector<const xAOD::Muon*> filtered_muons = object_filters.filter_muons(muons);
        std::vector<const xAOD::Electron*> filtered_electrons = object_filters.filter_electrons(electrons);

        // Write event
        SG::AuxElement::ConstAccessor<float> accessPromptVar("PromptLeptonVeto");

        for (auto electron : filtered_electrons) {
            truth_type = xAOD::TruthHelpers::getParticleTruthType(*electron); // 2 = real prompt, 3 = HF
            if (truth_type != 2 && truth_type != 3) continue;

            pdgID = 11;
            pT = electron->pt();
            eta = electron->eta();
            phi = electron->phi();
            d0 = electron->trackParticle()->d0();
            d0_over_sigd0 = xAOD::TrackingHelpers::d0significance(electron->trackParticle());
            z0 = electron->trackParticle()->z0();
            dz0 = electron->trackParticle()->z0() - primary_vertex->z();
            electron->isolation(ref_ptcone20,xAOD::Iso::ptcone20_TightTTVA_pt1000);
            ref_ptcone30 = std::numeric_limits<float>::quiet_NaN();
            ref_ptcone40 = std::numeric_limits<float>::quiet_NaN();
            //calc_ptcone20 = 0; calc_ptcone30 = 0; calc_ptcone40 = 0; calc_ptvarcone20 = 0; calc_ptvarcone30 = 0; calc_ptvarcone40 = 0;
            //float var_R_20 = std::min(10e3/electron->pt(), 0.20); float var_R_30 = std::min(10e3/electron->pt(), 0.30); float var_R_40 = std::min(10e3/electron->pt(), 0.40);
            //std::set<const xAOD::TrackParticle*> electron_tracks = xAOD::EgammaHelpers::getTrackParticles((const xAOD::Egamma*)electron, true);
            //for (auto trk : filtered_tracks) {
                //if (!trk) continue;
                //bool matches_own_track = false;
                //for (auto own_track : electron_tracks)
                    //if (trk == own_track) matches_own_track = true;
                //if (matches_own_track) continue;
                //if (trk->vertex() && trk->vertex()!=primary_vertex) continue;
                //if (trk->p4().DeltaR(electron->p4()) < 0.20) calc_ptcone20 += trk->pt();
                //if (trk->p4().DeltaR(electron->p4()) < 0.30) calc_ptcone30 += trk->pt();
                //if (trk->p4().DeltaR(electron->p4()) < 0.40) calc_ptcone40 += trk->pt();
                //if (trk->p4().DeltaR(electron->p4()) < var_R_20) calc_ptvarcone20 += trk->pt();
                //if (trk->p4().DeltaR(electron->p4()) < var_R_30) calc_ptvarcone30 += trk->pt();
                //if (trk->p4().DeltaR(electron->p4()) < var_R_40) calc_ptvarcone40 += trk->pt();
            //}
            electron->isolation(ref_ptvarcone20,xAOD::Iso::ptvarcone20);
            electron->isolation(ref_ptvarcone30,xAOD::Iso::ptvarcone30_TightTTVA_pt1000);
            electron->isolation(ref_ptvarcone40,xAOD::Iso::ptvarcone40);
            electron->isolation(ref_topoetcone20,xAOD::Iso::topoetcone20);
            ref_topoetcone30 = std::numeric_limits<float>::quiet_NaN();
            electron->isolation(ref_topoetcone40,xAOD::Iso::topoetcone40);
            //calc_topoetcone20 = 0; calc_topoetcone30 = 0; calc_topoetcone40 = 0;
            //const xAOD::CaloCluster *egclus = this->m_current_electrons.at(idx)->caloCluster();
            //for (const auto& clus : *calo_clusters) {
                ////if (clus->e()<0) continue;
                //if (egclus->p4().DeltaR(clus->p4()) < 0.2) calc_etcone20 += clus->et();
                //if (egclus->p4().DeltaR(clus->p4()) < 0.3) calc_etcone30 += clus->et();
                //if (egclus->p4().DeltaR(clus->p4()) < 0.4) calc_etcone40 += clus->et();
            //}
            electron->isolation(ref_eflowcone20,xAOD::Iso::neflowisol20);
            PLT = accessPromptVar(*electron);

            outputTree->Fill();
        }

        for (auto muon : filtered_muons) {
            truth_type = xAOD::TruthHelpers::getParticleTruthType(*(muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle))); // 2 = real prompt, 3 = HF
            if (truth_type != 2 && truth_type != 3) continue;

            pdgID = 13;
            pT = muon->pt();
            eta = muon->eta();
            phi = muon->phi();
            d0 = muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle)->d0();
            d0_over_sigd0 = xAOD::TrackingHelpers::d0significance(muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle));
            z0 = muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle)->z0();
            dz0 = muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle)->z0() - primary_vertex->z();
            muon->isolation(ref_ptcone20,xAOD::Iso::ptcone20);
            muon->isolation(ref_ptcone30,xAOD::Iso::ptcone30);
            muon->isolation(ref_ptcone40,xAOD::Iso::ptcone40);
            //calc_ptcone20 = 0; calc_ptcone30 = 0; calc_ptcone40 = 0; calc_ptvarcone20 = 0; calc_ptvarcone30 = 0; calc_ptvarcone40 = 0;
            //float var_R_20 = std::min(10e3/muon->pt(), 0.20); float var_R_30 = std::min(10e3/muon->pt(), 0.30); float var_R_40 = std::min(10e3/muon->pt(), 0.40);
            //xAOD::Muon::TrackParticleType type = xAOD::Muon::TrackParticleType::InnerDetectorTrackParticle;
            //auto own_track = this_muon->trackParticle(type);
            //for (auto trk : filtered_tracks) {
                //if (!trk) continue;
                //if (trk == own_track) continue;
                //if (trk->p4().DeltaR(muon->p4()) < 0.20) calc_ptcone20 += trk->pt();
                //if (trk->p4().DeltaR(muon->p4()) < 0.30) calc_ptcone30 += trk->pt();
                //if (trk->p4().DeltaR(muon->p4()) < 0.40) calc_ptcone40 += trk->pt();
                //if (trk->p4().DeltaR(muon->p4()) < var_R_20) calc_ptvarcone20 += trk->pt();
                //if (trk->p4().DeltaR(muon->p4()) < var_R_30) calc_ptvarcone30 += trk->pt();
                //if (trk->p4().DeltaR(muon->p4()) < var_R_40) calc_ptvarcone40 += trk->pt();
            //}
            muon->isolation(ref_ptvarcone20,xAOD::Iso::ptvarcone20);
            muon->isolation(ref_ptvarcone30,xAOD::Iso::ptvarcone30);
            muon->isolation(ref_ptvarcone40,xAOD::Iso::ptvarcone40);
            muon->isolation(ref_topoetcone20,xAOD::Iso::topoetcone20);
            muon->isolation(ref_topoetcone30,xAOD::Iso::topoetcone30);
            muon->isolation(ref_topoetcone40,xAOD::Iso::topoetcone40);
            //calc_topoetcone20 = 0; calc_topoetcone30 = 0; calc_topoetcone40 = 0;
            //std::vector<fastjet::PseudoJet> input_clus;
            //for (const auto& cluster : *calo_clusters) {
                //if (!cluster) continue;
                //if (cluster->e()<0) continue;
                //float dR = cluster->p4().DeltaR(muon->p4());
                //if (dR < 0.2 && dR > 0.05) calc_etcone20 += cluster->et();
                //if (dR < 0.3 && dR > 0.05) calc_etcone30 += cluster->et();
                //if (dR < 0.4 && dR > 0.05) calc_etcone40 += cluster->et();
            //}
            muon->isolation(ref_eflowcone20,xAOD::Iso::neflowisol20);
            PLT = accessPromptVar(*muon);

            outputTree->Fill();
        }
    }

    outputTree->Write();
    outputFile.Close();

    return 0;
}
