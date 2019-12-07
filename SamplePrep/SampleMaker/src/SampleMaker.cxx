// local tools
#include "ObjectFilters.cxx"
#include "TFile.h"
#include "TChain.h"
#include "TTree.h"
#include "TLeaf.h"
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

// stl includes
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <cassert>

using namespace std;

int main (int argc, char *argv[]) {

    // Object filters
    ObjectFilters object_filters;

    // Parse input - split input TFile names by ','
    const char* ALG = argv[0];
    string inputFilenames = argv[1];

    std::vector<std::string> fileList;
    for (size_t i=0,n; i <= inputFilenames.length(); i=n+1)
    {
        n = inputFilenames.find_first_of(',',i);
        if (n == std::string::npos)
            n = inputFilenames.length();
        string tmp = inputFilenames.substr(i,n-i);
        fileList.push_back(tmp);
    }

    TChain* fChain = new TChain("CollectionTree");
    for (unsigned int iFile=0; iFile<fileList.size(); ++iFile)
    {
        cout << "Opened file: " << fileList[iFile].c_str() << endl;
        fChain->Add(fileList[iFile].c_str());
    }

    // Connect the event object to input files
    RETURN_CHECK(ALG, xAOD::Init());
    xAOD::TEvent event(xAOD::TEvent::kClassAccess);
    RETURN_CHECK(ALG, event.readFrom(fChain));

    // Leptons
    TFile outputFile("output.root", "recreate");
    TTree* unnormedTree = new TTree("UnnormedTree", "unnormalized tree");

    int entry_n; unnormedTree->Branch("event_n", &entry_n, "event_n/I");
    int pdgID; unnormedTree->Branch("pdgID", &pdgID, "pdgID/I");
    int truth_type; unnormedTree->Branch("truth_type", &truth_type, "truth_type/I");

    float ptcone20; unnormedTree->Branch("ptcone20", &ptcone20, "ptcone20/F");
    float ptcone30; unnormedTree->Branch("ptcone30", &ptcone30, "ptcone30/F");
    float ptcone40; unnormedTree->Branch("ptcone40", &ptcone40, "ptcone40/F");
    float ptvarcone20; unnormedTree->Branch("ptvarcone20", &ptvarcone20, "ptvarcone20/F");
    float ptvarcone30; unnormedTree->Branch("ptvarcone30", &ptvarcone30, "ptvarcone30/F");
    float ptvarcone40; unnormedTree->Branch("ptvarcone40", &ptvarcone40, "ptvarcone40/F");
    float topoetcone20; unnormedTree->Branch("topoetcone20", &topoetcone20, "topoetcone20/F");
    float topoetcone30; unnormedTree->Branch("topoetcone30", &topoetcone30, "topoetcone30/F");
    float topoetcone40; unnormedTree->Branch("topoetcone40", &topoetcone40, "topoetcone40/F");
    float eflowcone20; unnormedTree->Branch("eflowcone20", &eflowcone20, "eflowcone20/F");
    float PLT; unnormedTree->Branch("PLT", &PLT, "PLT/F");

    float lep_pT; unnormedTree->Branch("lep_pT", &lep_pT, "lep_pT/F");
    float lep_eta; unnormedTree->Branch("lep_eta", &lep_eta, "lep_eta/F");
    float lep_theta; unnormedTree->Branch("lep_theta", &lep_theta, "lep_theta/F");
    float lep_phi; unnormedTree->Branch("lep_phi", &lep_phi, "lep_phi/F");
    float lep_d0; unnormedTree->Branch("lep_d0", &lep_d0, "lep_d0/F");
    float lep_d0_over_sigd0; unnormedTree->Branch("lep_d0_over_sigd0", &lep_d0_over_sigd0, "lep_d0_over_sigd0/F");
    float lep_z0; unnormedTree->Branch("lep_z0", &lep_z0, "lep_z0/F");
    float lep_dz0; unnormedTree->Branch("lep_dz0", &lep_dz0, "lep_dz0/F");

    vector<float>* trk_lep_dR = new vector<float>; unnormedTree->Branch("trk_lep_dR", "vector<float>", &trk_lep_dR);
    vector<float>* trk_pT = new vector<float>; unnormedTree->Branch("trk_pT", "vector<float>", &trk_pT);
    vector<float>* trk_eta = new vector<float>; unnormedTree->Branch("trk_eta", "vector<float>", &trk_eta);
    vector<float>* trk_phi = new vector<float>; unnormedTree->Branch("trk_phi", "vector<float>", &trk_phi);
    vector<float>* trk_d0 = new vector<float>; unnormedTree->Branch("trk_d0", "vector<float>", &trk_d0);
    vector<float>* trk_z0 = new vector<float>; unnormedTree->Branch("trk_z0", "vector<float>", &trk_z0);
    vector<float>* trk_lep_dEta = new vector<float>; unnormedTree->Branch("trk_lep_dEta", "vector<float>", &trk_lep_dEta);
    vector<float>* trk_lep_dPhi = new vector<float>; unnormedTree->Branch("trk_lep_dPhi", "vector<float>", &trk_lep_dPhi);
    vector<float>* trk_lep_dD0 = new vector<float>; unnormedTree->Branch("trk_lep_dD0", "vector<float>", &trk_lep_dD0);
    vector<float>* trk_lep_dZ0 = new vector<float>; unnormedTree->Branch("trk_lep_dZ0", "vector<float>", &trk_lep_dZ0);
    vector<float>* trk_chi2 = new vector<float>; unnormedTree->Branch("trk_chi2", "vector<float>", &trk_chi2);
    vector<int>* trk_charge = new vector<int>; unnormedTree->Branch("trk_charge", "vector<int>", &trk_charge);
    vector<int>* trk_nIBLHits = new vector<int>; unnormedTree->Branch("trk_nIBLHits", "vector<int>", &trk_nIBLHits);
    vector<int>* trk_nPixHits = new vector<int>; unnormedTree->Branch("trk_nPixHits", "vector<int>", &trk_nPixHits);
    vector<int>* trk_nPixHoles = new vector<int>; unnormedTree->Branch("trk_nPixHoles", "vector<int>", &trk_nPixHoles);
    vector<int>* trk_nPixOutliers = new vector<int>; unnormedTree->Branch("trk_nPixOutliers", "vector<int>", &trk_nPixOutliers);
    vector<int>* trk_nSCTHits = new vector<int>; unnormedTree->Branch("trk_nSCTHits", "vector<int>", &trk_nSCTHits);
    vector<int>* trk_nSCTHoles = new vector<int>; unnormedTree->Branch("trk_nSCTHoles", "vector<int>", &trk_nSCTHoles);
    vector<int>* trk_nTRTHits = new vector<int>; unnormedTree->Branch("trk_nTRTHits", "vector<int>", &trk_nTRTHits);

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
        bool d0_over_sigd0_cut = (is_electron and (fabs(lep_d0_over_sigd0) < 5)) or (!is_electron and (fabs(lep_d0_over_sigd0) < 3));
        bool passes_cuts = (dz0_cut and d0_over_sigd0_cut);
        if (!passes_cuts) return false;

        // store tracks associated to lepton in dR cone of 0.5
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

        trk_lep_dR->clear(); trk_pT->clear(); trk_eta->clear(); trk_phi->clear();
        trk_d0->clear(); trk_z0->clear(); trk_charge->clear(); trk_chi2->clear();
        trk_lep_dEta->clear(); trk_lep_dPhi->clear(); trk_lep_dD0->clear(); trk_lep_dZ0->clear();
        trk_nIBLHits->clear(); trk_nPixHits->clear(); trk_nPixHoles->clear(); trk_nPixOutliers->clear();
        trk_nSCTHits->clear(); trk_nSCTHoles->clear(); trk_nTRTHits->clear();
        set<const xAOD::TrackParticle*> own_tracks;
        if (is_electron) own_tracks = get_electron_own_tracks((const xAOD::Electron*)lepton);
        else own_tracks = get_muon_own_tracks((const xAOD::Muon*)lepton);

        bool has_associated_tracks = false;
        for (auto track : filtered_tracks) {
            if (!track->vertex() or track->vertex()!=primary_vertex) continue;
            bool matches_own_track = false;
            for (auto own_track : own_tracks)
                if (track == own_track) matches_own_track = true;
            if (matches_own_track) continue;
            float dR = track->p4().DeltaR(lepton->p4());
            if (dR > 0.5) continue; 

            has_associated_tracks = true;

            trk_lep_dR->push_back(dR);
            trk_pT->push_back(track->pt());
            trk_eta->push_back(track->eta());
            trk_phi->push_back(track->phi());
            trk_d0->push_back(track->d0());
            trk_z0->push_back(track->z0());
            trk_charge->push_back(track->charge());
            trk_chi2->push_back(track->chiSquared());

            trk_lep_dEta->push_back(track->eta() - lep_eta);
            trk_lep_dPhi->push_back(track->phi() - lep_phi);
            trk_lep_dD0->push_back(track->d0() - lep_d0);
            trk_lep_dZ0->push_back(track->z0() - lep_z0);

            uint8_t placeholder;
            track->summaryValue(placeholder, xAOD::numberOfInnermostPixelLayerHits); trk_nIBLHits->push_back(placeholder);
            track->summaryValue(placeholder, xAOD::numberOfPixelHits); trk_nPixHits->push_back(placeholder);
            track->summaryValue(placeholder, xAOD::numberOfPixelHoles); trk_nPixHoles->push_back(placeholder);
            track->summaryValue(placeholder, xAOD::numberOfPixelOutliers); trk_nPixOutliers->push_back(placeholder);
            track->summaryValue(placeholder, xAOD::numberOfSCTHits); trk_nSCTHits->push_back(placeholder);
            track->summaryValue(placeholder, xAOD::numberOfSCTHoles); trk_nSCTHoles->push_back(placeholder);
            track->summaryValue(placeholder, xAOD::numberOfTRTHits); trk_nTRTHits->push_back(placeholder);
        }

        // remove leptons with no associated tracks
        return has_associated_tracks;
    };

    // Cutflow table [HF_electron/isolated_electron/HF_muon/isolated_muon][truth_type/medium/impact_params/isolation]
    int cutflow_table[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    auto update_cutflow = [&] (vector<pair<const xAOD::Electron*, int>> electrons, vector<pair<const xAOD::Muon*, int>> muons, int stage) {
        //cout << "Stage " << stage << " " << electrons.size() << " " << muons.size() << endl;
        for (auto electron_info : electrons) {
            truth_type = electron_info.second;
            int is_isolated = (truth_type == 2);
            cutflow_table[is_isolated][stage]++;
        }
        for (auto muon_info : muons) {
            truth_type = muon_info.second;
            int is_isolated = (truth_type == 6);
            cutflow_table[2+is_isolated][stage]++;
        }
    };

    auto print_cutflow = [&] () {
        cout << "Printing cutflow table:" << endl;
        for (int i=0; i<4; i++) {
            cout << cutflow_table[i][0] << " " << cutflow_table[i][1] << " " << cutflow_table[i][2] << " " << cutflow_table[i][3] << endl;
        }
    };

    // Loop over entries
    int entries = event.getEntries();
    cout << "\nReading input files" << endl;
    cout << "Retrieved " << entries << " events" << endl;
    //entries = 1000;
    cout << "\nProcessing leptons" << endl;
    for (entry_n = 0; entry_n < entries; ++entry_n) {

        // Get event
        if (entry_n%500 == 0) cout << "Processing event " << entry_n << "/" << entries << "\n";
        event.getEntry(entry_n);

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
            unnormedTree->Fill();
        }
        for (auto muon_info : filtered_muons) {
            const xAOD::Muon* muon = muon_info.first;
            truth_type = muon_info.second;
            pdgID = 13;
            if (!process_lepton(muon, muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle), false)) continue;
            new_filtered_muons.push_back(muon_info);
            unnormedTree->Fill();
        }
        update_cutflow(new_filtered_electrons, new_filtered_muons, 2);

        // Additional cutflow step - what passes isolation cut?
        vector<pair<const xAOD::Electron*, int>> isolated_filtered_electrons;
        vector<pair<const xAOD::Muon*, int>> isolated_filtered_muons;
        float temp_ptvarcone40;
        float temp_lep_pT;
        for (auto electron_info : new_filtered_electrons) {
            const xAOD::Electron* electron = electron_info.first;
            electron->isolation(temp_ptvarcone40,xAOD::Iso::ptvarcone40);
            temp_lep_pT = electron->pt();
            //cout << temp_ptvarcone40 << " " << temp_lep_pT << " " << temp_ptvarcone40/temp_lep_pT << endl;
            if (temp_ptvarcone40/temp_lep_pT < 0.1)
                isolated_filtered_electrons.push_back(electron_info);
        }
        for (auto muon_info : new_filtered_muons) {
            const xAOD::Muon* muon = muon_info.first;
            muon->isolation(temp_ptvarcone40,xAOD::Iso::ptvarcone40);
            temp_lep_pT = muon->pt();
            //cout << temp_ptvarcone40 << " " << temp_lep_pT << " " << temp_ptvarcone40/temp_lep_pT << endl;
            if (temp_ptvarcone40/temp_lep_pT < 0.1)
                isolated_filtered_muons.push_back(muon_info);
        }
        update_cutflow(isolated_filtered_electrons, isolated_filtered_muons, 3);
    }

    cout << "\n" << endl;
    print_cutflow(); // Print # leptons passing each step
    unnormedTree->Write();

    // Create normalized tree
    cout << "\nCreating normalized tree" << endl;
    TTree* normalizedTree = new TTree("NormalizedTree", "normalized tree");
    TObjArray* myBranches = (TObjArray*)(unnormedTree->GetListOfBranches())->Clone();
    myBranches->SetOwner(kFALSE);

    for (int i=0; i<myBranches->GetEntries(); i++) {

        // Get branch
        string currentBranchName = myBranches->At(i)->GetName();
        TBranch* currentBranch = unnormedTree->GetBranch((TString)currentBranchName);

        // Find data type of branch
        TClass* branchClass; EDataType branchType;
        currentBranch->GetExpectedType(branchClass, branchType);
        string varType;
        if (branchType == -1)
            varType = currentBranch->GetClassName();
        else if (branchType == 3)
            varType = "I";
        else if (branchType == 5)
            varType = "F";
        else {
            cout << "Unrecognized branch type" << endl;
            exit(0);
        }

        // Get branch mean and RMS
        TH1F* histo = new TH1F("histo", "", 1, -1000000, 100000);
        unnormedTree->Draw((currentBranchName + ">>histo").c_str());
        float branchMean = histo->GetMean();
        float branchRMS = histo->GetRMS();
        delete histo;
        if (currentBranchName.rfind("lep_",0)!=0 && currentBranchName.rfind("trk_",0)!=0) {
            // don't normalize branches that don't start with lep_ or trk_
            branchMean = 0;
            branchRMS = 1;
        }

        // Fill tree with normalized branch
        unnormedTree->SetBranchStatus(currentBranchName.c_str(), 1);
        int intVar; float floatVar;
        vector<int>* intVecVar = new vector<int>; vector<float>* floatVecVar = new vector<float>;
        auto fillNonVecBranch = [&] (auto branchVar) {
            unnormedTree->SetBranchAddress(currentBranchName.c_str(), &branchVar);
            float newFloatVar;
            normalizedTree->Branch(currentBranchName.c_str(), &newFloatVar, (currentBranchName+"/F").c_str());
            Long64_t nentries = unnormedTree->GetEntries();
            for (Long64_t i=0; i<nentries; i++) {
                unnormedTree->GetEntry(i);
                newFloatVar = (branchVar-branchMean) / branchRMS;
                normalizedTree->Fill();
            }
        };
        auto fillVecBranch = [&] (auto branchVar) {
            unnormedTree->SetBranchAddress(currentBranchName.c_str(), &branchVar);
            vector<float>* newFloatVecVar = new vector<float>;
            normalizedTree->Branch(currentBranchName.c_str(), "vector<float>", &newFloatVecVar);
            Long64_t nentries = unnormedTree->GetEntries();
            for (int i=0; i<nentries; i++) {
                unnormedTree->GetEntry(i);
                newFloatVecVar->clear();
                for (int i=0; i<branchVar->size(); i++)
                    newFloatVecVar->push_back((branchVar->at(i)-branchMean) / branchRMS);
                normalizedTree->Fill();
            }
        };
        if (varType == "I") fillNonVecBranch(intVar);
        else if (varType == "F") fillNonVecBranch(floatVar);
        else if (varType == "vector<int>") fillVecBranch(intVecVar);
        else if (varType == "vector<float>") fillVecBranch(floatVecVar);
    }
    normalizedTree->Write();

    outputFile.Close();
    delete fChain;

    return 0;
}
