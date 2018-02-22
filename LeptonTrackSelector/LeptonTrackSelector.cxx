#include "SusySkim2LJetsLegacy/LeptonTrackSelector.h"
#include <iostream>


// ------------------------------------------------------------------------------------------ //
LeptonTrackSelector::LeptonTrackSelector() : BaseUser("SusySkim2LJetsLegacy","LeptonTrackSelector")
{

}
// ------------------------------------------------------------------------------------------ //
void LeptonTrackSelector::setup(ConfigMgr*& configMgr)
{

    // Lepton variables
    configMgr->treeMaker->addVecIntVariable("lepIso_lep_q");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_pt");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_eta");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_phi");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_m");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_d0");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_z0");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_d0Err");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_z0Err");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_pTErr");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_ptcone20");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_ptcone30");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_ptcone40");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_topoetcone20");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_topoetcone30");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_topoetcone40");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_ptvarcone20");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_ptvarcone30");
    configMgr->treeMaker->addVecFloatVariable("lepIso_lep_ptvarcone40");
    configMgr->treeMaker->addVecIntVariable("lepIso_lep_truthAuthor");
    configMgr->treeMaker->addVecIntVariable("lepIso_lep_truthType");
    configMgr->treeMaker->addVecIntVariable("lepIso_lep_truthOrigin");

    // Track variables
    configMgr->treeMaker->addVecIntVariable("lepIso_track_q");
    configMgr->treeMaker->addVecFloatVariable("lepIso_track_pt");
    configMgr->treeMaker->addVecFloatVariable("lepIso_track_eta");
    configMgr->treeMaker->addVecFloatVariable("lepIso_track_phi");
    configMgr->treeMaker->addVecFloatVariable("lepIso_track_m");
    configMgr->treeMaker->addVecFloatVariable("lepIso_track_fitQuality");
    configMgr->treeMaker->addVecFloatVariable("lepIso_track_d0");
    configMgr->treeMaker->addVecFloatVariable("lepIso_track_z0");
    configMgr->treeMaker->addVecFloatVariable("lepIso_track_d0Err");
    configMgr->treeMaker->addVecFloatVariable("lepIso_track_z0Err");
    configMgr->treeMaker->addVecIntVariable("lepIso_track_nIBLHits");
    configMgr->treeMaker->addVecIntVariable("lepIso_track_nPixHits");
    configMgr->treeMaker->addVecIntVariable("lepIso_track_nPixHoles");
    configMgr->treeMaker->addVecIntVariable("lepIso_track_nPixOutliers");
    configMgr->treeMaker->addVecIntVariable("lepIso_track_nSCTHits");
    configMgr->treeMaker->addVecIntVariable("lepIso_track_nTRTHits");

    // Make a cutflow stream
    configMgr->cutflow->defineCutFlow("cutFlow",configMgr->treeMaker->getFile("tree"));


    // Object class contains the definitions of all physics objects, eg muons, electrons, jets
    // See SusySkimMaker::Objects for available methods; configMgr->obj

}
// ------------------------------------------------------------------------------------------ //
bool LeptonTrackSelector::doAnalysis(ConfigMgr*& configMgr)
{

    /*
       This is the main method, which is called for each event
       */

    // Skims events by imposing any cuts you define in this method below
    if( !passCuts(configMgr) ) return false;

    // Fill lepton variables
    std::vector<int> lep_q;
    std::vector<float> lep_pt;
    std::vector<float> lep_eta;
    std::vector<float> lep_phi;
    std::vector<float> lep_m;
    std::vector<float> lep_d0;
    std::vector<float> lep_z0;
    std::vector<float> lep_d0Err;
    std::vector<float> lep_z0Err;
    std::vector<float> lep_pTErr;
    std::vector<float> lep_ptcone20;
    std::vector<float> lep_ptcone30;
    std::vector<float> lep_ptcone40;
    std::vector<float> lep_topoetcone20;
    std::vector<float> lep_topoetcone30;
    std::vector<float> lep_topoetcone40;
    std::vector<float> lep_ptvarcone20;
    std::vector<float> lep_ptvarcone30;
    std::vector<float> lep_ptvarcone40;
    std::vector<int> lep_truthAuthor;
    std::vector<int> lep_truthType;
    std::vector<int> lep_truthOrigin;

    for (unsigned i = 0; i < configMgr->obj->electrons.size(); i++) {
        lep_q.push_back(configMgr->obj->electrons[i]->q);
        lep_pt.push_back(configMgr->obj->electrons[i]->Pt());
        lep_eta.push_back(configMgr->obj->electrons[i]->Eta());
        lep_phi.push_back(configMgr->obj->electrons[i]->Phi());
        lep_m.push_back(configMgr->obj->electrons[i]->M());
        lep_d0.push_back(configMgr->obj->electrons[i]->d0);
        lep_z0.push_back(configMgr->obj->electrons[i]->z0);
        lep_d0Err.push_back(configMgr->obj->electrons[i]->d0Err);
        lep_z0Err.push_back(configMgr->obj->electrons[i]->z0Err);
        lep_pTErr.push_back(configMgr->obj->electrons[i]->pTErr);
        lep_ptcone20.push_back(configMgr->obj->electrons[i]->ptcone20);
        lep_ptcone30.push_back(configMgr->obj->electrons[i]->ptcone30);
        lep_ptcone40.push_back(configMgr->obj->electrons[i]->ptcone40);
        lep_topoetcone20.push_back(configMgr->obj->electrons[i]->topoetcone20);
        lep_topoetcone30.push_back(configMgr->obj->electrons[i]->topoetcone30);
        lep_topoetcone40.push_back(configMgr->obj->electrons[i]->topoetcone40);
        lep_ptvarcone20.push_back(configMgr->obj->electrons[i]->ptvarcone20);
        lep_ptvarcone30.push_back(configMgr->obj->electrons[i]->ptvarcone30);
        lep_ptvarcone40.push_back(configMgr->obj->electrons[i]->ptvarcone40);
        lep_truthAuthor.push_back(configMgr->obj->electrons[i]->author);
        lep_truthType.push_back(configMgr->obj->electrons[i]->type);
        lep_truthOrigin.push_back(configMgr->obj->electrons[i]->origin);
    }

    for (unsigned i = 0; i < configMgr->obj->muons.size(); i++) {
        lep_q.push_back(configMgr->obj->muons[i]->q);
        lep_pt.push_back(configMgr->obj->muons[i]->Pt());
        lep_eta.push_back(configMgr->obj->muons[i]->Eta());
        lep_phi.push_back(configMgr->obj->muons[i]->Phi());
        lep_m.push_back(configMgr->obj->muons[i]->M());
        lep_d0.push_back(configMgr->obj->muons[i]->d0);
        lep_z0.push_back(configMgr->obj->muons[i]->z0);
        lep_d0Err.push_back(configMgr->obj->muons[i]->d0Err);
        lep_z0Err.push_back(configMgr->obj->muons[i]->z0Err);
        lep_pTErr.push_back(configMgr->obj->muons[i]->pTErr);
        lep_ptcone20.push_back(configMgr->obj->muons[i]->ptcone20);
        lep_ptcone30.push_back(configMgr->obj->muons[i]->ptcone30);
        lep_ptcone40.push_back(configMgr->obj->muons[i]->ptcone40);
        lep_topoetcone20.push_back(configMgr->obj->muons[i]->topoetcone20);
        lep_topoetcone30.push_back(configMgr->obj->muons[i]->topoetcone30);
        lep_topoetcone40.push_back(configMgr->obj->muons[i]->topoetcone40);
        lep_ptvarcone20.push_back(configMgr->obj->muons[i]->ptvarcone20);
        lep_ptvarcone30.push_back(configMgr->obj->muons[i]->ptvarcone30);
        lep_ptvarcone40.push_back(configMgr->obj->muons[i]->ptvarcone40);
        lep_truthAuthor.push_back(configMgr->obj->muons[i]->author);
        lep_truthType.push_back(configMgr->obj->muons[i]->type);
        lep_truthOrigin.push_back(configMgr->obj->muons[i]->origin);
    }

    configMgr->treeMaker->setVecIntVariable("lepIso_lep_q",lep_q);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_pt",lep_pt);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_eta",lep_eta);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_phi",lep_phi);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_m",lep_m);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_d0",lep_d0);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_z0",lep_z0);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_d0Err",lep_d0Err);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_z0Err",lep_z0Err);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_pTErr",lep_pTErr);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_ptcone20",lep_ptcone20);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_ptcone30",lep_ptcone30);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_ptcone40",lep_ptcone40);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_topoetcone20",lep_topoetcone20);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_topoetcone30",lep_topoetcone30);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_topoetcone40",lep_topoetcone40);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_ptvarcone20",lep_ptvarcone20);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_ptvarcone30",lep_ptvarcone30);
    configMgr->treeMaker->setVecFloatVariable("lepIso_lep_ptvarcone40",lep_ptvarcone40);
    configMgr->treeMaker->setVecIntVariable("lepIso_lep_truthAuthor",lep_truthAuthor);
    configMgr->treeMaker->setVecIntVariable("lepIso_lep_truthType",lep_truthType);
    configMgr->treeMaker->setVecIntVariable("lepIso_lep_truthOrigin",lep_truthOrigin);

    // Fill track variables
    std::vector<int> track_q;
    std::vector<float> track_pt;
    std::vector<float> track_eta;
    std::vector<float> track_phi;
    std::vector<float> track_m;
    std::vector<float> track_fitQuality;
    std::vector<float> track_d0;
    std::vector<float> track_z0;
    std::vector<float> track_d0Err;
    std::vector<float> track_z0Err;
    std::vector<int> track_nIBLHits;
    std::vector<int> track_nPixHits;
    std::vector<int> track_nPixHoles;
    std::vector<int> track_nPixOutliers;
    std::vector<int> track_nSCTHits;
    std::vector<int> track_nTRTHits;

    for (unsigned i = 0; i < configMgr->obj->tracks.size(); i++) {
        track_q.push_back(configMgr->obj->tracks[i]->q);
        track_pt.push_back(configMgr->obj->tracks[i]->Pt());
        track_eta.push_back(configMgr->obj->tracks[i]->Eta());
        track_phi.push_back(configMgr->obj->tracks[i]->Phi());
        track_m.push_back(configMgr->obj->tracks[i]->M());
        track_fitQuality.push_back(configMgr->obj->tracks[i]->fitQuality);
        track_d0.push_back(configMgr->obj->tracks[i]->d0);
        track_z0.push_back(configMgr->obj->tracks[i]->z0);
        track_d0Err.push_back(configMgr->obj->tracks[i]->d0Err);
        track_z0Err.push_back(configMgr->obj->tracks[i]->z0Err);
        track_nIBLHits.push_back(configMgr->obj->tracks[i]->nIBLHits);
        track_nPixHits.push_back(configMgr->obj->tracks[i]->nPixHits);
        track_nPixHoles.push_back(configMgr->obj->tracks[i]->nPixHoles);
        track_nPixOutliers.push_back(configMgr->obj->tracks[i]->nPixOutliers);
        track_nSCTHits.push_back(configMgr->obj->tracks[i]->nSCTHits);
        track_nTRTHits.push_back(configMgr->obj->tracks[i]->nTRTHits);
    }

    configMgr->treeMaker->setVecIntVariable("lepIso_track_q",track_q);
    configMgr->treeMaker->setVecFloatVariable("lepIso_track_pt",track_pt);
    configMgr->treeMaker->setVecFloatVariable("lepIso_track_eta",track_eta);
    configMgr->treeMaker->setVecFloatVariable("lepIso_track_phi",track_phi);
    configMgr->treeMaker->setVecFloatVariable("lepIso_track_m",track_m);
    configMgr->treeMaker->setVecFloatVariable("lepIso_track_fitQuality",track_fitQuality);
    configMgr->treeMaker->setVecFloatVariable("lepIso_track_d0",track_d0);
    configMgr->treeMaker->setVecFloatVariable("lepIso_track_z0",track_z0);
    configMgr->treeMaker->setVecFloatVariable("lepIso_track_d0Err",track_d0Err);
    configMgr->treeMaker->setVecFloatVariable("lepIso_track_z0Err",track_z0Err);
    configMgr->treeMaker->setVecIntVariable("lepIso_track_nIBLHits",track_nIBLHits);
    configMgr->treeMaker->setVecIntVariable("lepIso_track_nPixHits",track_nPixHits);
    configMgr->treeMaker->setVecIntVariable("lepIso_track_nPixHoles",track_nPixHoles);
    configMgr->treeMaker->setVecIntVariable("lepIso_track_nPixOutliers",track_nPixOutliers);
    configMgr->treeMaker->setVecIntVariable("lepIso_track_nSCTHits",track_nSCTHits);
    configMgr->treeMaker->setVecIntVariable("lepIso_track_nTRTHits",track_nTRTHits);

    // Fill the output tree
    configMgr->treeMaker->Fill(configMgr->getSysState(),"tree");

    return true;

}
// ------------------------------------------------------------------------------------------ //
bool LeptonTrackSelector::passCuts(ConfigMgr*& configMgr)
{

    /*
       This method is used to apply any cuts you wish before writing
       the output trees
       */

    double weight = configMgr->objectTools->getWeight(configMgr->obj);

    // Fill cutflow histograms
    configMgr->cutflow->bookCut("cutFlow","allEvents",weight );

    // Apply all recommended event cleaning cuts
    if( !configMgr->obj->passEventCleaning( configMgr->cutflow, "cutFlow", weight ) ) return false;

    return true;

}
// ------------------------------------------------------------------------------------------ //
void LeptonTrackSelector::finalize(ConfigMgr*& configMgr)
{

    /*
       This method is called at the very end of the job. Can be used to merge cutflow histograms 
       for example. See CutFlowTool::mergeCutFlows(...)
       */

}
// ------------------------------------------------------------------------------------------ //
