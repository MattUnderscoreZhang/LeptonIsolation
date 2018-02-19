#include "SusySkimHiggsino/LeptonTrackSelector.h"
#include <iostream>


// ------------------------------------------------------------------------------------------ //
LeptonTrackSelector::LeptonTrackSelector() : BaseUser("SusySkimHiggsino","LeptonTrackSelector")
{

}
// ------------------------------------------------------------------------------------------ //
void LeptonTrackSelector::setup(ConfigMgr*& configMgr)
{

    // Define any variables you want to write out here. An example is given below
    configMgr->treeMaker->addIntVariable("q",0);
    configMgr->treeMaker->addFloatVariable("pt",0.0);
    configMgr->treeMaker->addFloatVariable("eta",0.0);
    configMgr->treeMaker->addFloatVariable("phi",0.0);
    configMgr->treeMaker->addFloatVariable("m",0.0);
    configMgr->treeMaker->addFloatVariable("fitQuality",0.0);
    configMgr->treeMaker->addFloatVariable("d0",0.0);
    //configMgr->treeMaker->addFloatVariable("z0",0.0);
    //configMgr->treeMaker->addFloatVariable("d0Err",0.0);
    configMgr->treeMaker->addFloatVariable("z0Err",0.0);
    configMgr->treeMaker->addIntVariable("nIBLHits",0);
    configMgr->treeMaker->addIntVariable("nPixHits",0);
    configMgr->treeMaker->addIntVariable("nPixHoles",0);
    configMgr->treeMaker->addIntVariable("nPixOutliers",0);
    configMgr->treeMaker->addIntVariable("nSCTHits",0);
    configMgr->treeMaker->addIntVariable("nTRTHits",0);
    configMgr->treeMaker->addIntVariable("truthType",0);
    configMgr->treeMaker->addIntVariable("truthOrigin",0);

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

    // Fill output trees, build observables, what ever you like here.
    // You need to define the variable above in setup(...), before filling here
    for (unsigned i = 0; i < configMgr->obj->tracks.size(); i++) {
        configMgr->treeMaker->setIntVariable("q",configMgr->obj->tracks[i]->q);
        configMgr->treeMaker->setFloatVariable("pt",configMgr->obj->tracks[i]->Pt());
        configMgr->treeMaker->setFloatVariable("eta",configMgr->obj->tracks[i]->Eta());
        configMgr->treeMaker->setFloatVariable("phi",configMgr->obj->tracks[i]->Phi());
        configMgr->treeMaker->setFloatVariable("m",configMgr->obj->tracks[i]->M());
        configMgr->treeMaker->setFloatVariable("fitQuality",configMgr->obj->tracks[i]->fitQuality);
        configMgr->treeMaker->setFloatVariable("d0",configMgr->obj->tracks[i]->d0);
        //configMgr->treeMaker->setFloatVariable("z0",configMgr->obj->tracks[i]->z0);
        //configMgr->treeMaker->setFloatVariable("d0Err",configMgr->obj->tracks[i]->d0Err);
        configMgr->treeMaker->setFloatVariable("z0Err",configMgr->obj->tracks[i]->z0Err);
        configMgr->treeMaker->setIntVariable("nIBLHits",configMgr->obj->tracks[i]->nIBLHits);
        configMgr->treeMaker->setIntVariable("nPixHits",configMgr->obj->tracks[i]->nPixHits);
        configMgr->treeMaker->setIntVariable("nPixHoles",configMgr->obj->tracks[i]->nPixHoles);
        configMgr->treeMaker->setIntVariable("nPixOutliers",configMgr->obj->tracks[i]->nPixOutliers);
        configMgr->treeMaker->setIntVariable("nSCTHits",configMgr->obj->tracks[i]->nSCTHits);
        configMgr->treeMaker->setIntVariable("nTRTHits",configMgr->obj->tracks[i]->nTRTHits);
        //configMgr->treeMaker->setFloatVariable("truthType",configMgr->obj->tracks[i]->type);
        //configMgr->treeMaker->setFloatVariable("truthOrigin",configMgr->obj->tracks[i]->origin);
    }

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
