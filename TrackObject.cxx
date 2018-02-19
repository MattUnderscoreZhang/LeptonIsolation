#include "SUSYTools/SUSYObjDef_xAOD.h"
#include "SusySkimMaker/ConstAccessors.h"
#include "SusySkimMaker/StatusCodeCheck.h"
#include "SusySkimMaker/CentralDB.h"
#include "SusySkimMaker/CentralDBFields.h"
#include "SusySkimMaker/Timer.h"
#include "SusySkimMaker/MsgLog.h"
#include "SusySkimMaker/TreeMaker.h"

//
#include <sstream>
#include <iostream>

// xAOD/RootCore
#include "SusySkimMaker/TrackObject.h"
#include "xAODTruth/xAODTruthHelpers.h"
#include "xAODTruth/TruthParticle.h"
#include "xAODTruth/TruthParticleContainer.h"
#include "xAODTracking/TrackParticlexAODHelpers.h"
#include "ElectronPhotonSelectorTools/ElectronSelectorHelpers.h"

// ROOT
#include "TMath.h"

TrackObject::TrackObject()
{
  m_trackVectorMap.clear();
}
// ------------------------------------------------------------------------- //
StatusCode TrackObject::init(TreeMaker*& treeMaker,xAOD::TEvent* event)
{

  const char* APP_NAME = "TrackObject";

  for( auto& sysName : treeMaker->getSysVector() ){
    TrackVector* tlv  = new TrackVector();
    // Save tracklet vector into map
    m_trackVectorMap.insert( std::pair<TString,TrackVector*>(sysName,tlv) );
    // Get tree created by createTrees
    TTree* sysTree = treeMaker->getTree("skim",sysName);
    // Don't write it out
    if(sysTree==NULL) continue;
    else{
      MsgLog::INFO("TrackObject::init", "Adding a branch tracklets to skims: %s", sysTree->GetName() );
      std::map<TString,TrackVector*>::iterator trackItr = m_trackVectorMap.find(sysName);
      sysTree->Branch("tracks",&trackItr->second);
    }      
  }

  //
  CHECK( init_tools() );

  // Set global TEvent object
  m_event = event;
  
  if( !m_event ){
    MsgLog::ERROR("TrackObject::init","TEvent pointer is not valid!");
    return StatusCode::FAILURE;
  }

  // Return gracefully
  return StatusCode::SUCCESS;

}
// ------------------------------------------------------------------------- //
StatusCode TrackObject::init_tools()
{

  // 
  CentralDB::retrieve(CentralDBFields::UNITS,m_convertFromMeV);

  return StatusCode::SUCCESS;
 
}
// -------------------------------------------------------------------------- //
StatusCode TrackObject::fillTrackContainer(std::string sys_name)
{

  //const char* APP_NAME = "TrackObject";

  std::map<TString,TrackVector*>::iterator it = m_trackVectorMap.find(sys_name);

  if( it==m_trackVectorMap.end() ){
    MsgLog::WARNING("TrackObject::fillTrackContainer","Request to get track for unknown systematic %s ", sys_name.c_str() );
    return StatusCode::FAILURE;
  }

  const xAOD::TrackParticleContainer* pixelTracklets = 0;
  //m_event->retrieve( pixelTracklets, "InDetPixelPrdAssociationTrackParticles");
  m_event->retrieve( pixelTracklets, "InDetTrackParticles");


  for( const auto& track : *pixelTracklets ){

    //
    TrackVariable* tracklet = new TrackVariable();

    // TLV
    tracklet->SetPtEtaPhiM( track->pt() * m_convertFromMeV, 
                            track->eta(),
                            track->phi(),
                            track->m() * m_convertFromMeV );

    // Charge
    tracklet->q = track->charge();
    
    // Quality cuts
    tracklet->fitQuality = (track->chiSquared() / track->numberDoF() );
    
    // Impact parameters
    tracklet->d0     = TrackObject::getD0(track);
    // TODO: pass PV and event info
    //tracklet->z0     = TrackObject::getZ0(track,primVertex);
    //tracklet->d0Err  = TrackObject::getD0Err(track,eventInfo);
    tracklet->z0Err  = TrackObject::getZ0Err(track);
    
    // Fill hits on the tracklet
    track->summaryValue(tracklet->nIBLHits, xAOD::numberOfInnermostPixelLayerHits);
    track->summaryValue(tracklet->nPixHits, xAOD::numberOfPixelHits);
    track->summaryValue(tracklet->nPixHoles, xAOD::numberOfPixelHoles);
    track->summaryValue(tracklet->nPixOutliers, xAOD::numberOfPixelOutliers);
    track->summaryValue(tracklet->nSCTHits, xAOD::numberOfSCTHits);
    track->summaryValue(tracklet->nTRTHits, xAOD::numberOfTRTHits);
    
    // Truth classification
    tracklet->type   = xAOD::TruthHelpers::getParticleTruthType(*track);
    tracklet->origin = xAOD::TruthHelpers::getParticleTruthOrigin(*track);
    //std::cout << xAOD::TruthHelpers::getParticleTruthType(*track) << std::endl;
    //std::cout << xAOD::TruthHelpers::getParticleTruthOrigin(*track) << std::endl;
    
    // Truth
    const xAOD::TruthParticle* truthTrack = xAOD::TruthHelpers::getTruthParticle( *track );
    if( truthTrack ){
      tracklet->truthTLV.SetPtEtaPhiM( truthTrack->pt() * m_convertFromMeV,
				       truthTrack->eta(),
				       truthTrack->phi(),
				       truthTrack->m() * m_convertFromMeV );
    }
    
    
    // TODO: Get information associated to caloirmeter energy deposits
    //       Perform the matching here
    //       For now, use quantities in the derivations
    
    it->second->push_back(tracklet);
    
  }
  

/*

  const xAOD::TruthParticleContainer* truthParticles = 0;
  m_event->retrieve( truthParticles, "TruthParticles");

 std::multimap<const xAOD::TruthParticle*,const xAOD::TruthParticle*> list;

  for( const auto& truthParticle : *truthParticles ){
    if( truthParticle->status() != 1 ) continue;
 
    // Search for charginos
    if( TMath::Abs(truthParticle->pdgId())==1000024 ){

      const xAOD::TruthParticle* truth_pion = 0;
      for( unsigned int c=0; c<truthParticle->nChildren(); c++ ){
        if( TMath::Abs(truthParticle->child(c)->pdgId())==211 ) truth_pion = truthParticle->child(c);
      } 

      if( truth_pion ){
        list.insert( std::make_pair(truthParticle,truth_pion) );
        continue;
      }
      else{
        std::cout << "No pion found. Listing other children of the chargino...." << std::endl;
        for( unsigned int c=0; c<truthParticle->nChildren(); c++ ){
          std::cout << "  >>>> PDG ID : " << truthParticle->child(c)->pdgId() << "   with pT : " << truthParticle->child(c)->pt() << std::endl;
        }
      }
    }
  }

  for( auto& cp : list ){

    TrackVariable* trk = new TrackVariable();

    trk->truthTLV.SetPtEtaPhiM( cp.first->pt() * m_convertFromMeV, 
                                cp.first->eta(), 
                                cp.first->phi(), 
                                cp.first->m() * m_convertFromMeV );

    //
    if( cp.first->hasDecayVtx() ){
      trk->truthProVtx = cp.first->decayVtx()->perp();
    }

    trk->getAssociatedTrack()->truthTLV.SetPtEtaPhiM( cp.second->pt() * m_convertFromMeV, 
                                                      cp.second->eta(), 
                                                      cp.second->phi(), 
                                                      cp.second->m() * m_convertFromMeV );


    if( cp.second->hasProdVtx() ){
      trk->getAssociatedTrack()->truthProVtx = cp.second->prodVtx()->perp();
    }

    it->second->push_back(trk);

  }

  //
  list.clear();
*/

  // Return gracefully 
  return StatusCode::SUCCESS;

}
// -------------------------------------------------------------------------- //
float TrackObject::getPtErr(const xAOD::TrackParticle* track)
{

  //
  if( !track ) return 0.0;

  return TMath::Sqrt( xAOD::TrackingHelpers::pTErr2( track ) );

}
// -------------------------------------------------------------------------- //
float TrackObject::getD0(const xAOD::TrackParticle* track)
{

  return track->d0();

}
// -------------------------------------------------------------------------- //
float TrackObject::getD0Err(const xAOD::TrackParticle* track,const xAOD::EventInfo* eventInfo)
{

  /*
    Calculate the d0 uncertainty, taking into account the uncertainty on the beam spot
  */

  double sigma2_d0 = track->definingParametersCovMatrixVec().at(0);
  
  float sigma2_beamSpotd0 = xAOD::TrackingHelpers::d0UncertaintyBeamSpot2(track->phi(),
									  eventInfo->beamPosSigmaX(), 
									  eventInfo->beamPosSigmaY(), 
									  eventInfo->beamPosSigmaXY() );

  float d0Err = TMath::Sqrt(sigma2_d0 + sigma2_beamSpotd0);
  return d0Err;

}
// -------------------------------------------------------------------------- //
float TrackObject::getZ0(const xAOD::TrackParticle* track,const xAOD::VertexContainer* vertex)
{

  // Get primary vertex
  const xAOD::Vertex* pV = 0;

  for( const auto& vx : *vertex ) {
    if(vx->vertexType() == xAOD::VxType::PriVtx){
      pV = vx;
      break;
    }
  }

  if( !pV ){
    // Return z0 found in xAODs
    std::cout << "<TrackObject::getZ0> WARNING Cannot get primary vertex (PV), z0 will not be extrapolated to the PV" << std::endl;
    return track->z0();
  }
  else{
    // Extrapolate back to PV
    double z0_exPV = track->z0() + track->vz() - pV->z();
    return z0_exPV;
  }
  
}
// -------------------------------------------------------------------------- //
float TrackObject::getZ0Err(const xAOD::TrackParticle* track)
{

  /*
    Index
      => d0 uncertainty 0
      => z0 uncertainty 2
      => qOverP uncertainty 14
  */

  std::vector<float> CovMatrixVec = track->definingParametersCovMatrixVec();

  const int idx = 2;

  if(CovMatrixVec.size()==15) return TMath::Sqrt(CovMatrixVec[idx]);
  else                        return (-100.0);

}
// -------------------------------------------------------------------------- //
uint8_t TrackObject::getNPixHitsPlusDeadSensors(const xAOD::TrackParticle* track)
{
  // Note that this works just as well for muons, despite being 
  // provided in the ElectronPhotonSelectorTools package
  return ElectronSelectorHelpers::numberOfPixelHitsAndDeadSensors(track);
}
// -------------------------------------------------------------------------- //
bool TrackObject::getPassBL(const xAOD::TrackParticle* track)
{
  // Note that this works just as well for muons, despite being 
  // provided in the ElectronPhotonSelectorTools package
  return ElectronSelectorHelpers::passBLayerRequirement(track);
}
// ------------------------------------------------------------------------- //
const TrackVector* TrackObject::getObj(TString sysName) 
{

  std::map<TString,TrackVector*>::iterator it = m_trackVectorMap.find(sysName);

  if(it==m_trackVectorMap.end()){
    MsgLog::WARNING("TrackObject::getObj","WARNING Cannot get track vector for systematic %s",sysName.Data() );
    return NULL;
  }

  return it->second;
  
}
// ------------------------------------------------------------------------- //
void TrackObject::Reset()
{

  std::map<TString,TrackVector*>::iterator it;
  for(it = m_trackVectorMap.begin(); it != m_trackVectorMap.end(); it++){

    // Free up memory
    for (TrackVector::iterator muItr = it->second->begin(); muItr != it->second->end(); muItr++) {
      delete *muItr;
    }

    it->second->clear();
  }

}
