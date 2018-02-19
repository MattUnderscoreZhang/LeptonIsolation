#include "SusySkimMaker/TrackVariable.h"

// -------------------------------------------------------------------------------- //
TrackVariable::TrackVariable() : m_associatedTrack(0)
{

  setDefault(q,-0);
  setDefault(d0,-1.0);
  setDefault(z0,-1.0);
  setDefault(d0Err,-1.0);
  setDefault(z0Err,-1.0);
  setDefault(nIBLHits,0);
  setDefault(nPixHits,0);
  setDefault(nIBLHits,0);
  setDefault(nPixHoles,0);
  setDefault(nPixOutliers,0);
  setDefault(nSCTHits,0);
  setDefault(nTRTHits,0);
  setDefault(type,-1);
  setDefault(origin,-1);
  setDefault(fitQuality,-1);
  setDefault(truthProVtx,-1);

}
// -------------------------------------------------------------------------------- //
TrackVariable::TrackVariable(const TrackVariable &rhs):
  ObjectVariable(rhs),
  q(rhs.q),
  d0(rhs.d0),
  z0(rhs.z0),
  d0Err(rhs.d0Err),
  z0Err(rhs.z0Err),
  nIBLHits(rhs.nIBLHits),
  nPixHits(rhs.nPixHits),
  nPixHoles(rhs.nPixHoles),
  nPixOutliers(rhs.nPixOutliers),
  nSCTHits(rhs.nSCTHits),
  nTRTHits(rhs.nTRTHits),
  type(rhs.type),
  origin(rhs.origin),
  fitQuality(rhs.fitQuality),
  truthProVtx(rhs.truthProVtx),
  caloCluster(rhs.caloCluster),
  m_associatedTrack(rhs.m_associatedTrack)

{

}
// -------------------------------------------------------------------------------- //
TrackVariable& TrackVariable::operator=(const TrackVariable &rhs)
{

  if (this != &rhs) {
    ObjectVariable::operator=(rhs);
    q             = rhs.q;
    d0            = rhs.d0;
    z0            = rhs.z0;
    d0Err         = rhs.d0Err;
    z0Err         = rhs.z0Err;
    nIBLHits      = rhs.nIBLHits;
    nPixHits      = rhs.nPixHits;
    nPixHoles     = rhs.nPixHoles;
    nPixOutliers  = rhs.nPixOutliers;
    nSCTHits      = rhs.nSCTHits;
    nTRTHits      = rhs.nTRTHits;
    type          = rhs.type;
    origin        = rhs.origin;
    fitQuality    = rhs.fitQuality;
    truthProVtx   = rhs.truthProVtx;
    caloCluster   = rhs.caloCluster;
    m_associatedTrack = rhs.m_associatedTrack;
  }

  return *this;

}
// -------------------------------------------------------------------------------- //
void TrackVariable::makeAssociatedTrack()
{
  m_associatedTrack = new TrackVariable();
}
// -------------------------------------------------------------------------------- //
TrackVariable* TrackVariable::getAssociatedTrack()
{

  if( !m_associatedTrack ){
    makeAssociatedTrack();
  }

  // Return
  return m_associatedTrack;

}




