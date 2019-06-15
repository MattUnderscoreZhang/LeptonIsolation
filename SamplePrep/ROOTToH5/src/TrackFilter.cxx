#include "../headers/TrackFilter.h"

TrackFilter::TrackFilter() {

    m_trkseltool = new InDet::InDetTrackSelectionTool("trackSel");
    m_trkseltool->setProperty("CutLevel", "Loose");
    //m_trkseltool->setProperty("minPt", 500.);
    m_trkseltool->setProperty("minPt", 1000.);
    m_trkseltool->setProperty("maxZ0SinTheta", 3.);
    m_trkseltool->initialize();
}

TrackFilter::~TrackFilter() {
    delete m_trkseltool;
}

std::vector<const xAOD::TrackParticle*>* TrackFilter::filter_tracks(const xAOD::TrackParticleContainer& tracks) {
    // using https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/IsolationManualCalculation
    // https://twiki.cern.ch/twiki/bin/view/AtlasProtected/Run2IsolationHarmonisation
    // and https://twiki.cern.ch/twiki/bin/view/AtlasProtected/TrackingCPRecsEarly2018

    std::vector<const xAOD::TrackParticle*> m_current_tracks;

    for (const xAOD::TrackParticle *track : tracks) {
        if (!m_trkseltool->accept(*track)) continue;
        m_current_tracks.push_back(track);
    }

    return &m_current_tracks;
}
