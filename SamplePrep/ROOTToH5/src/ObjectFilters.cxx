#include "../headers/ObjectFilters.h"

ObjectFilters::ObjectFilters() {

    m_muonSelectionTool = new CP::MuonSelectionTool("MuonObject_MuonSelectionTool");
    m_muonSelectionTool->initialize();

    m_trkseltool = new InDet::InDetTrackSelectionTool("trackSel");
    m_trkseltool->setProperty("CutLevel", "Loose");
    //m_trkseltool->setProperty("minPt", 500.);
    m_trkseltool->setProperty("minPt", 1000.);
    m_trkseltool->setProperty("maxZ0SinTheta", 3.);
    m_trkseltool->initialize();
}

ObjectFilters::~ObjectFilters() {
    delete m_muonSelectionTool;
    delete m_trkseltool;
}

std::vector<const xAOD::Electron*> ObjectFilters::filter_electrons(const xAOD::ElectronContainer& electrons) {

    std::vector<const xAOD::Electron*> m_current_electrons;

    for (const xAOD::Electron *electron : electrons) {
        if(cacc_lhmedium.isAvailable(*electron) ){
            if (!cacc_lhmedium(*electron)) continue;
            m_current_electrons.push_back(electron);
        }
    }

    return m_current_electrons;
}

std::vector<const xAOD::Muon*> ObjectFilters::filter_muons(const xAOD::MuonContainer& muons) {

    std::vector<const xAOD::Muon*> m_current_muons;

    for (const xAOD::Muon *muon : muons) {
        // check that muon won't segfault
        if (muon == NULL) continue;
        if (muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle) == NULL) continue;
        // check that muon passes selections
        xAOD::Muon::Quality muonQuality = m_muonSelectionTool->getQuality(*muon);
        if (muonQuality < xAOD::Muon::Medium) continue;
        // store muons
        m_current_muons.push_back(muon);
    }

    return m_current_muons;
}

std::vector<const xAOD::TrackParticle*> ObjectFilters::filter_tracks(const xAOD::TrackParticleContainer& tracks) {
    // using https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/IsolationManualCalculation
    // https://twiki.cern.ch/twiki/bin/view/AtlasProtected/Run2IsolationHarmonisation
    // and https://twiki.cern.ch/twiki/bin/view/AtlasProtected/TrackingCPRecsEarly2018

    std::vector<const xAOD::TrackParticle*> m_current_tracks;

    for (const xAOD::TrackParticle *track : tracks) {
        if (!m_trkseltool->accept(*track)) continue;
        m_current_tracks.push_back(track);
    }

    return m_current_tracks;
}
