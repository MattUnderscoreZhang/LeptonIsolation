#include "../headers/MuonFilter.h"

MuonFilter::MuonFilter() {
    m_muonSelectionTool = new CP::MuonSelectionTool("MuonObject_MuonSelectionTool");
    m_muonSelectionTool->initialize();
}

MuonFilter::~MuonFilter() {
    delete m_muonSelectionTool;
}

std::vector<const xAOD::Muon*> MuonFilter::filter_muons(const xAOD::MuonContainer& muons) {

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
