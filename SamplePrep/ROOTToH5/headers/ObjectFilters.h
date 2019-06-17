#ifndef OBJECT_FILTERS_H
#define OBJECT_FILTERS_H

namespace xAOD {
    class Electron_v1;
    typedef Electron_v1 Electron;
    class Muon_v1;
    typedef Muon_v1 Muon;
    class TrackParticle_v1;
    typedef TrackParticle_v1 TrackParticle;
}

// ATLAS things
#include "xAODEgamma/ElectronContainer.h"
#include "xAODEgamma/Electron.h"
#include "xAODMuon/MuonContainer.h"
#include "xAODMuon/Muon.h"
#include "MuonSelectorTools/MuonSelectionTool.h"
#include "xAODTracking/TrackParticleContainer.h"
#include "xAODTracking/TrackParticle.h"
#include "xAODEgamma/EgammaxAODHelpers.h"
#include "InDetTrackSelectionTool/InDetTrackSelectionTool.h"

static SG::AuxElement::ConstAccessor<char> cacc_lhmedium("DFCommonElectronsLHMedium");

class ObjectFilters
{
    public:
        // constructor and destructor
        ObjectFilters();
        ~ObjectFilters();

        // electron selection
        std::vector<const xAOD::Electron*> filter_electrons(const xAOD::ElectronContainer* electrons);

        // muon selection
        std::vector<const xAOD::Muon*> filter_muons(const xAOD::MuonContainer* muons);

        // track selection
        std::vector<const xAOD::TrackParticle*> filter_tracks(const xAOD::TrackParticleContainer* tracks, const xAOD::Vertex* primary_vertex);

    private:
        // muon selector
        CP::MuonSelectionTool* m_muonSelectionTool;

        // track selector
        InDet::InDetTrackSelectionTool *m_trkseltool;
};

#endif
