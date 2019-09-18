// ATLAS things
#include "xAODEgamma/ElectronContainer.h"
#include "xAODEgamma/Electron.h"
#include "xAODMuon/MuonContainer.h"
#include "xAODMuon/Muon.h"
#include "MuonSelectorTools/MuonSelectionTool.h"
#include "xAODTracking/TrackParticleContainer.h"
#include "xAODTracking/TrackParticle.h"
#include "xAODEgamma/EgammaxAODHelpers.h"
#include "xAODTruth/xAODTruthHelpers.h"
#include "InDetTrackSelectionTool/InDetTrackSelectionTool.h"

namespace xAOD {
    class Electron_v1;
    typedef Electron_v1 Electron;
    class Muon_v1;
    typedef Muon_v1 Muon;
    class TrackParticle_v1;
    typedef TrackParticle_v1 TrackParticle;
}

using namespace std;

static SG::AuxElement::ConstAccessor<char> cacc_lhmedium("DFCommonElectronsLHMedium");

class ObjectFilters {

    public:

        ObjectFilters() {

            m_muonSelectionTool = new CP::MuonSelectionTool("MuonObject_MuonSelectionTool");
            m_muonSelectionTool->initialize();

            m_trkseltool = new InDet::InDetTrackSelectionTool("trackSel");
            m_trkseltool->setProperty("CutLevel", "Loose");
            m_trkseltool->setProperty("minPt", 1000.);
            m_trkseltool->setProperty("maxZ0SinTheta", 3.);
            m_trkseltool->initialize();
        }

        ~ObjectFilters() {
            delete m_muonSelectionTool;
            delete m_trkseltool;
        }

        vector<pair<const xAOD::Electron*, int>> filter_electrons(const xAOD::ElectronContainer* electrons) {

            vector<pair<const xAOD::Electron*, int>> m_current_electrons;

            for (const xAOD::Electron *electron : *electrons) {
                // check that electron passes selections
                if (!cacc_lhmedium.isAvailable(*electron)) continue;
                if (!cacc_lhmedium(*electron)) continue;
                // truth type: 2 = real prompt, 3 = HF
                int truth_type = xAOD::TruthHelpers::getParticleTruthType(*electron);
                if (truth_type != 2 && truth_type != 3) continue;
                // store electron
                m_current_electrons.push_back(make_pair(electron, truth_type));
            }

            return m_current_electrons;
        }

        vector<pair<const xAOD::Muon*, int>> filter_muons(const xAOD::MuonContainer* muons) {

            vector<pair<const xAOD::Muon*, int>> m_current_muons;

            for (const xAOD::Muon *muon : *muons) {
                // check that muon won't segfault
                if (muon == NULL) continue;
                if (muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle) == NULL) continue;
                // check that muon passes selections
                xAOD::Muon::Quality muonQuality = m_muonSelectionTool->getQuality(*muon);
                if (muonQuality < xAOD::Muon::Medium) continue;
                // truth type: 6 = real prompt, 7 = HF
                int truth_type = xAOD::TruthHelpers::getParticleTruthType(*(muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle)));
                if (truth_type != 6 && truth_type != 7) continue;
                // store muon
                m_current_muons.push_back(make_pair(muon, truth_type));
            }

            return m_current_muons;
        }

        vector<const xAOD::TrackParticle*> filter_tracks(const xAOD::TrackParticleContainer* tracks, const xAOD::Vertex* primary_vertex) {
            // using https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/IsolationManualCalculation
            // https://twiki.cern.ch/twiki/bin/view/AtlasProtected/Run2IsolationHarmonisation
            // and https://twiki.cern.ch/twiki/bin/view/AtlasProtected/TrackingCPRecsEarly2018

            vector<const xAOD::TrackParticle*> m_current_tracks;

            for (const xAOD::TrackParticle *track : *tracks) {
                if (!m_trkseltool->accept(*track, primary_vertex)) continue;
                m_current_tracks.push_back(track);
            }

            return m_current_tracks;
        }

    private:

        CP::MuonSelectionTool* m_muonSelectionTool;
        InDet::InDetTrackSelectionTool *m_trkseltool;

};
