// ATLAS things
#include "xAODEgamma/ElectronContainer.h"
#include "xAODEgamma/Electron.h"
#include "xAODMuon/MuonContainer.h"
#include "xAODMuon/Muon.h"
#include "ElectronPhotonSelectorTools/AsgElectronLikelihoodTool.h"
#include "MuonSelectorTools/MuonSelectionTool.h"
#include "xAODTracking/TrackParticleContainer.h"
#include "xAODTracking/TrackParticle.h"
#include "xAODEgamma/EgammaxAODHelpers.h"
#include "xAODTruth/xAODTruthHelpers.h"
#include "xAODTruth/TruthEvent.h"
#include "InDetTrackSelectionTool/InDetTrackSelectionTool.h"
#include "xAODJet/JetContainer.h"
#include "xAODJet/Jet.h"

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
static SG::AuxElement::ConstAccessor<char> cacc_lhloos("DFCommonElectronsLHLoose");
static SG::AuxElement::ConstAccessor<char> cacc_lhtight("DFCommonElectronLHTight");
class ObjectFilters {
    public:
        ObjectFilters() {
            //m_electronLHMediumSelectionTool = new AsgElectronLikelihoodTool("ElectronLHMediumSelectionTool");
            //m_electronLHMediumSelectionTool->setProperty("WorkingPoint", "MediumLHElectron");
            //m_electronLHMediumSelectionTool->initialize();

            //m_electronLHLooseSelectionTool = new AsgElectronLikelihoodTool("ElectronLHLooseSelectionTool");
            //m_electronLHLooseSelectionTool->setProperty("WorkingPoint", "LooseLHElectron");
            //m_electronLHLooseSelectionTool->initialize();

            m_muonSelectionTool = new CP::MuonSelectionTool("MuonObject_MuonSelectionTool");
            m_muonSelectionTool->initialize();

            m_trkseltool = new InDet::InDetTrackSelectionTool("trackSel");
            m_trkseltool->setProperty("CutLevel", "Loose");
            m_trkseltool->setProperty("minPt", 1000.);
            m_trkseltool->setProperty("maxZ0SinTheta", 3.);
            m_trkseltool->initialize();
        }

        ~ObjectFilters() {
            //delete m_electronLHMediumSelectionTool;
            //delete m_electronLHLooseSelectionTool;
            delete m_muonSelectionTool;
            delete m_trkseltool;
        }
 

        vector<pair<const xAOD::Electron*, int>> filter_electrons_probe(vector<pair<const xAOD::Electron*, int>> electrons) {
            // for use in tag and probe
            vector<pair<const xAOD::Electron*, int>> m_current_electrons;
            for (pair<const xAOD::Electron*, int> electron : electrons) {
                // check that electron passes selections
                //if (!m_electronLHLooseSelectionTool->accept((electron.first))) continue;
                if (!cacc_lhloos.isAvailable(*(electron.first))) continue;
                if (!cacc_lhloos(*(electron.first))) continue;
                if (electron.first->pt()< 5000) continue; //value in MeV
                m_current_electrons.push_back(electron);
            }
            return m_current_electrons;
        }

         vector<pair<const xAOD::Electron*, int>> filter_electrons_tag(vector<pair<const xAOD::Electron*, int>> electrons) {
            // for use in tag and probe
            vector<pair<const xAOD::Electron*, int>> m_current_electrons;
            for (pair<const xAOD::Electron*, int> electron : electrons) {
                // check that electron passes selections
                if (!cacc_lhtight.isAvailable(*(electron.first))) continue;
                if (!cacc_lhtight(*(electron.first))) continue;
                if ( electron.first->pt()< 10000) continue; //value in MeV
                m_current_electrons.push_back(electron);
            }
            return m_current_electrons;
        }


        vector<pair<const xAOD::Electron*, int>> filter_electrons_truth_type(const xAOD::ElectronContainer* electrons) {
            vector<pair<const xAOD::Electron*, int>> m_current_electrons;
            for (const xAOD::Electron *electron : *electrons) {
                // truth type: 2 = real prompt, 3 = HF
                int truth_type = xAOD::TruthHelpers::getParticleTruthType(*electron);
                if (truth_type != 2 && truth_type != 3) continue;
                // truth parent (filter out tau decays)
                const xAOD::TruthParticle* particle = xAOD::TruthHelpers::getTruthParticle(*electron);
                if (particle == 0) continue;
                if (particle->parent() == 0) continue;
                if (abs(particle->parent()->pdgId()) == 15 || abs(particle->parent()->pdgId()) == 17) continue;
                // store electron
                m_current_electrons.push_back(make_pair(electron, truth_type));
            }
            return m_current_electrons;
        }

        vector<pair<const xAOD::Electron*, int>> filter_electrons_medium(vector<pair<const xAOD::Electron*, int>> electrons) {
            vector<pair<const xAOD::Electron*, int>> m_current_electrons;
            for (pair<const xAOD::Electron*, int> electron : electrons) {
                // check that electron passes selections
                //if (!m_electronLHMediumSelectionTool->accept((electron.first))) continue;
                if (!cacc_lhmedium.isAvailable(*(electron.first))) continue;
                if (!cacc_lhmedium(*(electron.first))) continue;
                m_current_electrons.push_back(electron);
            }
            return m_current_electrons;
        }

        vector<pair<const xAOD::Electron*, int>> filter_electrons_low_pT(vector<pair<const xAOD::Electron*, int>> electrons) {
            vector<pair<const xAOD::Electron*, int>> m_current_electrons;
            for (pair<const xAOD::Electron*, int> electron : electrons) {
                // check that electron passes selections
                if (electron.first->pt()>10000) continue;
                m_current_electrons.push_back(electron);
            }
            return m_current_electrons;
        }

        vector<pair<const xAOD::Muon*, int>> filter_muons_probe(vector<pair<const xAOD::Muon*, int>> muons) {
            // for use in tag and probe
            vector<pair<const xAOD::Muon*, int>> m_current_muons;
            for (pair<const xAOD::Muon*, int> muon : muons) {
                // check that muon passes selections
                xAOD::Muon::Quality muonQuality = m_muonSelectionTool->getQuality(*(muon.first));
                if (muonQuality < xAOD::Muon::Loose) continue;
                if (muon.first->pt()<5000) continue; //value in MeV
                // store muon
                m_current_muons.push_back(muon);
            }
            return m_current_muons;
        }
        
        vector<pair<const xAOD::Muon*, int>> filter_muons_tag(vector<pair<const xAOD::Muon*, int>> muons) {
            // for use in tag and probe
            vector<pair<const xAOD::Muon*, int>> m_current_muons;
            for (pair<const xAOD::Muon*, int> muon : muons) {
                // check that muon passes selections
                xAOD::Muon::Quality muonQuality = m_muonSelectionTool->getQuality(*(muon.first));
                if (muonQuality < xAOD::Muon::Tight) continue;
                if (muon.first->pt()<10000) continue; //value in MeV
                // store muon
                m_current_muons.push_back(muon);
            }
            return m_current_muons;
        }

        vector<pair<const xAOD::Muon*, int>> filter_muons_truth_type(const xAOD::MuonContainer* muons) {
            vector<pair<const xAOD::Muon*, int>> m_current_muons;
            for (const xAOD::Muon *muon : *muons) {
                // check that muon won't segfault
                if (muon == NULL) continue;
                if (muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle) == NULL) continue;
                // truth type: 6 = real prompt, 7 = HF
                int truth_type = xAOD::TruthHelpers::getParticleTruthType(*(muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle)));
                if (truth_type != 6 && truth_type != 7) continue;
                // truth parent (filter out tau decays)
                const xAOD::TruthParticle* particle = xAOD::TruthHelpers::getTruthParticle(*(muon->trackParticle(xAOD::Muon::InnerDetectorTrackParticle)));
                if (particle == 0) continue;
                if (particle->parent() == 0) continue;
                if (abs(particle->parent()->pdgId()) == 15 || abs(particle->parent()->pdgId()) == 17) continue;
                // store muon
                m_current_muons.push_back(make_pair(muon, truth_type));
            }
            return m_current_muons;
        }

        vector<pair<const xAOD::Muon*, int>> filter_muons_tight(vector<pair<const xAOD::Muon*, int>> muons) {
            vector<pair<const xAOD::Muon*, int>> m_current_muons;
            for (pair<const xAOD::Muon*, int> muon : muons) {
                // check that muon passes selections
                xAOD::Muon::Quality muonQuality = m_muonSelectionTool->getQuality(*(muon.first));
                if (muonQuality > xAOD::Muon::Tight) continue;
                // store muon
                m_current_muons.push_back(muon);
            }
            return m_current_muons;
        }

        vector<pair<const xAOD::Muon*, int>> filter_muons_low_pT(vector<pair<const xAOD::Muon*, int>> muons) {
            vector<pair<const xAOD::Muon*, int>> m_current_muons;
            for (pair<const xAOD::Muon*, int> muon : muons) {
                if (muon.first->pt()>10000) continue;
                m_current_muons.push_back(muon);
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

        vector<const xAOD::CaloCluster*> filter_calo_clusters(const xAOD::CaloClusterContainer* calo_clusters) {
            vector<const xAOD::CaloCluster*> m_current_calo_clusters;
            for (const xAOD::CaloCluster *calo_cluster : *calo_clusters) {
                m_current_calo_clusters.push_back(calo_cluster);
            }
            return m_current_calo_clusters;
        }

        vector<const xAOD::Jet*> filter_jets(const xAOD::JetContainer* jets) {
            vector<const xAOD::Jet*> m_current_jets;
            for (const xAOD::Jet *jet : *jets) {
                m_current_jets.push_back(jet);
            }
            return m_current_jets;
        }

    private:
        CP::MuonSelectionTool* m_muonSelectionTool;
        //AsgElectronLikelihoodTool* m_electronLHMediumSelectionTool ;
        //AsgElectronLikelihoodTool* m_electronLHLooseSelectionTool ;
        InDet::InDetTrackSelectionTool *m_trkseltool;
};
