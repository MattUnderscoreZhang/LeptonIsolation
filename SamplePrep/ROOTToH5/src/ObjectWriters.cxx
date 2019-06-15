#include "../headers/ObjectWriters.h"

// EDM things
#include "xAODMuon/MuonContainer.h"
#include "xAODMuon/Muon.h"
#include "xAODEgamma/ElectronContainer.h"
#include "xAODEgamma/Electron.h"
#include "xAODTruth/xAODTruthHelpers.h"
#include "xAODTracking/TrackParticlexAODHelpers.h"

// HDF5 things
#include "HDF5Utils/HdfTuple.h"
#include "H5Cpp.h"

ObjectWriters::ObjectWriters(H5::Group& output_group):
    m_electron_idx(1),
    m_electron_writer(nullptr),
    m_muon_idx(1),
    m_muon_writer(nullptr),
    m_track_idx(1),
    m_track_writer(nullptr)
{

    //-----------//
    // ELECTRONS //
    //-----------//

    H5Utils::VariableFillers* fillers = new H5Utils::VariableFillers();

    fillers->add<int>("pdgID",
        [this]() {
            return 11;
        }
    );
    fillers->add<float>("pT",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (float)(this->m_current_electrons.at(idx)->pt());
        }
    );
    fillers->add<float>("eta",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (float)(this->m_current_electrons.at(idx)->eta());
        }
    );
    fillers->add<float>("phi",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (float)(this->m_current_electrons.at(idx)->phi());
        }
    );
    fillers->add<float>("d0",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (float)(this->m_current_electrons.at(idx)->trackParticle()->d0());
        }
    );
    fillers->add<float>("d0_over_sigd0",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (float)(xAOD::TrackingHelpers::d0significance(this->m_current_electrons.at(idx)->trackParticle()));
        }
    );
    fillers->add<float>("z0",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (float)(this->m_current_electrons.at(idx)->trackParticle()->z0());
        }
    );
    fillers->add<float>("dz0",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (float)(this->m_current_electrons.at(idx)->trackParticle()->z0() - this->m_primary_vertices_z0.at(idx));
        }
    );
    fillers->add<float>("ptcone20",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            float ptcone20 = 0.0;
            this->m_current_electrons.at(idx)->isolationValue(ptcone20,xAOD::Iso::ptcone20);
            return ptcone20;
        }
    );
    fillers->add<float>("ptcone30",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            float ptcone30 = 0.0;
            this->m_current_electrons.at(idx)->isolationValue(ptcone30,xAOD::Iso::ptcone30);
            return ptcone30;
        }
    );
    fillers->add<float>("ptcone40",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            float ptcone40 = 0.0;
            this->m_current_electrons.at(idx)->isolationValue(ptcone40,xAOD::Iso::ptcone40);
            return ptcone40;
        }
    );
    fillers->add<float>("ptvarcone20",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            float ptvarcone20 = 0.0;
            this->m_current_electrons.at(idx)->isolationValue(ptvarcone20,xAOD::Iso::ptvarcone20);
            return ptvarcone20;
        }
    );
    fillers->add<float>("ptvarcone30",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            float ptvarcone30 = 0.0;
            this->m_current_electrons.at(idx)->isolationValue(ptvarcone30,xAOD::Iso::ptvarcone30);
            return ptvarcone30;
        }
    );
    fillers->add<float>("ptvarcone40",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            float ptvarcone40 = 0.0;
            this->m_current_electrons.at(idx)->isolationValue(ptvarcone40,xAOD::Iso::ptvarcone40);
            return ptvarcone40;
        }
    );
    fillers->add<int>("truth_type",
        [this]() -> int {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (int)(xAOD::TruthHelpers::getParticleTruthType(*(this->m_current_electrons.at(idx))));
            // 2 = real prompt, 3 = HF
        }
    );
    fillers->add<float>("PLT",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            SG::AuxElement::ConstAccessor<float> accessPromptVar("PromptLeptonVeto");
            const xAOD::IParticle *particle_pointer = NULL;
            particle_pointer = this->m_current_electrons.at(idx);
            return accessPromptVar(*particle_pointer);
        }
    );

    // Save up to 20 electrons per event
    m_electron_writer = new H5Utils::WriterXd(output_group, "electrons", *fillers, {20});

    //-------//
    // MUONS //
    //-------//

    delete fillers;
    fillers = new H5Utils::VariableFillers();

    fillers->add<int>("pdgID",
        [this]() {
            return 13;
        }
    );
    fillers->add<float>("pT",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            return (float)(this->m_current_muons.at(idx)->pt());
        }
    );
    fillers->add<float>("eta",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            return (float)(this->m_current_muons.at(idx)->eta());
        }
    );
    fillers->add<float>("phi",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            return (float)(this->m_current_muons.at(idx)->phi());
        }
    );
    fillers->add<float>("d0",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            return (float)(this->m_current_muons.at(idx)->trackParticle(xAOD::Muon::InnerDetectorTrackParticle)->d0());
        }
    );
    fillers->add<float>("d0_over_sigd0",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            return (float)(xAOD::TrackingHelpers::d0significance(this->m_current_muons.at(idx)->trackParticle(xAOD::Muon::InnerDetectorTrackParticle)));
        }
    );
    fillers->add<float>("z0",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            return (float)(this->m_current_muons.at(idx)->trackParticle(xAOD::Muon::InnerDetectorTrackParticle)->z0());
        }
    );
    fillers->add<float>("dz0",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            return (float)(this->m_current_muons.at(idx)->trackParticle(xAOD::Muon::InnerDetectorTrackParticle)->z0() - this->m_primary_vertices_z0.at(idx));
        }
    );
    fillers->add<float>("ptcone20",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            float ptcone20 = 0.0;
            this->m_current_muons.at(idx)->isolation(ptcone20,xAOD::Iso::ptcone20);
            return ptcone20;
        }
    );
    fillers->add<float>("ptcone30",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            float ptcone30 = 0.0;
            this->m_current_muons.at(idx)->isolation(ptcone30,xAOD::Iso::ptcone30);
            return ptcone30;
        }
    );
    fillers->add<float>("ptcone40",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            float ptcone40 = 0.0;
            this->m_current_muons.at(idx)->isolation(ptcone40,xAOD::Iso::ptcone40);
            return ptcone40;
        }
    );
    fillers->add<float>("ptvarcone20",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            float ptvarcone20 = 0.0;
            this->m_current_muons.at(idx)->isolation(ptvarcone20,xAOD::Iso::ptvarcone20);
            return ptvarcone20;
        }
    );
    fillers->add<float>("ptvarcone30",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            float ptvarcone30 = 0.0;
            this->m_current_muons.at(idx)->isolation(ptvarcone30,xAOD::Iso::ptvarcone30);
            return ptvarcone30;
        }
    );
    fillers->add<float>("ptvarcone40",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            float ptvarcone40 = 0.0;
            this->m_current_muons.at(idx)->isolation(ptvarcone40,xAOD::Iso::ptvarcone40);
            return ptvarcone40;
        }
    );
    fillers->add<int>("truth_type",
        [this]() -> int {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            const xAOD::TrackParticle* track = this->m_current_muons.at(idx)->trackParticle(xAOD::Muon::InnerDetectorTrackParticle);
            return (int)(xAOD::TruthHelpers::getParticleTruthType(*track));
            // 2 = real prompt, 3 = HF
        }
    );
    fillers->add<float>("PLT",
        [this]() {
            size_t idx = this->m_muon_idx.at(0);
            if (this->m_current_muons.size() <= idx) return NAN;
            SG::AuxElement::ConstAccessor<float> accessPromptVar("PromptLeptonVeto");
            const xAOD::IParticle *particle_pointer = NULL;
            particle_pointer = this->m_current_muons.at(idx);
            return accessPromptVar(*particle_pointer);
        }
    );

    // Save up to 20 muons per event
    m_muon_writer = new H5Utils::WriterXd(output_group, "muons", *fillers, {20});

    //--------//
    // TRACKS //
    //--------//

    delete fillers;
    fillers = new H5Utils::VariableFillers();
    //CP::IRetrievePFOTool *m_pfotool;//!

    fillers->add<float>("pT",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->pt());
        }
    );
    fillers->add<float>("eta",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->eta());
        }
    );
    fillers->add<float>("phi",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->phi());
        }
    );
    fillers->add<float>("charge",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->charge());
        }
    );
    fillers->add<float>("d0",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->d0());
        }
    );
    fillers->add<float>("z0",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->z0());
        }
    );
    fillers->add<float>("theta",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->theta());
        }
    );
    fillers->add<float>("chiSquared",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->chiSquared());
        }
    );
    fillers->add<float>("nIBLHits",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            uint8_t nHits = 0;
            this->m_current_tracks.at(idx)->summaryValue(nHits, xAOD::numberOfInnermostPixelLayerHits);
            return (float)nHits;
        }
    );
    fillers->add<float>("nPixHits",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            uint8_t nHits = 0;
            this->m_current_tracks.at(idx)->summaryValue(nHits, xAOD::numberOfPixelHits);
            return (float)nHits;
        }
    );
    fillers->add<float>("nPixHoles",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            uint8_t nHits = 0;
            this->m_current_tracks.at(idx)->summaryValue(nHits, xAOD::numberOfPixelHoles);
            return (float)nHits;
        }
    );
    fillers->add<float>("nPixOutliers",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            uint8_t nHits = 0;
            this->m_current_tracks.at(idx)->summaryValue(nHits, xAOD::numberOfPixelOutliers);
            return (float)nHits;
        }
    );
    fillers->add<float>("nSCTHits",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            uint8_t nHits = 0;
            this->m_current_tracks.at(idx)->summaryValue(nHits, xAOD::numberOfSCTHits);
            return (float)nHits;
        }
    );
    fillers->add<float>("nSCTHoles",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            uint8_t nHits = 0;
            this->m_current_tracks.at(idx)->summaryValue(nHits, xAOD::numberOfSCTHoles);
            return (float)nHits;
        }
    );
    fillers->add<float>("nTRTHits",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            uint8_t nHits = 0;
            this->m_current_tracks.at(idx)->summaryValue(nHits, xAOD::numberOfTRTHits);
            return (float)nHits;
        }
    );

    //// topocluster stuff - using https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/IsolationManualCalculation
    //fillers->add<float>("nTRTHits",
        //[this]() {
            //size_t idx = this->m_track_idx.at(0);
            //if (this->m_current_tracks.size() <= idx) return NAN;
            //uint8_t nHits = 0;
            //this->m_current_tracks.at(idx)->summaryValue(nHits, xAOD::numberOfTRTHits);
            //return (float)nHits;
        //}
    //);

    // Save up to 3000 tracks per event
    m_track_writer = new H5Utils::WriterXd(output_group, "tracks", *fillers, {3000});
}

ObjectWriters::~ObjectWriters() {
    if (m_electron_writer) m_electron_writer->flush();
    if (m_muon_writer) m_muon_writer->flush();
    if (m_track_writer) m_track_writer->flush();
    delete m_electron_writer;
    delete m_muon_writer;
    delete m_track_writer;
}

void ObjectWriters::extract_vertex_z0(const xAOD::VertexContainer& primary_vertices) {
    m_primary_vertices_z0.clear();
    for (const xAOD::Vertex *vertex : primary_vertices) {
        m_primary_vertices_z0.push_back(vertex->z());
    }
}

void ObjectWriters::write(std::vector<const xAOD::Electron*> electrons, std::vector<const xAOD::Muon*> muons, std::vector<const xAOD::TrackParticle*> tracks, const xAOD::VertexContainer& primary_vertices) {

    m_current_electrons = electrons;
    m_current_muons = muons;
    m_current_tracks = tracks;

    // extract primary vertex z0 values
    extract_vertex_z0(primary_vertices);

    // sort objects by descending pT
    auto sort_objects = [](const auto* t1, const auto* t2) {
          return t1->pt() > t2->pt();
    };
    std::sort(m_current_electrons.begin(), m_current_electrons.end(), sort_objects);
    std::sort(m_current_muons.begin(), m_current_muons.end(), sort_objects);
    std::sort(m_current_tracks.begin(), m_current_tracks.end(), sort_objects);

    // write objects
    m_electron_writer->fillWhileIncrementing(m_electron_idx);
    m_muon_writer->fillWhileIncrementing(m_muon_idx);
    m_track_writer->fillWhileIncrementing(m_track_idx);
}
