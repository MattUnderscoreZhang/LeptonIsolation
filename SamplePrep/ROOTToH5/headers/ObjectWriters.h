#ifndef OBJECT_WRITERS_H
#define OBJECT_WRITERS_H

// forward declare HDF5 things
namespace H5 {
    class Group;
}
namespace H5Utils {
    class VariableFillers;
    class WriterXd;
}

// forward declare EDM things
namespace xAOD {
    class Electron_v1;
    typedef Electron_v1 Electron;
    class Muon_v1;
    typedef Muon_v1 Muon;
    class TrackParticle_v1;
    typedef TrackParticle_v1 TrackParticle;
}

// HDF5 things
#include "HDF5Utils/HdfTuple.h"
#include "H5Cpp.h"

// EDM includes
#include "xAODEgamma/ElectronContainer.h"
#include "xAODEgamma/Electron.h"
#include "xAODMuon/MuonContainer.h"
#include "xAODMuon/Muon.h"
#include "xAODTracking/TrackParticleContainer.h"
#include "xAODTracking/TrackParticle.h"
#include "xAODEgamma/EgammaxAODHelpers.h"
#include "xAODTruth/xAODTruthHelpers.h"
#include "xAODTracking/TrackParticlexAODHelpers.h"
#include "InDetTrackSelectionTool/InDetTrackSelectionTool.h"

// etcone
#include <fastjet/tools/JetMedianBackgroundEstimator.hh>
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include <fastjet/AreaDefinition.hh>
#include <fastjet/Selector.hh>

class ObjectWriters
{
    public:
        // constructor: the writer will create the output dataset in some group
        ObjectWriters(H5::Group& output_group);

        // destructor (takes care of flushing output file too)
        ~ObjectWriters();

        // we want to disable copying and assignment, it's not trivial to
        // make this play well with output files
        ObjectWriters(ObjectWriters&) = delete;
        ObjectWriters operator=(ObjectWriters&) = delete;

        // extract primary vertex z0 values
        std::vector<float> extract_vertex_z0(const xAOD::VertexContainer& primary_vertices);

        // function that's actually called to write the event
        void write(std::vector<const xAOD::Electron*> electrons, std::vector<const xAOD::Muon*> muons, std::vector<const xAOD::TrackParticle*> tracks, const xAOD::Vertex* primary_vertex, const xAOD::CaloClusterContainer* calo_clusters);

    private:
        // vectors relating to the current event
        std::vector<const xAOD::Electron*> m_current_electrons;
        std::vector<const xAOD::Muon*> m_current_muons;
        std::vector<const xAOD::TrackParticle*> m_current_tracks;
        const xAOD::Vertex* m_current_primary_vertex;
        const xAOD::CaloClusterContainer* m_current_calo_clusters;

        // track selector
        InDet::InDetTrackSelectionTool *m_trkseltool;

        // for writing
        std::vector<size_t> m_electron_idx;
        H5Utils::WriterXd* m_electron_writer;
        std::vector<size_t> m_muon_idx;
        H5Utils::WriterXd* m_muon_writer;
        std::vector<size_t> m_track_idx;
        H5Utils::WriterXd* m_track_writer;

        // etcone
        fastjet::JetMedianBackgroundEstimator* gmbec;
        fastjet::JetMedianBackgroundEstimator* gmbef;
        fastjet::JetMedianBackgroundEstimator* gmbec_pf;
        fastjet::JetMedianBackgroundEstimator* gmbef_pf;
};

#endif
