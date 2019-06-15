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

// EDM includes
#include "xAODEgamma/ElectronContainer.h"
#include "xAODMuon/MuonContainer.h"
#include "xAODTracking/TrackParticleContainer.h"
#include "xAODTracking/TrackParticle.h"
#include "xAODEgamma/EgammaxAODHelpers.h"

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
        void extract_vertex_z0(const xAOD::VertexContainer& primary_vertices);

        // function that's actually called to write the event
        void write(std::vector<const xAOD::Electron*> electrons, std::vector<const xAOD::Muon*> muons, std::vector<const xAOD::TrackParticle*> tracks, const xAOD::VertexContainer& primary_vertices);

    private:
        // the functions that fill the output need to be defined when the
        // class is initialized. They will fill from this electron pointer, which
        // must be updated each time we write.
        std::vector<const xAOD::Electron*> m_current_electrons;
        std::vector<size_t> m_electron_idx;
        std::vector<const xAOD::Muon*> m_current_muons;
        std::vector<size_t> m_muon_idx;
        std::vector<const xAOD::TrackParticle*> m_current_tracks;
        std::vector<size_t> m_track_idx;
        std::vector<float> m_primary_vertices_z0;

        // The writers themselves
        H5Utils::WriterXd* m_electron_writer;
        H5Utils::WriterXd* m_muon_writer;
        H5Utils::WriterXd* m_track_writer;
};

#endif
