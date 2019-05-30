#ifndef MUON_WRITER_H
#define MUON_WRITER_H

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
    class Muon_v1;
    typedef Muon_v1 Muon;
}

// EDM includes
#include "xAODMuon/MuonContainer.h"
#include "MuonSelectorTools/MuonSelectionTool.h"

class MuonWriter
{
    public:
        // constructor: the writer will create the output dataset in some group
        MuonWriter(H5::Group& output_group);

        // destructor (takes care of flushing output file too)
        ~MuonWriter();

        // we want to disable copying and assignment, it's not trivial to
        // make this play well with output files
        MuonWriter(MuonWriter&) = delete;
        MuonWriter operator=(MuonWriter&) = delete;

        // muon selection
        void filter_muons_first_stage(const xAOD::MuonContainer& muons);

        // extract primary vertex z0 values
        void extract_vertex_z0(const xAOD::VertexContainer& primary_vertices);

        // function that's actually called to write the event
        void write(const xAOD::MuonContainer& muons, const xAOD::VertexContainer& primary_vertices);

    private:
        // the functions that fill the output need to be defined when the
        // class is initialized. They will fill from this muon pointer, which
        // must be updated each time we wright.
        std::vector<const xAOD::Muon*> m_current_muons;
        std::vector<size_t> m_muon_idx;
        std::vector<float> m_primary_vertices_z0;

        // selector tool
        CP::MuonSelectionTool* m_muonSelectionTool;

        // The writer itself
        H5Utils::WriterXd* m_writer;
};

#endif
