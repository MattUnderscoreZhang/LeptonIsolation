#ifndef ELECTRON_WRITER_H
#define ELECTRON_WRITER_H

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
}

// EDM includes
#include "xAODEgamma/ElectronContainer.h"

class ElectronWriter
{
    public:
        // constructor: the writer will create the output dataset in some group
        ElectronWriter(H5::Group& output_group);

        // destructor (takes care of flushing output file too)
        ~ElectronWriter();

        // we want to disable copying and assignment, it's not trivial to
        // make this play well with output files
        ElectronWriter(ElectronWriter&) = delete;
        ElectronWriter operator=(ElectronWriter&) = delete;

        // extract primary vertex z0 values
        void extract_vertex_z0(const xAOD::VertexContainer& primary_vertices);

        // function that's actually called to write the event
        void write(std::vector<const xAOD::Electron*> electrons, const xAOD::VertexContainer& primary_vertices);

    private:
        // the functions that fill the output need to be defined when the
        // class is initialized. They will fill from this electron pointer, which
        // must be updated each time we write.
        std::vector<const xAOD::Electron*> m_current_electrons;
        std::vector<size_t> m_electron_idx;
        std::vector<float> m_primary_vertices_z0;

        // The writer itself
        H5Utils::WriterXd* m_writer;
};

#endif
