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

        // function that's actually called to write the muon
        void write(const xAOD::Muon& muon);

    private:
        // the functions that fill the output need to be defined when the
        // class is initialized. They will fill from this muon pointer, which
        // must be updated each time we wright.
        const xAOD::Muon* m_current_muon;

        // The writer itself
        H5Utils::WriterXd* m_writer;
};

#endif
