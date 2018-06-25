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

        // function that's actually called to write the electron
        void write(const xAOD::Electron& electron, int eventN);

    private:
        // the functions that fill the output need to be defined when the
        // class is initialized. They will fill from this electron pointer, which
        // must be updated each time we wright.
        const xAOD::Electron* m_current_electron;

        // event number
        int eventN;

        // The writer itself
        H5Utils::WriterXd* m_writer;
};

#endif
