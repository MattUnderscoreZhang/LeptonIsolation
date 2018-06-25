// this class's header
#include "ElectronWriter.h"

// HDF5 things
#include "HDF5Utils/HdfTuple.h"
#include "H5Cpp.h"

// ATLAS things
#include "xAODEgamma/Electron.h"


ElectronWriter::ElectronWriter(H5::Group& output_group):
    m_current_electron(nullptr),
    m_writer(nullptr)
{
    // define the variable filling functions. Each function takes no
    // arguments, but includes a pointer to the class instance, and by
    // extension to the current electron.
    H5Utils::VariableFillers fillers;

    fillers.add<int>("eventN",
        [this]() {return this->eventN;}
    );
    fillers.add<float>("pT",
        [this]() {return this->m_current_electron->pt();}
    );
    fillers.add<float>("eta",
        [this]() {return this->m_current_electron->eta();}
    );
    fillers.add<float>("phi",
        [this]() {return this->m_current_electron->phi();}
    );
    //fillers.add<float>("pdgID",
        //[this]() {return this->m_current_electron->pdgId();}
    //);
    fillers.add<float>("d0",
        [this]() {return this->m_current_electron->trackParticle()->d0();}
    );
    fillers.add<float>("z0",
        [this]() {return this->m_current_electron->trackParticle()->z0();}
    );

    m_writer = new H5Utils::WriterXd(output_group, "electrons", fillers, {});
}

ElectronWriter::~ElectronWriter() {
    if (m_writer) m_writer->flush();
    delete m_writer;
}

void ElectronWriter::write(const xAOD::Electron& electron, int eventN) {
    m_current_electron = &electron;
    m_writer->fillWhileIncrementing();
    this->eventN = eventN;
}
