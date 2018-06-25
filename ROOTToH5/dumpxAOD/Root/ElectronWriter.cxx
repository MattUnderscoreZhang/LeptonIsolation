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
    fillers.add<float>("ptcone20",
        [this]() {
            float ptcone20 = 0.0;
            this->m_current_electron->isolationValue(ptcone20,xAOD::Iso::ptcone20);
            return ptcone20;
        }
    );
    fillers.add<float>("ptcone30",
        [this]() {
            float ptcone30 = 0.0;
            this->m_current_electron->isolationValue(ptcone30,xAOD::Iso::ptcone30);
            return ptcone30;
        }
    );
    fillers.add<float>("ptcone40",
        [this]() {
            float ptcone40 = 0.0;
            this->m_current_electron->isolationValue(ptcone40,xAOD::Iso::ptcone40);
            return ptcone40;
        }
    );
    fillers.add<float>("ptvarcone20",
        [this]() {
            float ptvarcone20 = 0.0;
            this->m_current_electron->isolationValue(ptvarcone20,xAOD::Iso::ptvarcone20);
            return ptvarcone20;
        }
    );
    fillers.add<float>("ptvarcone30",
        [this]() {
            float ptvarcone30 = 0.0;
            this->m_current_electron->isolationValue(ptvarcone30,xAOD::Iso::ptvarcone30);
            return ptvarcone30;
        }
    );
    fillers.add<float>("ptvarcone40",
        [this]() {
            float ptvarcone40 = 0.0;
            this->m_current_electron->isolationValue(ptvarcone40,xAOD::Iso::ptvarcone40);
            return ptvarcone40;
        }
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
