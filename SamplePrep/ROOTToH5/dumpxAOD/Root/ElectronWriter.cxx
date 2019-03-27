// this class's header
#include "ElectronWriter.h"

// EDM things
#include "xAODEgamma/ElectronContainer.h"
#include "xAODTruth/xAODTruthHelpers.h"
#include "xAODEgamma/Electron.h"

// HDF5 things
#include "HDF5Utils/HdfTuple.h"
#include "H5Cpp.h"

ElectronWriter::ElectronWriter(H5::Group& output_group):
    m_electron_idx(1),
    m_writer(nullptr)
{

    // define the variable filling functions. Each function takes no
    // arguments, but includes a pointer to the class instance, and by
    // extension to the current electron.
    H5Utils::VariableFillers fillers;

    fillers.add<int>("pdgID",
        [this]() {
            return 11;
        }
    );
    fillers.add<float>("pT",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (float)(this->m_current_electrons.at(idx)->pt());
        }
    );
    fillers.add<float>("eta",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (float)(this->m_current_electrons.at(idx)->eta());
        }
    );
    fillers.add<float>("phi",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (float)(this->m_current_electrons.at(idx)->phi());
        }
    );
    fillers.add<float>("d0",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (float)(this->m_current_electrons.at(idx)->trackParticle()->d0());
        }
    );
    fillers.add<float>("z0",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (float)(this->m_current_electrons.at(idx)->trackParticle()->z0());
        }
    );
    fillers.add<float>("ptcone20",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            float ptcone20 = 0.0;
            this->m_current_electrons.at(idx)->isolationValue(ptcone20,xAOD::Iso::ptcone20);
            return ptcone20;
        }
    );
    fillers.add<float>("ptcone30",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            float ptcone30 = 0.0;
            this->m_current_electrons.at(idx)->isolationValue(ptcone30,xAOD::Iso::ptcone30);
            return ptcone30;
        }
    );
    fillers.add<float>("ptcone40",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            float ptcone40 = 0.0;
            this->m_current_electrons.at(idx)->isolationValue(ptcone40,xAOD::Iso::ptcone40);
            return ptcone40;
        }
    );
    fillers.add<float>("ptvarcone20",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            float ptvarcone20 = 0.0;
            this->m_current_electrons.at(idx)->isolationValue(ptvarcone20,xAOD::Iso::ptvarcone20);
            return ptvarcone20;
        }
    );
    fillers.add<float>("ptvarcone30",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            float ptvarcone30 = 0.0;
            this->m_current_electrons.at(idx)->isolationValue(ptvarcone30,xAOD::Iso::ptvarcone30);
            return ptvarcone30;
        }
    );
    fillers.add<float>("ptvarcone40",
        [this]() {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            float ptvarcone40 = 0.0;
            this->m_current_electrons.at(idx)->isolationValue(ptvarcone40,xAOD::Iso::ptvarcone40);
            return ptvarcone40;
        }
    );
    fillers.add<int>("truth_type",
        [this]() -> int {
            size_t idx = this->m_electron_idx.at(0);
            if (this->m_current_electrons.size() <= idx) return NAN;
            return (int)(xAOD::TruthHelpers::getParticleTruthType(*(this->m_current_electrons.at(idx))));
            // 2 = real prompt, 3 = HF
        }
    );
    fillers.add<float>("PLT",
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
    m_writer = new H5Utils::WriterXd(output_group, "electrons", fillers, {20});
}

ElectronWriter::~ElectronWriter() {
    if (m_writer) m_writer->flush();
    delete m_writer;
}

void ElectronWriter::filter_electrons_first_stage(const xAOD::ElectronContainer& electrons) {
    m_current_electrons.clear();
    for (const xAOD::Electron *electron : electrons) {
        if(cacc_lhmedium.isAvailable(*electron) ){
            if (!cacc_lhmedium(*electron)) continue;
            m_current_electrons.push_back(electron);
        }
    }
}

void ElectronWriter::write(const xAOD::ElectronContainer& electrons) {

    // electron selection
    filter_electrons_first_stage(electrons);

    // sort electrons by descending pT
    std::sort(m_current_electrons.begin(), m_current_electrons.end(),
        [](const auto* t1, const auto* t2) {
          return t1->pt() > t2->pt();
    });

    // write electrons
    m_writer->fillWhileIncrementing(m_electron_idx);
}
