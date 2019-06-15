#ifndef ELECTRON_FILTER_H
#define ELECTRON_FILTER_H

namespace xAOD {
    class Electron_v1;
    typedef Electron_v1 Electron;
}

// ATLAS things
#include "xAODEgamma/ElectronContainer.h"
#include "xAODEgamma/Electron.h"

static SG::AuxElement::ConstAccessor<char> cacc_lhmedium("DFCommonElectronsLHMedium");

class ElectronFilter
{
    public:
        // constructor and destructor
        ElectronFilter();
        ~ElectronFilter();

        // electron selection
        std::vector<const xAOD::Electron*> filter_electrons(const xAOD::ElectronContainer& electrons);

    private:
        // electron selector
};

#endif
