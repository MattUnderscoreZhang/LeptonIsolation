#include "../headers/ElectronFilter.h"

ElectronFilter::ElectronFilter() {
}

ElectronFilter::~ElectronFilter() {
}

std::vector<const xAOD::Electron*> ElectronFilter::filter_electrons(const xAOD::ElectronContainer& electrons) {

    std::vector<const xAOD::Electron*> m_current_electrons;

    for (const xAOD::Electron *electron : electrons) {
        if(cacc_lhmedium.isAvailable(*electron) ){
            if (!cacc_lhmedium(*electron)) continue;
            m_current_electrons.push_back(electron);
        }
    }

    return m_current_electrons;
}
