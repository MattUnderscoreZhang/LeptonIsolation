#ifndef MUON_FILTER_H
#define MUON_FILTER_H

namespace xAOD {
    class Muon_v1;
    typedef Muon_v1 Muon;
}

// ATLAS things
#include "xAODMuon/MuonContainer.h"
#include "xAODMuon/Muon.h"
#include "MuonSelectorTools/MuonSelectionTool.h"

class MuonFilter
{
    public:
        // constructor and destructor
        MuonFilter();
        ~MuonFilter();

        // muon selection
        std::vector<const xAOD::Muon*> filter_muons(const xAOD::MuonContainer& muons);

    private:
        // muon selector
        CP::MuonSelectionTool* m_muonSelectionTool;
};

#endif
