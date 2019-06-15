#ifndef TRACK_FILTER_H
#define TRACK_FILTER_H

namespace xAOD {
    class TrackParticle_v1;
    typedef TrackParticle_v1 TrackParticle;
}

// ATLAS things
#include "xAODTracking/TrackParticleContainer.h"
#include "xAODTracking/TrackParticle.h"
#include "xAODEgamma/EgammaxAODHelpers.h"
#include "InDetTrackSelectionTool/InDetTrackSelectionTool.h"

class TrackFilter
{
    public:
        // constructor and destructor
        TrackFilter();
        ~TrackFilter();

        // track selection
        std::vector<const xAOD::TrackParticle*> filter_tracks(const xAOD::TrackParticleContainer& tracks);

        // function that's actually called to write the event
        void write(const xAOD::TrackParticleContainer& tracks);

    private:
        // track selector
        InDet::InDetTrackSelectionTool *m_trkseltool;
};

#endif
