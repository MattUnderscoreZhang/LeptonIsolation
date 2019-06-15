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
        // constructor
        TrackFilter();

        // destructor
        ~TrackFilter();

        // we want to disable copying and assignment, it's not trivial to
        // make this play well with output files
        TrackFilter(TrackFilter&) = delete;
        TrackFilter operator=(TrackFilter&) = delete;

        // track selection
        std::vector<const xAOD::TrackParticle*>* filter_tracks(const xAOD::TrackParticleContainer& tracks);

        // function that's actually called to write the event
        void write(const xAOD::TrackParticleContainer& tracks);

    private:
        // track selector
        InDet::InDetTrackSelectionTool *m_trkseltool;
};

#endif
