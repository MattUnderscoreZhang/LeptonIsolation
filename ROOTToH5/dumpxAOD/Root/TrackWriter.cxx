// this class's header
#include "TrackWriter.h"

// EDM things
#include "xAODTracking/TrackParticleContainer.h"

// HDF5 things
#include "HDF5Utils/HdfTuple.h"
#include "H5Cpp.h"

// ATLAS things
#include "xAODTracking/TrackParticle.h"

TrackWriter::TrackWriter(H5::Group& output_group):
    m_track_idx(1),
    m_writer(nullptr)
{

    // define the variable filling functions. Each function takes no
    // arguments, but includes a pointer to the class instance, and by
    // extension to the current event.
    H5Utils::VariableFillers fillers;

    fillers.add<float>("pT",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->pt());
        }
    );
    fillers.add<float>("eta",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->eta());
        }
    );
    fillers.add<float>("phi",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->phi());
        }
    );
    fillers.add<float>("charge",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->charge());
        }
    );
    fillers.add<float>("d0",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->d0());
        }
    );
    fillers.add<float>("z0",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->z0());
        }
    );
    fillers.add<float>("chiSquared",
        [this]() {
            size_t idx = this->m_track_idx.at(0);
            if (this->m_current_tracks.size() <= idx) return NAN;
            return (float)(this->m_current_tracks.at(idx)->chiSquared());
        }
    );

    // Save up to 3000 tracks per event
    m_writer = new H5Utils::WriterXd(output_group, "tracks", fillers, {3000});
}

TrackWriter::~TrackWriter() {
    if (m_writer) m_writer->flush();
    delete m_writer;
}

void TrackWriter::write(const xAOD::TrackParticleContainer& tracks) {

    m_current_tracks.clear();
    for (const xAOD::TrackParticle *track : tracks) {
        m_current_tracks.push_back(track);
    }

    // Sort tracks by descending pT
    std::sort(m_current_tracks.begin(), m_current_tracks.end(),
        [](const auto* t1, const auto* t2) {
          return t1->pt() > t2->pt();
    });

    m_writer->fillWhileIncrementing(m_track_idx);
}
