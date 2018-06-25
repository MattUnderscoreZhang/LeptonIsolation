// this class's header
#include "TrackWriter.h"

// HDF5 things
#include "HDF5Utils/HdfTuple.h"
#include "H5Cpp.h"

// ATLAS things
#include "xAODTracking/TrackParticle.h"


TrackWriter::TrackWriter(H5::Group& output_group):
    m_current_track(nullptr),
    m_writer(nullptr)
{
    // define the variable filling functions. Each function takes no
    // arguments, but includes a pointer to the class instance, and by
    // extension to the current track.
    H5Utils::VariableFillers fillers;

    fillers.add<int>("eventN",
        [this]() {return this->eventN;}
    );
    fillers.add<float>("pT",
        [this]() {return this->m_current_track->pt();}
    );
    fillers.add<float>("eta",
        [this]() {return this->m_current_track->eta();}
    );
    fillers.add<float>("phi",
        [this]() {return this->m_current_track->phi();}
    );
    fillers.add<float>("charge",
        [this]() {return this->m_current_track->charge();}
    );
    fillers.add<float>("d0",
        [this]() {return this->m_current_track->d0();}
    );
    fillers.add<float>("z0",
        [this]() {return this->m_current_track->z0();}
    );
    fillers.add<float>("chiSquared",
        [this]() {return this->m_current_track->chiSquared();}
    );

    m_writer = new H5Utils::WriterXd(output_group, "tracks", fillers, {});
}

TrackWriter::~TrackWriter() {
    if (m_writer) m_writer->flush();
    delete m_writer;
}

void TrackWriter::write(const xAOD::TrackParticle& track, int eventN) {
    m_current_track = &track;
    m_writer->fillWhileIncrementing();
    this->eventN = eventN;
}
