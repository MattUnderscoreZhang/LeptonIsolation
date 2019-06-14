cmake_minimum_required( VERSION 3.2 FATAL_ERROR )

# get analysis base
find_package( AnalysisBase 21.2 REQUIRED )

# set up a work directory project:
atlas_project( WorkDir 1.0 USE AnalysisBase ${AnalysisBase_VERSION} )

# set the name of the package:
atlas_subdir( LeptonTrackROOTToH5 )

# packages that this package depends on
atlas_depends_on_subdirs(
  PRIVATE
  Control/xAODRootAccess
  Event/xAOD/xAODTrack
  Event/xAOD/xAODTruth
  Event/xAOD/xAODEgamma
  Event/xAOD/xAODMuon
  PhysicsAnalysis/AnalysisCommon/HDF5Utils
)

# external(s) used by the package:
find_package(ROOT REQUIRED COMPONENTS RIO Hist Tree Net Core)
find_package(HDF5 1.10.1 REQUIRED COMPONENTS CXX C)
find_package(lwtnn)

# common requirements
set(_common
  src/TrackWriter.cxx
  src/ElectronWriter.cxx
  src/MuonWriter.cxx
  INCLUDE_DIRS ${ROOT_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${LWTNN_INCLUDE_DIRS}
  LINK_LIBRARIES ${ROOT_LIBRARIES} ${HDF5_LIBRARIES} ${LWTNN_LIBRARIES}
  xAODRootAccess
  xAODTracking
  xAODTruth
  xAODEgamma
  xAODMuon
  HDF5Utils
  MuonSelectorToolsLib
  ElectronPhotonSelectorToolsLib
)

# build the test executable:
atlas_add_executable( ROOT_to_H5 src/ROOT_to_H5.cxx ${_common} )

# Set up grid magic. Atlas uses CPack to package up the local files
# and submit them to the grid. If you're not looking to run on the
# grid this is unneeded.
atlas_cpack_setup()
