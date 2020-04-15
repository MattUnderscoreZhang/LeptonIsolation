cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(predictor)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_PREFIX_PATH "libs/libtorch")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} $ENV{ROOTSYS}/etc/cmake)
list(APPEND CMAKE_PREFIX_PATH  $ENV{ROOTSYS})
include_directories(include libs)


file(GLOB SOURCES "src/*.cpp")
find_package(Torch REQUIRED)
find_package(ROOT CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${ROOT_CXX_FLAGS}")

include(${ROOT_USE_FILE})

separate_arguments(ROOT_EXE_LINKER_FLAGS)
# string(STRIP ${SOURCES} SOURCES)
add_executable(predict ${SOURCES})
target_link_libraries(predict PUBLIC ${TORCH_LIBRARIES} ${ROOT_LIBRARIES})
set_property(TARGET predict PROPERTY CXX_STANDARD 14)
