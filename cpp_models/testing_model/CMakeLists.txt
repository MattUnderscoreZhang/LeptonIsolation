cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(predictor)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_PREFIX_PATH "libs/libtorch")

include_directories(include libs)


file(GLOB SOURCES "src/*.cpp")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(predict ${SOURCES})
target_link_libraries(predict "${TORCH_LIBRARIES}")
set_property(TARGET predict PROPERTY CXX_STANDARD 14)
