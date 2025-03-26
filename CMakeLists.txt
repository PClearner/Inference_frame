cmake_minimum_required(VERSION 3.10.0)
project(inference_frame VERSION 0.1.0 LANGUAGES C CXX)

# add_executable(inference_frame main.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/usr/local/include)

link_directories(/usr/local/lib)

add_executable(tensor_test 
        ${CMAKE_SOURCE_DIR}/test/tensor_test.cpp 
        ${CMAKE_SOURCE_DIR}/src/Tensor.cpp)


find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
target_link_libraries(tensor_test glog::glog ${ARMADILLO_LIBRARIES})


