cmake_minimum_required(VERSION 3.10.0)
project(inference_frame VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
# 强制使用 C++17，不可回退，如果编译器不支持 C++ 17，直接报错
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
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

aux_source_directory(./src DIR_SOURCE_ARMA)
aux_source_directory(./src/runtime DIR_SOURCE_RUNTIME)
# add_executable(graph_test 
#         ${CMAKE_SOURCE_DIR}/test/graph_test.cpp 
#         ${CMAKE_SOURCE_DIR}/src/Tensor.cpp
#         ${CMAKE_SOURCE_DIR}/src/load_data.cpp
#         ${CMAKE_SOURCE_DIR}/src/tensor_util.cpp
#         ${CMAKE_SOURCE_DIR}/src/runtime/runtime_attr.cpp
#         # ${CMAKE_SOURCE_DIR}/include/runtime/runtime_datatype.hpp
#         )

add_executable(graph_test 
        ${CMAKE_SOURCE_DIR}/test/graph_test.cpp 
        ${DIR_SOURCE_ARMA}
        ${DIR_SOURCE_RUNTIME}
        ${CMAKE_SOURCE_DIR}/include/runtime/runtime_datatype.hpp)

target_link_libraries(graph_test glog::glog ${ARMADILLO_LIBRARIES})

set_target_properties(graph_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/graph
    )

