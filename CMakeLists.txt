cmake_minimum_required(VERSION 3.10.0)
project(inference_frame VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
# 强制使用 C++17，不可回退，如果编译器不支持 C++ 17，直接报错
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# add_executable(inference_frame main.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/usr/local/include)

link_directories(/usr/local/lib)

add_executable(tensor_test 
        ${CMAKE_SOURCE_DIR}/test/tensor_test.cpp 
        ${CMAKE_SOURCE_DIR}/src/Tensor.cpp)


find_package(glog REQUIRED)
find_package(GTest REQUIRED)
find_package(Armadillo REQUIRED)
target_link_libraries(tensor_test glog::glog ${ARMADILLO_LIBRARIES})

aux_source_directory(${CMAKE_SOURCE_DIR}/src DIR_SOURCE_ARMA)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/runtime DIR_SOURCE_RUNTIME)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/layer/abstract DIR_SOURCE_LAYER_ABSTRACT)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/layer/detail/ DIR_SOURCE_LAYER_DETIAL)
# add_executable(graph_test 
#         ${CMAKE_SOURCE_DIR}/test/graph_test.cpp 
#         ${CMAKE_SOURCE_DIR}/src/Tensor.cpp
#         ${CMAKE_SOURCE_DIR}/src/load_data.cpp
#         ${CMAKE_SOURCE_DIR}/src/tensor_util.cpp
#         ${CMAKE_SOURCE_DIR}/src/runtime/runtime_attr.cpp
#         # ${CMAKE_SOURCE_DIR}/include/runtime/runtime_datatype.hpp
#         )

set(SOURCE
    ${CMAKE_SOURCE_DIR}/src/ir.cpp 
    ${CMAKE_SOURCE_DIR}/src/load_data.cpp 
    ${CMAKE_SOURCE_DIR}/src/store_zip.cpp 
    ${CMAKE_SOURCE_DIR}/src/tensor_util.cpp 
    ${CMAKE_SOURCE_DIR}/src/Tensor.cpp 
    ${CMAKE_SOURCE_DIR}/src/runtime/runtime_attr.cpp 
    ${CMAKE_SOURCE_DIR}/src/runtime/runtime_ir.cpp 
    ${CMAKE_SOURCE_DIR}/src/runtime/runtime_op.cpp 
    ${CMAKE_SOURCE_DIR}/src/layer/abstract/layer_factory.cpp 
    ${CMAKE_SOURCE_DIR}/src/layer/abstract/layer.cpp 
    ${CMAKE_SOURCE_DIR}/src/layer/abstract/non_param_layer.cpp 
    ${CMAKE_SOURCE_DIR}/src/layer/abstract/param_layer.cpp 
    ${CMAKE_SOURCE_DIR}/src/layer/detail/relu.cpp 
    ${CMAKE_SOURCE_DIR}/src/layer/detail/sigmoid.cpp
    ${CMAKE_SOURCE_DIR}/src/layer/detail/maxpooling.cpp
    ${CMAKE_SOURCE_DIR}/src/layer/detail/convolution.cpp
    ${CMAKE_SOURCE_DIR}/src/layer/detail/avgpooling.cpp
)

add_executable(graph_test 
        ${CMAKE_SOURCE_DIR}/test/graph_test.cpp 
        # ${DIR_SOURCE_ARMA}
        # ${DIR_SOURCE_RUNTIME}
        # ${CMAKE_SOURCE_DIR}/include/runtime/runtime_datatype.hpp
        ${SOURCE}
        )

target_link_libraries(graph_test glog::glog ${ARMADILLO_LIBRARIES})

set_target_properties(graph_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/graph
    )


add_executable(graph_build 
        ${CMAKE_SOURCE_DIR}/test/graph_build.cpp 
        # ${DIR_SOURCE_ARMA}
        # ${DIR_SOURCE_RUNTIME}
        ${SOURCE}
)

target_link_libraries(graph_build glog::glog  ${ARMADILLO_LIBRARIES})
set_target_properties(graph_build PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/graph
    )


add_executable(layer_test 
    ${CMAKE_SOURCE_DIR}/test/layer_test.cpp 
#     ${DIR_SOURCE_ARMA}
#     ${DIR_SOURCE_RUNTIME}
#     ${DIR_SOURCE_LAYER_ABSTRACT}
#     ${DIR_SOURCE_LAYER_DETIAL}
    ${SOURCE}
)

target_link_libraries(layer_test glog::glog  ${ARMADILLO_LIBRARIES})
set_target_properties(layer_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/layer
    )

add_executable(conv_test
    ${CMAKE_SOURCE_DIR}/test/conv_test.cpp
    ${SOURCE}
)

target_link_libraries(conv_test glog::glog  ${ARMADILLO_LIBRARIES})
set_target_properties(conv_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/conv
    )

