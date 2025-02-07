cmake_minimum_required(VERSION 3.5)
project(vad)
option(BUILD_ONNX "Build on onnx runtime." OFF)
option(BUILD_NNCASE "Build on nncase." ON)

if (BUILD_ONNX)
    add_definitions(-DONNX)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

if (BUILD_ONNX)
set(ONNXRUNTIME_PATH ${CMAKE_SOURCE_DIR}/3rd_party/onnxruntime/onnxruntime-linux-x64-1.12.1)
include_directories(${ONNXRUNTIME_PATH}/include)
link_directories(${ONNXRUNTIME_PATH}/lib)
else()
if(CMAKE_CROSSCOMPILING)
set(NNCASE_PATH ${CMAKE_SOURCE_DIR}/3rd_party/nncase/riscv64)
link_directories(${CMAKE_SOURCE_DIR}/3rd_party/mmz/riscv64)
else()
set(NNCASE_PATH ${CMAKE_SOURCE_DIR}/3rd_party/nncase/x86_64)
endif()
include_directories(3rd_party
                    ${NNCASE_PATH}/include
                    ${NNCASE_PATH}/include/nncase/runtime
                    )
link_directories(${NNCASE_PATH}/lib)
endif()



include_directories(${CMAKE_SOURCE_DIR}/examples/cpp/)


set(bin silero-vad)
add_executable(${bin} ${CMAKE_SOURCE_DIR}/examples/cpp/silero-vad.cpp)

if (BUILD_ONNX)
target_link_libraries(${bin} PRIVATE onnxruntime)
else()
if(CMAKE_CROSSCOMPILING)
target_link_libraries(${bin} PUBLIC nncase.rt_modules.k230 Nncase.Runtime.Native functional_k230 mmz)
else()
target_link_libraries(${bin} PUBLIC Nncase.Runtime.Native)
endif()
endif()