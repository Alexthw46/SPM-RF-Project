cmake_minimum_required(VERSION 3.28)
project(spm_project)

find_path(FASTFLOW_ROOT ff/ff.hpp PATHS $ENV{FF_HOME} $ENV{HOME}/fastflow)

if (FASTFLOW_ROOT)
    message(STATUS "FastFlow headers found: ${FASTFLOW_ROOT}")
    include_directories(${FASTFLOW_ROOT})
else ()
    message(FATAL_ERROR "FastFlow headers not found. Set FF_HOME or install fastflow in $HOME/fastflow.")
endif ()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -fno-inline-functions")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -DNDEBUG")

include_directories(include)

add_executable(Comparison
        src/Common/main.cpp
        src/Common/DecisionTree.cpp
        src/Common/RandomForest.cpp
)

add_executable(RandomForestSeq
        src/Naive/main.cpp
        src/Naive/DecisionTree.cpp
        src/Naive/RandomForest.cpp
)

add_executable(RandomForestMPI
        src/MPI/main.cpp
        src/MPI/RandomForest.cpp
        src/Common/DecisionTree.cpp
)

target_link_libraries(Comparison PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(RandomForestMPI PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)
