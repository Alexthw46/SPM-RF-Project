cmake_minimum_required(VERSION 3.28)
project(spm_project)

# cerca ff/ff.hpp; considera anche variabile d'ambiente FF_HOME o ~/fastflow
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

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -g -fno-inline-functions")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -DNDEBUG")

include_directories(include)

add_executable(RandomForestSeq
        src/Seq/main.cpp
        src/Seq/DecisionTree.cpp
        src/Seq/RandomForest.cpp
)

add_executable(RandomForestIndexed
        src/Optimized/main.cpp
        src/Optimized/DecisionTree.cpp
        src/Optimized/RandomForest.cpp
)

add_executable(RandomForestFF
        src/FF/main.cpp
        src/FF/DecisionTree.cpp
        src/FF/RandomForest.cpp
)

add_executable(RandomForestOMP
        src/OMP/main.cpp
        src/OMP/DecisionTree.cpp
        src/OMP/RandomForest.cpp
)

add_executable(RandomForestMPI
        src/MPI/main.cpp
        src/MPI/DecisionTree.cpp
        src/MPI/RandomForest.cpp
)

target_link_libraries(RandomForestOMP PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(RandomForestMPI PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)