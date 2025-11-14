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

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_executable(RandomForestSeq
        src/Seq/main.cpp
        src/Seq/DecisionTree.cpp
        src/Seq/RandomForest.cpp
)

add_executable(RandomForestOptimized
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