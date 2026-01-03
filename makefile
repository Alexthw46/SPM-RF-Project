# Makefile

# Tools
CXX := g++
MPICXX := $(shell which mpicxx 2>/dev/null)
ifeq ($(MPICXX),)
MPICXX := g++
endif

# Build type: set BUILD=Debug for debug flags
BUILD ?= Release

# FastFlow include: prefer FF_HOME, else $HOME/fastflow
FASTFLOW_DIR := $(if $(FF_HOME),$(FF_HOME),$(HOME)/fastflow)
ifeq ($(wildcard $(FASTFLOW_DIR)/ff/ff.hpp),)
$(error FastFlow headers not found. Set FF_HOME or install fastflow in $(HOME)/fastflow.)
endif

INCLUDES := -Iinclude -I$(FASTFLOW_DIR)

# Common flags
CXXFLAGS := -std=c++20 -Wall $(INCLUDES)
ifeq ($(BUILD),Debug)
CXXFLAGS += -g -fno-inline-functions
else
CXXFLAGS += -O3 -ffast-math -DNDEBUG
endif

OPENMP_FLAGS := -fopenmp

# Sources per target
COMPARISON_SRCS := src/Common/main.cpp src/Common/DecisionTree.cpp src/Common/RandomForest.cpp
SEQ_SRCS        := src/Naive/main.cpp src/Naive/DecisionTree.cpp src/Naive/RandomForest.cpp
OPT_SRCS        := src/Optimized/main.cpp src/Optimized/RandomForest.cpp src/Common/DecisionTree.cpp
FF_SRCS         := src/FF/main.cpp src/FF/RandomForest.cpp src/Common/DecisionTree.cpp
OMP_SRCS        := src/OMP/main.cpp src/OMP/RandomForest.cpp src/Common/DecisionTree.cpp
MPI_SRCS        := src/MPI/main.cpp src/MPI/RandomForest.cpp src/Common/DecisionTree.cpp

# Object lists (replace .cpp -> .o)
COMPARISON_OBJS := $(COMPARISON_SRCS:.cpp=.o)
SEQ_OBJS        := $(SEQ_SRCS:.cpp=.o)
OPT_OBJS        := $(OPT_SRCS:.cpp=.o)
FF_OBJS         := $(FF_SRCS:.cpp=.o)
OMP_OBJS        := $(OMP_SRCS:.cpp=.o)
MPI_OBJS        := $(MPI_SRCS:.cpp=.o)

# Targets
ALL_TARGETS := Comparison RandomForestSeq RandomForestOptimized RandomForestFF RandomForestOMP RandomForestMPI

.PHONY: all clean
all: $(ALL_TARGETS)

# Link rules
Comparison: $(COMPARISON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENMP_FLAGS)

RandomForestSeq: $(SEQ_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

RandomForestOptimized: $(OPT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

RandomForestFF: $(FF_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

RandomForestOMP: $(OMP_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENMP_FLAGS)

RandomForestMPI: $(MPI_OBJS)
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(OPENMP_FLAGS)

# MPI-specific compile rule
src/MPI/%.o: src/MPI/%.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(CXXFLAGS) $(OPENMP_FLAGS) -c $< -o $@

# Generic compile rule
%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -c $< -o $@

clean:
	-rm -f $(ALL_TARGETS) */*.o */*/*.o */*/*/*.o
