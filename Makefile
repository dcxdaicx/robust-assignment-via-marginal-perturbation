CXX ?= g++
CXXFLAGS ?= -O3
CPP_STANDARD_FLAGS := -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS ?=

BIN_DIR := build/bin
CPP_BINARY := $(BIN_DIR)/bvn

.PHONY: all cpp rebuild clean

all: cpp

cpp: $(CPP_BINARY)

$(BIN_DIR):
	mkdir -p $@

$(BIN_DIR)/bvn: cpp/bvn.cpp | $(BIN_DIR)
	$(CXX) $(CPP_STANDARD_FLAGS) $(CXXFLAGS) $< $(LDFLAGS) -o $@

rebuild: clean cpp

clean:
	$(RM) -r build
