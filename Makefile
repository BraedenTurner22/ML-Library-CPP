# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDES = -I./include

# Directories
SRC_DIR = src
BUILD_DIR = build
LIB_DIR = lib

# Source files and objects
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Library name
LIB = $(LIB_DIR)/libml.a

# Default target
all: directories $(LIB)

# Create build directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(LIB_DIR)

# Build static library
$(LIB): $(OBJS)
	ar rcs $@ $^

# Compile source files to objects
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)

# Rebuild from scratch
rebuild: clean all

.PHONY: all clean rebuild directories
