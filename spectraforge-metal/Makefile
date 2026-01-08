# SpectraForge Metal - Makefile for Apple Silicon
# Requires macOS with Metal support and Xcode command line tools

# Use system clang (not homebrew LLVM) for proper module support
CC = /usr/bin/clang
OBJC = /usr/bin/clang

# Compiler flags
CFLAGS = -Wall -Wextra -O3 -std=c17
OBJCFLAGS = -Wall -Wextra -O3 -fobjc-arc -fmodules
LDFLAGS = -framework Metal -framework MetalKit -framework Foundation -framework CoreGraphics -framework QuartzCore -framework Cocoa

# For debug builds
DEBUG_CFLAGS = -Wall -g -O0 -DDEBUG
DEBUG_OBJCFLAGS = -Wall -g -O0 -DDEBUG -fobjc-arc -fmodules

# Source directories
SRC_DIR = src
SHADER_DIR = shaders
INCLUDE_DIR = include
LIB_DIR = lib
BUILD_DIR = build
OUTPUT_DIR = output

# Source files
C_SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJC_SOURCES = $(wildcard $(SRC_DIR)/*.m)
LIB_SOURCES = $(wildcard $(LIB_DIR)/*.c)

# Object files
C_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SOURCES))
OBJC_OBJECTS = $(patsubst $(SRC_DIR)/%.m,$(BUILD_DIR)/%.o,$(OBJC_SOURCES))
LIB_OBJECTS = $(patsubst $(LIB_DIR)/%.c,$(BUILD_DIR)/lib_%.o,$(LIB_SOURCES))

# Output
TARGET = $(BUILD_DIR)/spectraforge

# Default target
.PHONY: all clean debug run test dirs

all: dirs $(TARGET)
	@echo ""
	@echo "Build complete! Run with: ./$(TARGET)"
	@echo "  Quick test: ./$(TARGET) --width 400 --height 300 --samples 4"
	@echo "  Full render: ./$(TARGET) --samples 64"

dirs:
	@mkdir -p $(BUILD_DIR) $(OUTPUT_DIR)

# Link final executable
$(TARGET): $(C_OBJECTS) $(OBJC_OBJECTS) $(LIB_OBJECTS)
	$(OBJC) $(OBJCFLAGS) $(LDFLAGS) -o $@ $^

# Compile C sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | dirs
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(LIB_DIR) -c -o $@ $<

# Compile Objective-C sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.m | dirs
	$(OBJC) $(OBJCFLAGS) -I$(INCLUDE_DIR) -I$(LIB_DIR) -c -o $@ $<

# Compile library sources (cJSON, etc.)
$(BUILD_DIR)/lib_%.o: $(LIB_DIR)/%.c | dirs
	$(CC) $(CFLAGS) -w -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: OBJCFLAGS = $(DEBUG_OBJCFLAGS)
debug: all

# Run the renderer
run: all
	cd $(BUILD_DIR)/.. && ./$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(OUTPUT_DIR)/*.png

# Run quick render test
test: all
	cd $(BUILD_DIR)/.. && ./$(TARGET) --width 200 --height 150 --samples 2

# Run parameter verification tests (requires Python)
test-params: all
	cd $(BUILD_DIR)/.. && python tests/test_parameters.py --quick

# Run full parameter test suite
test-params-full: all
	cd $(BUILD_DIR)/.. && python tests/test_parameters.py --verbose

# Performance benchmark
benchmark: all
	cd $(BUILD_DIR)/.. && ./$(TARGET) --benchmark --width 1920 --height 1080 --samples 64

# Quick preview render
preview: all
	cd $(BUILD_DIR)/.. && ./$(TARGET) --width 800 --height 600 --samples 4 --output $(OUTPUT_DIR)/preview.png

# High quality render
render: all
	cd $(BUILD_DIR)/.. && ./$(TARGET) --width 1920 --height 1080 --samples 256 --output $(OUTPUT_DIR)/render.png

.PHONY: help
help:
	@echo "SpectraForge Metal - GPU Ray Tracer"
	@echo ""
	@echo "Note: Shaders are compiled at runtime from source files in shaders/"
	@echo "      No Metal toolchain installation required!"
	@echo ""
	@echo "Targets:"
	@echo "  all             - Build executable (default)"
	@echo "  debug           - Build with debug symbols"
	@echo "  run             - Build and run with default settings"
	@echo "  preview         - Quick preview render (800x600, 4 spp)"
	@echo "  render          - High quality render (1920x1080, 256 spp)"
	@echo "  benchmark       - Run performance benchmark"
	@echo "  test            - Quick test render"
	@echo "  test-params     - Verify UI parameters affect output (quick)"
	@echo "  test-params-full- Full parameter verification with details"
	@echo "  clean           - Remove build artifacts"
	@echo "  help            - Show this help"
