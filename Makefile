# Builds build/face_gesture on the robot (JetPack 5.1.1: CUDA 11.4, TensorRT 8.5, OpenCV 4.2, g++ 9.4).
#   make -j4          (the robot's clock may be unset, so make can warn about clock skew: harmless)
# Nothing outside these preinstalled libraries is used.

CUDA      ?= /usr/local/cuda
NVCC      ?= $(CUDA)/bin/nvcc
CXX       ?= g++
# CUDA_ARCH: sm_87 = Jetson Orin
CUDA_ARCH ?= sm_87

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

CXXFLAGS  := -std=c++17 -O3 -Wall -Wextra -Wno-deprecated-declarations -Wno-format-truncation -I$(CUDA)/include $(OPENCV_CFLAGS)
NVCCFLAGS := -std=c++17 -O3 -arch=$(CUDA_ARCH) -Xcompiler -Wall
LDLIBS    := -L$(CUDA)/lib64 -lnvinfer -lcudart $(OPENCV_LIBS) -lpthread

SRCS := main.cpp trt_engine.cpp camera.cpp ocsort.cpp video_sink.cpp live_view.cpp gestures.cpp
OBJS := $(addprefix build/,$(SRCS:.cpp=.o)) build/preprocess.o
HDRS := $(wildcard src/*.hpp)

all: build/face_gesture

build/face_gesture: $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDLIBS)
	@echo "built $@"

build/%.o: src/%.cpp $(HDRS) | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/preprocess.o: src/preprocess.cu src/preprocess.hpp src/preprocess_math.hpp src/cuda_utils.hpp | build
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build:
	mkdir -p build

clean:
	rm -f build/*.o build/face_gesture

.PHONY: all clean
