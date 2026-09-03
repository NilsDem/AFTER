PYTHON ?= python
TORCH_ROOT := $(shell $(PYTHON) -c "import pathlib, torch; print(pathlib.Path(torch.__file__).parent)")
TORCH_INCLUDE := $(TORCH_ROOT)/include
TORCH_LIB := $(TORCH_ROOT)/lib

.PHONY: all clean

all: benchmark_aoti benchmark_coreml

benchmark_aoti: benchmark_aoti.cpp
	$(CXX) -std=c++20 -O3 $< \
		-I$(TORCH_INCLUDE) \
		-I$(TORCH_INCLUDE)/torch/csrc/api/include \
		-L$(TORCH_LIB) \
		-Wl,-rpath,$(TORCH_LIB) \
		-ltorch -ltorch_cpu -lc10 \
		-o $@

benchmark_coreml: benchmark_coreml.mm
	$(CXX) -std=c++20 -O3 -fobjc-arc $< \
		-framework Foundation \
		-framework CoreML \
		-o $@

clean:
	rm -f benchmark_aoti benchmark_coreml
