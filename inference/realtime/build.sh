#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
clang++ -std=c++17 -O3 -fobjc-arc -mmacosx-version-min=15.0 -Wall -Wextra main.mm \
    -framework Foundation -framework CoreML -framework CoreAudio -framework AudioToolbox \
    -o build/realtime_coreml
clang++ -std=c++17 -O2 -Wall -Wextra test_ring.cpp -o build/test_ring
./build/test_ring
