#!/bin/bash

mkdir -p build && \
cd build && \
cmake -DCMAKE_PREFIX_PATH=/home/noaa_ml_water_modelers/libtorch .. && \
make && \
cp lstm_run ../ && \
cd ../ && \
./lstm_run ./lstm.ptc 
