#!/bin/bash

mkdir -p build && \
cd build && \
cmake -DCMAKE_PREFIX_PATH=/glade/scratch/jframe/libtorch .. && \
make && \
echo "\nRunning the program...\n" &&\
./lstm_run ../lstm.ptc && \
cd ..
