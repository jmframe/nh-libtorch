#!/bin/bash

mkdir -p build && \
cd build && \
cmake -DCMAKE_PREFIX_PATH=/glade/scratch/jframe/nh-libtorch .. && \
make && \
cp lstm_run ../ && \
cd ../ && \
./lstm_run ./data/nosnow_normalarea_672/lstm.ptc 
