#!/bin/bash

nvcc --compiler-bindir=/opt/gcc-4.4.6/bin --gpu-architecture sm_20 -I./include/ -c src/pga.cu -o pga.o
mpicc -I./include/ -c test/test.cu -o test.o
mpicc test.o pga.o -o mpicuda -L/usr/local/cuda/lib64 -lm -lcudart -lcurand -lstdc++
