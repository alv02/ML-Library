#!/bin/bash
set -e

mkdir -p build

# Compile each cpp file to object files
g++ -I include -c src/tensor.cpp -o build/tensor.o
g++ -I include -c src/tensor_iterator.cpp -o build/tensor_iterator.o
g++ -I include -c src/autograd.cpp -o build/autograd.o
g++ -I include -c src/ops.cpp -o build/ops.o   
g++ -I include -c src/models.cpp -o build/models.o
g++ -I include -c src/optimizers.cpp -o build/optimizers.o

# Link everything together
g++ -I include main.cpp build/tensor.o build/tensor_iterator.o build/autograd.o build/ops.o build/models.o build/optimizers.o -o build/main

# Run
./build/main
