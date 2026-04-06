#!/bin/bash
set -e

mkdir -p build

gcc -c util/arena.c -o build/arena.o
g++  -I include -c src/tensor.cpp -o build/tensor.o
g++  -I include main.cpp build/arena.o build/tensor.o -o build/main

./build/main
