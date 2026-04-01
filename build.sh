gcc -c util/arena.c -o arena.o
g++ -c math/tensor.cpp -o tensor.o
g++ main.cpp arena.o tensor.o -o ./build/main
