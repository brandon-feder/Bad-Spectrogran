#!/bin/bash
nvcc main.cu -o "bin/main.out"  -lSDL2;
./bin/main.out;
rm ./bin/main.out;
