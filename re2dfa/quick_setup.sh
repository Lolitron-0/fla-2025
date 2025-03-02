#!/bin/bash
cmake -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g -O0" -DCMAKE_CXX_COMPILER=clang++ -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cp build/compile_commands.json . 
cmake --build build 
