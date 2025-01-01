#!/bin/bash

cd build/MonoVulkan
make
cd ../../bin/debug
./MONO
