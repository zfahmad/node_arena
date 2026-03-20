#!/bin/bash

cmake -S . -B build
cmake --build build
pip install -r requirements.txt
