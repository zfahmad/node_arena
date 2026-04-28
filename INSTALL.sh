#!/bin/bash

python3 -m venv pyenv
. pyenv/bin/activate
pip install -r requirements.txt
cmake -S . -B build
cmake --build build
