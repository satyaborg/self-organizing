#!/bin/bash
kernprof -l train.py
python -m line_profiler train.py.lprof
