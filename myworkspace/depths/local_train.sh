#!/usr/bin/env sh

TOOLS=./../../build/tools

$TOOLS/caffe train	\
    --solver=local_solver.prototxt	\
    --weights=models/NYU_DEPTH_GLOBAL_9_28_iter_200000.caffemodel
