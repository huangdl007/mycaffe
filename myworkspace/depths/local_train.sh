#!/usr/bin/env sh

TOOLS=./../../build/tools

$TOOLS/caffe train	\
    --solver=local_solver.prototxt	\
    --weights=NYU_DEPTH_iter_300000.caffemodel
