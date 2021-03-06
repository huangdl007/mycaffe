#!/usr/bin/env sh

TOOLS=./../../build/tools

$TOOLS/caffe train \
    --solver=branch_solver.prototxt \
    --weights=models/backbone_11_18_iter_50000.caffemodel

# reduce learning rate by factor of 10
#$TOOLS/caffe train \
#    --solver=examples/cifar10/cifar10_full_solver_lr1.prototxt \
#    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate

# reduce learning rate by factor of 10
#$TOOLS/caffe train \
#    --solver=examples/cifar10/cifar10_full_solver_lr2.prototxt \
#    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate
