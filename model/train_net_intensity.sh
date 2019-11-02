#!/usr/bin/env sh

set -e

/code/caffe/build/tools/caffe train -solver FERA2015_solver_intensity.prototxt -gpu 0 $@
