#!/usr/bin/env sh

set -e

/code/caffe/build/tools/caffe train -solver BP4D_solver_intensity.prototxt -gpu 0 $@
