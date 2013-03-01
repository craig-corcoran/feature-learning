#!/bin/bash

t=$(mktemp -d)

THEANO_FLAGS="config.compiledir=$t" PYTHONPATH=$(pwd)/../..:$PYTHONPATH python cartpole_experiment.py $*

rm -fr $t
