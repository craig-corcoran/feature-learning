#!/bin/bash

t=$(mktemp -d)

. /u/leif/.py/utcs/bin/activate

export THEANO_FLAGS="config.compiledir=$t"
export PYTHONPATH=$(pwd)/../..:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib32:$LD_LIBRARY_PATH

ulimit -d 1800000  # 1.8G data size limit
ulimit -t 3600  # one hour runtime limit

for s in 16 32 64 128 256 512 1024
do python cartpole_experiment.py -n-samples $s $*
done

rm -fr $t
