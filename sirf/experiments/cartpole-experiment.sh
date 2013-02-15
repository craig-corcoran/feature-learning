#!/bin/bash

PYTHONPATH=$(pwd)/../..:$PYTHONPATH python cartpole_experiment.py $*
