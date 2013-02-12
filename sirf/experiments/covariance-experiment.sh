#!/bin/bash

PYTHONPATH=$(pwd)/../..:$PYTHONPATH python covariance_experiment.py $*
