#!/usr/bin/env bash
rm fuego.cpp
rm fuego.so
rm hash_maps.c
rm hash_maps.so
python setup.py build_ext --inplace

