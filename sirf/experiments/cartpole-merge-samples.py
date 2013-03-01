#!/usr/bin/env python

import numpy as np
import os
import re
import sys

from matplotlib import pyplot as plt

LOSSES = 'policy true-bellman test-bellman test-model test-reward'.split()


def key(filename):
    m = re.search(r'n(\d+)-(m\d)-(g[^-]+)-(k[\d,]+)', filename)
    samples, method, nonlin, features = m.groups()
    return '%05d %s %s %s' % (int(samples), method, nonlin, features)


def extract(filename):
    losses = dict((l, None) for l in LOSSES)
    with open(filename) as handle:
        for line in handle:
            for l in losses:
                if 'loss ' + l in line:
                    losses[l] = float(line.strip().split()[-1])
    return losses


def main(filenames):
    db = {}
    for filename in filenames:
        k = key(filename)
        losses = extract(filename)
        if k not in db:
            db[k] = dict((k, [v]) for k, v in losses.iteritems())
        else:
            for l in LOSSES:
                db[k][l].append(losses[l])
    for k, losses in sorted(db.iteritems()):
        for l, vs in losses.iteritems():
            tv = [v for v in vs if v is not None]
            print k, l, np.mean(tv), np.std(tv) / np.sqrt(len(tv))


if __name__ == '__main__':
    main(sys.argv[1:])

