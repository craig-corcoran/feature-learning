#!/usr/bin/env python

import collections
import numpy as np
import os
import re
import sys

from matplotlib import pyplot as plt

LOSSES = 'policy true-bellman test-bellman test-model test-reward'.split()

def loss(line):
    m = re.search(r'n-(\d+) .* loss (\S+) ([.\d]+)$', line)
    try:
        samples, loss, value = m.groups()
        return '%05d' % int(samples), loss, float(value)
    except:
        return '', '', 0.


def extract(filename):
    with open(filename) as handle:
        for line in handle:
            yield loss(line)


def key(filename):
    return re.search(r'm(\d)-g([^-]+)-k([,\d]+)', filename).groups()


def main(filenames):
    db = collections.defaultdict(list)
    for filename in filenames:
        localdb = {}
        method, nonlin, features = key(filename)
        for samples, loss, value in extract(filename):
            if not loss: continue
            k = method, nonlin, features, samples, loss
            localdb[k] = value
        for k, v in localdb.iteritems():
            db[k].append(v)
    for k, vs in sorted(db.iteritems()):
        print '%s %s %s %s %s' % k, len(vs), np.mean(vs), np.std(vs) / np.sqrt(len(vs))


if __name__ == '__main__':
    main(sys.argv[1:])

