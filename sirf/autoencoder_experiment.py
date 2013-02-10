#!/usr/bin/env python

import logging
import numpy as np
import os
import sys
import lmj.tnn
import theano.tensor as TT

from matplotlib import pyplot as plt

import grid_world

g = lmj.tnn.FLAGS.add_argument_group('Experiment Options')
g.add_argument('--samples', type=int, default=128, metavar='N',
               help='sample this many state transitions from world')
g.add_argument('--state-rep', default='tabular', choices=['tabular'],
               help='represent states using this strategy')
g.add_argument('--distribution', default='uniform', choices=['uniform', 'policy'],
               help='sample states using this strategy')


class CrossEntropyRegressor(lmj.tnn.Regressor):
    @property
    def cost(self):
        return -TT.mean(TT.log(self.y)[self.k])


class Main(lmj.tnn.Main):
    def get_network(self):
        return lmj.tnn.Classifier

    def get_datasets(self):
        n = self.args.samples
        states, next_states, _, _ = grid_world.MDP(
            walls_on=True, size=int(np.sqrt(self.args.layers[0]))).sample_grid_world(
            n, self.args.state_rep, self.args.distribution)
        logging.info('got %s states %s, next %s', states.dtype, states.shape, next_states.shape)
        c = 7 * n // 10
        def idx(z):
            return np.array([k.argmax() for k in z]).astype(np.int32)
        return (states[:c], idx(next_states[:c])), (states[c:], idx(next_states[c:]))


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        format='%(levelname).1s %(asctime)s %(message)s',
        level=logging.INFO)

    m = Main()

    layers = ','.join(str(n) for n in m.args.layers)
    #path = 'autoencoder-%s-%s.pickle.gz' % (layers, m.args.activation, m.args.samples)

    #if os.path.exists(path):
    #    m.net.load(path)

    m.train()
    #m.net.save(path)

    W = m.net.weights[0].get_value(borrow=True)
    v = abs(W).max()
    logging.info('feature scale: +/- %.3f', v)
    n = int(np.sqrt(m.args.layers[0]))
    c = int(np.ceil(np.sqrt(m.args.layers[1])))
    imgs = []
    fig = plt.figure()
    for i in range(m.args.layers[1]):
        ax = fig.add_subplot(c, c, i + 1, aspect='equal')
        imgs.append(ax.imshow(W[:, i].reshape((n, n)), vmin=-v, vmax=v, cmap='hot', interpolation='nearest'))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.8)
    fig.colorbar(imgs[0], cax=fig.add_axes([0.85, 0.15, 0.05, 0.7]))
    fig.savefig('autoencoder-%s-samples%d.pdf' % (m.args.activation, m.args.samples))
