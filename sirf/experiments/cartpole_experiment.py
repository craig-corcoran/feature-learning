#!/usr/bin/env python

import numpy
import pickle
import re
import scipy.optimize
import scipy.sparse
import sirf
import matplotlib.pyplot as plt

logger = sirf.get_logger(__name__)


def sample(n):
    '''Generate n state/reward samples from the cart-pole world.'''
    states = []
    rewards = []
    w = sirf.CartPole()
    episodes = 0
    while len(states) < n:
        # episodes are lists of [pstate, paction, reward, state, next_action]
        for state, _, reward, next_state, _ in w.single_episode():
            episodes += 1
            states.append(state + [1])
            rewards.append([reward])
            if len(states) == n:
                states.append(next_state + [1])
                break
    s = numpy.asarray(states)
    r = numpy.asarray(rewards)
    logger.info('sampled %s states and %s rewards from %d episodes',
                s.shape, r.shape, episodes)
    return scipy.sparse.csr_matrix(s), scipy.sparse.csr_matrix(r)


@sirf.annotations(
    ks=('number of features', 'option', None),
    lam=('lambda for TD-lambda training', 'option', None, float),
    gam=('discount factor', 'option', None, float),
    beta=('covariance loss paramter', 'option', None, float),
    eps=('epsilon for computing TD-lambda horizon', 'option', None, float),
    patience=('train until patience runs out', 'option', None, int),
    max_iter=('train for at most this many iterations', 'option', None, int),
    min_imp=('train until loss improvement is less than this', 'option', None, int),
    l1theta=('regularize theta with this L1 parameter', 'option', None, float),
    l1code=('regularize feature code with this L1 parameter', 'option', None, float),
    n_samples=('sample this many state transitions', 'option', None, int),
    nonlin=('feature nonlinearity', 'option', None, str, ['sigmoid', 'relu', 'linear']),
    nonzero=('penalty for zero theta vectors', 'option', None, float),
    method=('training method', 'option', None, int),
    output=('save results in this file', 'option', None, str),
    trace_every=('trace every N iterations', 'option', None, int),
    )
def main(ks = 16,
         lam = 0.,
         gam = 0.999,
         beta = 0.99,
         eps = 1e-5,
         patience = 15,
         max_iter = 15,
         min_imp = 0.,
         l1theta = None,
         l1code = None,
         n_samples = None,
         nonlin = None,
         nonzero = None,
         method = 0,
         output = None,
         trace_every = 3,
         ):
    n = 4

    method = (
        ('layered',                  ('all', )), # baseline
        ('prediction',               ('theta-all', )),
        ('covariance',               ('theta-all', )), # with reward, without fine-tuning
        ('prediction layered',       ('theta-all', 'all')),
        ('covariance layered',       ('theta-all', 'all')), # theta-model here for 2nd wrt?
        ('value_prediction',         ('theta-all', )),
        ('value_prediction layered', ('theta-all', 'all')),
        )[method]

    logger.info('constructing basis')
    reg = None
    if l1theta is not None:
        reg = ('l1-theta', l1theta)
    if l1code is not None:
        reg = ('l1-code', l1code)

    bb = sirf.BellmanBasis(n + 1, [int(k) for k in re.findall(r'\d+', str(ks))],
                           beta = beta / gam,
                           reg_tuple = reg,
                           nonlin = nonlin,
                           nonzero = nonzero)

    kw = dict(lam=lam, gam=gam, sampled=True, eps=eps)

    # sample data: training, validation, test, and bellman ("true") sets
    n_steps = sirf.BellmanBasis._calc_n_steps(lam = lam, gam = gam, eps = eps) - 1
    Mphi, Mrew = sirf.BellmanBasis.get_mixing_matrices(n_samples, **kw)
    S, R = sample(n_samples + n_steps)
    S_valid, R_valid = sample(n_samples + n_steps)
    S_test, R_test = sample(n_samples + n_steps)

    logger.info('training with %i samples and %s method', S.shape[0], method)
    loss_list, wrt_list = method
    assert len(loss_list.split()) == len(wrt_list)

    recordable = (
        ('test-bellman', bb.loss_be, S_test, R_test),
        ('test-reward', bb.loss_r, S_test, R_test),
        ('test-model', bb.loss_m, S_test, R_test),
        )

    losses = dict((r[0], []) for r in recordable)
    losses['true-bellman'] = []
    losses['policy'] = []
    def trace():
        Mphi_be, Mrew_be = sirf.BellmanBasis.get_mixing_matrices(1024, **kw)
        S_be, R_be = sample(1024 + n_steps)
        loss = bb.loss_be(*(bb.params + [S_be, R_be, Mphi_be, Mrew_be])) / 1024
        logger.info('loss true-bellman: %s', loss)
        losses['true-bellman'].append(loss)

        cp = sirf.CartPole()
        p = sirf.ValuePolicy(get_value = bb.estimated_value)
        n = 4096
        e = 0
        while n > 0:
            trace = cp.single_episode(p)
            e += 1
            n -= len(trace)
        logger.info('loss policy: %s', e)
        losses['policy'].append(e)

        for key, func, s, r in recordable:
            loss = func(*(bb.params + [s, r, Mphi, Mrew])) / n_samples
            logger.info('loss %s: %s', key, loss)
            losses[key].append(loss)

    bb.set_loss('layered', ['w'])
    bb.set_params(scipy.optimize.fmin_cg(
            bb.loss, bb.flat_params, bb.grad,
            args = (S, R, Mphi, Mrew),
            full_output = False,
            maxiter = max_iter,
            ))

    it = 0
    for loss, wrt in zip(loss_list.split(), wrt_list):
        best_test_loss = 1e10
        best_params = None
        waiting = 0

        try:
            while waiting < patience:
                if not it % trace_every:
                    trace()
                it += 1
                logger.info('** iteration %d', it)

                for loss_, wrt_ in ((loss, wrt.split()), ('layered', ['w'])):
                    bb.set_loss(loss_, wrt_)
                    bb.set_params(scipy.optimize.fmin_cg(
                            bb.loss, bb.flat_params, bb.grad,
                            args = (S, R, Mphi, Mrew),
                            full_output = False,
                            maxiter = max_iter,
                            ))

                err = bb.loss(bb.flat_params, S_valid, R_valid, Mphi, Mrew)
                if (best_test_loss - err) / best_test_loss > min_imp:
                    waiting = 0
                    best_test_loss = err
                    best_params = [p.copy() for p in bb.params]
                    logger.info('new best %s loss: %s', bb.loss_type, best_test_loss)
                    for d, t in enumerate(bb.thetas):
                        logger.info('theta-%d norms: %s', d, ' '.join('%.2f' % x for x in (t * t).sum(axis=0)))
                else:
                    waiting += 1
                    logger.info('iters without better %s loss: %s', bb.loss_type, waiting)

            bb.params = best_params
        except KeyboardInterrupt:
            print '\n user stopped current training loop'

    trace()

    if output:
        root = '%s-cartpole.k%s.g%.3f.l%.3f.n%d.%s.%s' % (
            output, ks, gam, lam, n_samples, '+'.join(loss_list.split()), nonlin or 'linear',
            )
        logger.info('saving results to %s.pickle.gz' % root)
        with sirf.openz(root + '.pickle.gz', 'wb') as handle:
            pickle.dump((bb.thetas, losses), handle, protocol=-1)

        #plot_features(root, bb)


def plot_features(root, bb):
    ranges = {'x': 3, r'\dot{x}': 15, r'\theta': 0.3, r'\dot{\theta}': 15}
    grid = 11
    space = lambda k: enumerate(numpy.linspace(-ranges[k], ranges[k], grid))

    logger.info('computing feature responses')
    probe = numpy.zeros((bb.thetas[-1].shape[1], grid, grid, grid, grid), float)
    for i, x in space(r'x'):
        for j, dx in space(r'\dot{x}'):
            for k, t in space(r'\theta'):
                for l, dt in space(r'\dot{\theta}'):
                    probe[:, i, j, k, l] = bb.encode([x, dx, t, dt])

    s = int(numpy.ceil(numpy.sqrt(len(probe))))
    for abs, ord, sum1, sum2 in (
        (r'x', r'\dot{x}', -1, -1),
        (r'\theta', r'\dot{\theta}', 0, 0),
        (r'x', r'\theta', 1, -1),
        (r'\dot{x}', r'\dot{\theta}', 0, -2),
        ):
        plt.clf()
        for i, pr in enumerate(probe):
            ax = plt.subplot(s, s, i + 1)
            ax.imshow(pr.sum(axis=sum1).sum(axis=sum2))
            if i >= (s - 1) * s:
                ax.set_xlabel('$%s$' % abs)
                ax.set_xticks([0, grid])
                ax.set_xticklabels([-ranges[abs], ranges[abs]])
            else:
                ax.set_xticks([])
            if i % s == 0:
                ax.set_ylabel('$%s$' % ord)
                ax.set_yticks([0, grid])
                ax.set_yticklabels([-ranges[ord], ranges[ord]])
            else:
                ax.set_yticks([])
        clean = re.sub(r'\W+', '-', ('%s %s' % (abs, ord)).strip('}').strip('\\'))
        plt.savefig('%s-%s.pdf' % (root, clean), dpi=1200)


if __name__ == '__main__':
    sirf.script(main)
