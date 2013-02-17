#!/usr/bin/env python

import numpy
import pickle
import re
import scipy.optimize
import scipy.sparse
import sirf
import theano
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
            states.append(state)
            rewards.append([reward])
            if len(states) == n:
                states.append(next_state)
                break
    s = numpy.asarray(states)
    r = numpy.asarray(rewards)
    logger.info('sampled %s states and %s rewards from %d episodes',
                s.shape, r.shape, episodes)
    return scipy.sparse.csc_matrix(s), scipy.sparse.csc_matrix(r)


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
    loss_types=('train on these losses in this order', 'option', None, str),
    grad_vars=('compute gradient using these variables', 'option', None, str),
    nonlin=('feature nonlinearity', 'option', None, str, ['sigmoid', 'relu']),
    nonzero=('penalty for zero theta vectors', 'option', None, float),
    method=('training method', 'option', None, int),
    output=('save results in this file', 'option', None, str),
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
         loss_types = 'covariance bellman',
         grad_vars = 'theta-all',
         nonlin = None,
         nonzero = None,
         method = 0,
         output = None,
         ):
    n = 4

    method = (
        ('layered',            ('all', )), # baseline
        ('prediction',         ('theta-all', )),
        ('covariance',         ('theta-all', )), # with reward, without fine-tuning
        ('prediction layered', ('theta-all', 'all')),
        ('covariance layered', ('theta-all', 'all')), # theta-model here for 2nd wrt?
        )[method]

    logger.info('constructing basis')
    reg = None
    if l1theta is not None:
        reg = ('l1-theta', l1theta)
    if l1code is not None:
        reg = ('l1-code', l1code)

    bb = sirf.BellmanBasis(n, [int(k) for k in re.findall(r'\d+', str(ks))],
                           beta = beta / gam,
                           reg_tuple = reg,
                           nonlin = nonlin,
                           nonzero = nonzero)

    kw = dict(lam=lam, gam=gam, sampled=True, eps=eps)

    # sample data: training, validation, test, and bellman ("true") sets
    Mphi, Mrew = sirf.BellmanBasis.get_mixing_matrices(n_samples, **kw)
    S, R = sample(n_samples)

    Mphi_test, Mrew_test = sirf.BellmanBasis.get_mixing_matrices(1024, **kw)
    S_valid, R_valid = sample(1024)
    S_test, R_test = sample(1024)

    Mphi_be, Mrew_be = sirf.BellmanBasis.get_mixing_matrices(16384, **kw)
    S_be, R_be = sample(16384)

    logger.info('training with %i samples and %s method', S.shape[0], method)
    loss_list, wrt_list = method
    assert len(loss_list.split()) == len(wrt_list)

    recordable = (
        ('test-bellman', bb.loss_be, S_test, R_test, Mphi_test, Mrew_test),
        ('test-reward', bb.loss_r, S_test, R_test, Mphi_test, Mrew_test),
        ('test-model', bb.loss_m, S_test, R_test, Mphi_test, Mrew_test),
        ('true-bellman', bb.loss_be, S_be, R_be, Mphi_be, Mrew_be),
        )

    losses = {r[0]: [] for r in recordable}
    def trace():
        for key, func, s, r, phi, rew in recordable:
            loss = func(*(bb.params + [s, r, phi, rew]))
            logger.info('loss %s: %s', key, loss)
            losses[key].append(loss)

    for loss, wrt in zip(loss_list.split(), wrt_list):
        bb.set_loss(loss, wrt.split())
        best_test_loss = 1e10
        best_params = None
        waiting = 0

        trace()

        try:
            it = 0
            while waiting < patience:
                it += 1
                logger.info('** iteration %d', it)

                bb.set_params(scipy.optimize.fmin_cg(
                        bb.loss, bb.flat_params, bb.grad,
                        args = (S, R, Mphi, Mrew),
                        full_output = False,
                        maxiter = max_iter,
                        ))

                err = bb.loss(bb.flat_params, S_valid, R_valid, Mphi_test, Mrew_test)
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

                if not it % 3:
                    trace()

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

        plot_features(root, bb)


def plot_features(root, bb):
    logger.info('computing feature responses')
    probe = numpy.zeros((bb.thetas[-1].shape[1], 11, 11, 11, 11), float)
    for i, x in enumerate(numpy.linspace(-3, 3, 11)):
        for j, dx in enumerate(numpy.linspace(-15, 15, 11)):
            for k, t in enumerate(numpy.linspace(-0.3, 0.3, 11)):
                for l, dt in enumerate(numpy.linspace(-15, 15, 11)):
                    probe[:, i, j, k, l] = bb.encode([x, dx, t, dt])

    s = int(numpy.ceil(numpy.sqrt(len(probe))))
    for title, sum1, sum2 in (
        (r'$x, \dot{x}$', -1, -1),
        (r'$\theta, \dot{\theta}$', 0, 0),
        (r'$x, \theta$', 1, -1),
        (r'$\dot{x}, \dot{\theta}$', 0, -2),
        ):
        plt.clf()
        plt.title(title)
        for i, pr in enumerate(probe):
            ax = plt.subplot(s, s, i + 1)
            ax.imshow(pr.sum(axis=sum1).sum(axis=sum2))
            ax.set_xticks([])
            ax.set_yticks([])
        clean = re.sub(r'\W+', '-', title[1:-1].strip('}').strip('\\'))
        plt.savefig('%s-%s.pdf' % (root, clean), dpi=1200)


if __name__ == '__main__':
    sirf.script(main)
