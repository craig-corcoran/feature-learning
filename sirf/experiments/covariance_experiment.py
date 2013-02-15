import copy
import itertools
import numpy
import pickle
import plac
import scipy.optimize
import scipy.sparse
import matplotlib.pyplot as plt

import sirf.grid_world as grid_world
import sirf.util as util
from sirf.rl import Model
from sirf.bellman_basis import plot_features, BellmanBasis as Basis

# add tracking of reward and model loss along with bellman error
# mark shifts and best theta

@plac.annotations(
    k=('number of features', 'option', None, int),
    env_size=('size of the grid world', 'option', 's', int),
    lam=('lambda for TD-lambda training', 'option', None, float),
    gam=('discount factor', 'option', None, float),
    beta=('covariance loss paramter', 'option', None, float),
    eps=('epsilon for computing TD-lambda horizon', 'option', None, float),
    patience=('train until patience runs out', 'option', None, int),
    max_iter=('train for at most this many iterations', 'option', 'i', int),
    weighting=('method for sampling from grid world', 'option', None, str, ['policy', 'uniform']),
    l1theta=('regularize theta with this L1 parameter', 'option', None, float),
    l1code=('regularize feature code with this L1 parameter', 'option', None, float),
    state_rep=('represent world states this way', 'option', None, str, ['tabular', 'factored']),
    n_samples=('sample this many state transitions', 'option', None, int),
    loss_types=('train on these losses in this order', 'option', None, str),
    grad_vars=('compute gradient using these variables', 'option', None, str),
    nonlin=('feature nonlinearity', 'option', None, str, ['sigmoid', 'relu']),
    nonzero=('penalty for zero theta vectors', 'option', None, float),
    )
def main(k = 16,
         env_size = 15,
         lam = 0.,
         gam = 0.999,
         beta = 0.99,
         eps = 1e-5, 
         partition = None,
         patience = 15,
         max_iter = 15,
         weighting = 'uniform', 
         l1theta = None,
         l1code = None,
         state_rep = 'tabular',
         n_samples = None,
         loss_types = 'covariance bellman',
         grad_vars = 'theta-all',
         nonlin = None,
         nonzero = None,
         ):

    beta_ratio = beta/gam 

    if partition is None:
        partition = {'theta-model':k-1, 'theta-reward':1}

    print 'building environment'
    mdp = grid_world.MDP(walls_on = True, size = env_size)
    n = env_size**2
    m = Model(mdp.env.R, mdp.env.P, gam = gam)
    m.gam = gam

    print 'constructing basis'
    reg = None
    if l1theta is not None:
        reg = ('l1-theta', l1theta)
    if l1code is not None:
        reg = ('l1-code', l1code)
    bb = Basis(n + 1, k, beta_ratio, partition = partition, reg_tuple = reg, nonlin = nonlin, nonzero = nonzero)
    bb.theta[:, -1] = numpy.hstack([m.R, [0]]) # initialize the last column as the reward function

    if n_samples:
        kw = dict(n_samples = n_samples, state_rep = state_rep, distribution = weighting)
        S, Sp, R, _ = mdp.sample_grid_world(**kw)
        X = scipy.sparse.vstack((S, Sp[-1, :]))
        S_test, Sp_test, R_test, _ = mdp.sample_grid_world(**kw)
        X_test = scipy.sparse.vstack((S_test, Sp_test[-1, :]))
    else:
        R = numpy.array([])
        X = scipy.sparse.eye(n, n)
        P = scipy.sparse.eye(n, n)
        for i in xrange(bb._calc_n_steps(lam, gam, eps)): # decay epsilon 
            R = numpy.append(R, P * m.R)
            P = m.P * P
            X = scipy.sparse.vstack((X, P))
        R = R[:,None]
        X_test = X
        R_test = R

    # build bellman operator matrices
    print 'making mixing matrices'
    Mphi, Mrew = bb.get_mixing_matrices(n_samples or n, lam, gam, sampled = bool(n_samples), eps = eps)

    test_loss = numpy.array([])
    test_be = numpy.array([])
    true_be = numpy.array([])
    true_lsq = numpy.array([])

    for lt, wrt in zip(loss_types.split(), itertools.cycle(grad_vars.split())):
        try:
            print 'training with %s loss with respect to %s vars' % (lt,wrt)
            bb.set_loss(loss_type=lt, wrt=[wrt])
            bb.set_regularizer(reg)
            
            theta_old = copy.deepcopy(bb.theta)
            bb, tel, teb, trb, trq = train_basis(bb, m, X, R, X_test, R_test, 
                        Mphi, Mrew, patience, max_iter, weighting)

            delta = numpy.linalg.norm(bb.theta - theta_old)
            print 'delta: ', delta
            
            test_loss = numpy.append(test_loss, tel)
            test_be = numpy.append(test_be, teb)
            true_be = numpy.append(true_be, trb)
            true_lsq = numpy.append(true_lsq, trq)

        except KeyboardInterrupt:
            print '\n user stopped current training loop'

    def output(prefix, suffix='pdf'):
        return '%s.k=%i.reg=%s.lam=%s.gam=%s.b=%s.%s.%s%s%s.%s' % (
            prefix, k,
            str(reg),
            lam, gam, beta, weighting, '+'.join(loss_types.split()),
            '.samples=%d' % n_samples if n_samples else '',
            '.nonlin=%s' % nonlin if nonlin else '',
            suffix)

    # save results!
    with util.openz(output('covariance_results', 'pickle.gz'), "wb") as out_file:
        pickle.dump((test_loss, test_be, true_be, true_lsq), out_file, protocol = -1)

    # plot basis functions
    plot_features(bb.theta[:-1])
    plt.savefig(output('basis'))
    
    # plot learning curves
    plot_learning_curves(numpy.array([test_loss, test_be, true_be, true_lsq]), 
                         ['test loss', 'test BE', 'true BE', 'true lsq'],
                         ['r-','g-','b-','k-'])
    plt.savefig(output('loss'))
    
    # plot value functions
    plot_value_functions(env_size, m, bb)
    plt.savefig(output('value'))


def plot_value_functions(size, m, b):
    # plot value functions, true and approx
    plt.clf()
    f = plt.figure()
    
    # todo something wrong with model vf?
    # true model value fn
    ax = f.add_subplot(311)
    ax.imshow(numpy.reshape(m.V, (size, size)), cmap = 'gray', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    # bellman error estimate (using true model)
    ax = f.add_subplot(312)
    w_be = m.get_lstd_weights(b.theta[:-1]) # TODO add lambda parameter here
    v_be = numpy.dot(b.theta[:-1], w_be)
    ax.imshow(numpy.reshape(v_be, (size, size)), cmap = 'gray', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # least squares solution with true value function
    ax = f.add_subplot(313)
    w_lsq = numpy.linalg.lstsq(b.theta[:-1], m.V)[0]
    v_lsq = numpy.dot(b.theta[:-1], w_lsq)
    ax.imshow(numpy.reshape(v_lsq, (size, size)), cmap = 'gray', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    print 'bellman error norm from v: ', numpy.linalg.norm(m.V - v_be)
    print 'lsq error norm from v: ', numpy.linalg.norm(m.V - v_lsq)

def plot_learning_curves(losses, labels, draw_styles = ['r-','g-','b-','k-']):
    plt.clf()
    ax = plt.axes()
    print losses, labels, draw_styles
    for i, l in enumerate(losses):
        x = range(len(l))
        ax.plot(x, l / l.mean(), draw_styles[i], label=labels[i])

    plt.ylim(0, 3)
    plt.title('Normalized Losses per CG Minibatch')
    plt.legend(loc='upper right')


def train_basis(basis, model, S, R, S_test, R_test, Mphi, Mrew, patience, 
                    max_iter, weighting):
    
    test_loss = numpy.append(numpy.array([]), basis.loss(basis.flat_params, S, R, Mphi, Mrew))
    test_be = numpy.append(numpy.array([]), basis.loss_be(basis.theta, basis.w, S, R, Mphi, Mrew))
    true_be = numpy.append(numpy.array([]), model.bellman_error(basis.theta[:-1], weighting = weighting))
    true_lsq = numpy.append(numpy.array([]), model.value_error(basis.theta[:-1], weighting = weighting))

    it = 0
    n_test_inc = 0
    best_theta = None
    best_test_loss = numpy.inf
    try:
        while (n_test_inc < patience):
            it += 1
            print '*** iteration', it, '***'
            old_theta = copy.deepcopy(basis.theta)
            basis.set_params( scipy.optimize.fmin_cg(basis.loss, basis.flat_params, basis.grad,
                              args = (S, R, Mphi, Mrew),
                              full_output = False,
                              maxiter = max_iter, 
                              gtol = 1e-8,
                              ) )
            delta = numpy.linalg.norm(old_theta-basis.theta)
            print 'delta theta: ', delta
            if delta < 1e-5:
                print 'delta theta too small, stopping'
                break

            norms = numpy.apply_along_axis(numpy.linalg.norm, 0, basis.theta)
            print 'column norms: %.2f min / %.2f avg / %.2f max' % (
                norms.min(), norms.mean(), norms.max())
            #basis.theta = numpy.apply_along_axis(lambda v: v/numpy.linalg.norm(v), 0, basis.theta)
            
            err = basis.loss(basis.flat_params, S_test, R_test, Mphi, Mrew)
            test_loss = numpy.append(test_loss, err)
            test_be = numpy.append(test_be, basis.loss_be(basis.theta, basis.w, S, R, Mphi, Mrew))
            true_be = numpy.append(true_be, model.bellman_error(basis.theta[:-1], weighting = weighting))
            true_lsq = numpy.append(true_lsq, model.value_error(basis.theta[:-1], weighting = weighting))
            
            if err < best_test_loss:
                best_test_loss = err
                best_theta = copy.deepcopy(basis.theta)
                n_test_inc = 0
                print 'new best %s loss: ' % basis.loss_type, best_test_loss
            else:
                n_test_inc += 1
                print 'iters without better %s loss: ' % basis.loss_type, n_test_inc

    except KeyboardInterrupt:
            print '\n user stopped current training loop'
    
    basis.theta = best_theta

    return basis, test_loss, test_be, true_be, true_lsq

if __name__ == '__main__':
    plac.call(main)
