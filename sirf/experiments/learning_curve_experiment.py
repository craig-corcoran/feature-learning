import copy
import itertools
import numpy
import pickle
import plac
import scipy.optimize
import scipy.sparse
import theano
import matplotlib.pyplot as plt

import condor
import sirf
import sirf.grid_world as grid_world
import sirf.util as util
from sirf.rl import Model
from sirf.bellman_basis import plot_features, BellmanBasis 

# mark shifts and best theta

theano.gof.compilelock.set_lock_status(False)
theano.config.on_unused_input = 'ignore'
theano.config.warn.sum_div_dimshuffle_bug = False

logger = sirf.get_logger(__name__)

@plac.annotations(
    k=('number of features', 'option', None, int),
    workers=('number of condor workers', 'option', None, int),
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
    nonlin=('feature nonlinearity', 'option', None, str, ['sigmoid', 'relu']),
    nonzero=('penalty for zero theta vectors', 'option', None, float),
    training_methods=('list of tuples of loss fn list and wrt params list', 'option', None, int),
    min_imp=('train until loss improvement percentage is less than this', 'option', None, int)
    )
def main(k = 16,
         workers = 0,
         env_size = 15,
         n_runs = 1,
         lam = 0.,
         gam = 0.998,
         beta = 0.998,
         eps = 1e-5, 
         patience = 15,
         max_iter = 15,
         weighting = 'uniform', 
         l1theta = None,
         l1code = None,
         state_rep = 'tabular',
         n_samples = None,
         nonlin = None,
         nonzero = None,
         training_methods = None,
         min_imp = 0.
         ):

    beta_ratio = beta/gam 

    if training_methods is None:
        training_methods = [
            (['prediction'],[['theta-all']]),
            (['prediction', 'layered'], [['theta-all'],['theta-all','w']]),
            (['covariance'],[['theta-all']]), # with reward, without fine-tuning
            (['covariance', 'layered'], [['theta-all'],['theta-all','w']]), # theta-model here for 2nd wrt?
            (['layered'], [['theta-all','w']])] # baseline

    print 'building environment'
    mdp = grid_world.MDP(walls_on = True, size = env_size)
    n = env_size**2
    m = Model(mdp.env.R, mdp.env.P, gam = gam)
    m.gam = gam

    if n_samples:
        print 'sampling from a grid world'
        kw = dict(n_samples = n_samples, state_rep = state_rep, distribution = weighting)
        S, Sp, R, _ = mdp.sample_grid_world(**kw)
        X = scipy.sparse.vstack((S, Sp[-1, :]))
        S_val, Sp_val, R_val, _ = mdp.sample_grid_world(**kw)
        X_val = scipy.sparse.vstack((S_val, Sp_val[-1, :]))
        S_test, Sp_test, R_test, _ = mdp.sample_grid_world(**kw)
        X_test = scipy.sparse.vstack((S_test, Sp_test[-1, :]))
        losses = ['test-bellman', 'test-reward', 'test-model', 'true-bellman', 'true-lsq']
       
    else:
        print 'using perfect information'
        R = numpy.array([])
        X = scipy.sparse.eye(n, n)
        P = scipy.sparse.eye(n, n)
        for i in xrange(BellmanBasis._calc_n_steps(lam, gam, eps)): # decay epsilon 
            R = numpy.append(R, P * m.R)
            P = m.P * P
            X = scipy.sparse.vstack((X, P))
        R = R[:,None]
        X_val = X_test = X
        R_val = R_test = R
        losses = ['true-bellman', 'true-model', 'true-reward', 'true-lsq']

    # build bellman operator matrices
    print 'making mixing matrices'
    Mphi, Mrew = BellmanBasis.get_mixing_matrices(n_samples or n, lam, gam, sampled = bool(n_samples), eps = eps)

    print 'constructing basis'
    reg = None
    if l1theta is not None:
        reg = ('l1-theta', l1theta)
    if l1code is not None:
        reg = ('l1-code', l1code)

    
    # initialize features with unit norm
    theta_init = numpy.random.standard_normal((n+1, k))
    theta_init /= numpy.sqrt((theta_init * theta_init).sum(axis=0))

    w_init = numpy.random.standard_normal((k+1,1)) 
    w_init = w_init / numpy.linalg.norm(w_init) 

    bb_params = [n + 1, [k], beta_ratio]
    bb_dict = dict( reg_tuple = reg, nonlin = nonlin,
        nonzero = nonzero, w = w_init, thetas = [theta_init])

    # initialize loss dictionary
    d_loss_data = {}
    for key in losses:
        d_loss_data[key] = numpy.array([])
    
    def yield_jobs():
        for tm in training_methods:
            loss_list, wrt_list = tm
            assert len(loss_list) == len(wrt_list)
            
            out_string = '%s.k=%i.reg=%s.lam=%s.gam=%s.b=%s.%s.%s%s%s.' % (
                str(tm),
                k,
                str(reg),
                lam, gam, beta, weighting, '+'.join(losses),
                '.samples=%d' % n_samples if n_samples else '',
                '.nonlin=%s' % nonlin if nonlin else '')

            yield (train_basis, [bb_params, bb_dict, tm, m, d_loss_data, X, R,  
                X_val, R_val, X_test, R_test, Mphi, Mrew, patience, max_iter, 
                weighting, out_string, min_imp, env_size])

    # launch condor jobs
    for (basis, d_loss_out) in condor.do(yield_jobs(), workers):
        pass


    def output(prefix, suffix='pdf'):
        return '%s.k=%i.reg=%s.lam=%s.gam=%s.b=%s.%s.%s%s%s.%s' % (
            prefix, k,
            str(reg),
            lam, gam, beta, weighting, '+'.join(losses),
            '.samples=%d' % n_samples if n_samples else '',
            '.nonlin=%s' % nonlin if nonlin else '',
            suffix)

def train_basis(basis_params, basis_dict, method, model, d_loss, S, R,  
            S_val, R_val, S_test, R_test, Mphi, Mrew, patience, max_iter, 
            weighting, out_string, min_imp, env_size):

    print 'training basis using training method: ', str(method)
    
    loss_list, wrt_list = method
    assert len(loss_list) == len(wrt_list)
    basis = BellmanBasis(*basis_params, **basis_dict)

    for i,loss in enumerate(loss_list):
        basis.set_loss(loss, wrt_list[i])
    
        it = 0
        waiting = 0
        best_theta = None
        best_test_loss = numpy.inf
        try:
            while (waiting < patience):
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

                norms = numpy.apply_along_axis(numpy.linalg.norm, 0, basis.theta)
                print 'column norms: %.2f min / %.2f avg / %.2f max' % (
                    norms.min(), norms.mean(), norms.max())
                #basis.theta = numpy.apply_along_axis(lambda v: v/numpy.linalg.norm(v), 0, basis.theta)
                
                err = basis.loss(basis.flat_params, S_val, R_val, Mphi, Mrew)

                if (best_test_loss - err) / best_test_loss > min_imp:
                    best_test_loss = err
                    best_theta = copy.deepcopy(basis.theta)
                    waiting = 0
                    print 'new best %s loss: ' % basis.loss_type, best_test_loss
                else:
                    waiting += 1
                    print 'iters without better %s loss: ' % basis.loss_type, waiting

                # record losses with test set
                for loss, arr in d_loss.items():
                    if loss == 'test-training':
                        val = basis.loss(basis.flat_params, S_test, R_test, Mphi, Mrew)
                    elif loss == 'test-bellman':
                        val = basis.loss_be(basis.theta, basis.w, S_val, R_val, Mphi, Mrew)
                    elif loss == 'test-reward':
                        val = basis.loss_r(basis.theta, basis.w, S_val, R_val, Mphi, Mrew)
                    elif loss == 'test-model':
                        val = basis.loss_r(basis.theta, basis.w, S_val, R_val, Mphi, Mrew)
                    elif loss == 'true-bellman':
                        val = model.bellman_error(basis.theta[:-1], weighting = weighting)
                    elif loss == 'true-reward':
                        val = model.reward_error(basis.theta[:-1], weighting = weighting)
                    elif loss == 'true-model':
                        val = model.model_error(basis.theta[:-1], weighting = weighting)

                    arr = numpy.append(arr, val)

        except KeyboardInterrupt:
            print '\n user stopped current training loop'

        # save results!
        with util.openz('sirf/output/pickle/learning_curve_results' + out_string + 'pickle.gz', "wb") as out_file:
            pickle.dump(d_loss, out_file, protocol = -1)

        # plot basis functions
        plot_features(basis.theta[:-1])
        plt.savefig('sirf/output/plots/basis' + out_string+ '.pdf')
        
        # plot learning curves
        plot_learning_curves(d_loss)
        plt.savefig('sirf/output/plots/loss' + out_string + '.pdf')
        
        # plot value functions
        plot_value_functions(env_size, model, basis)
        plt.savefig('sirf/output/plots/value' + out_string + '.pdf')
    
        basis.theta = best_theta

    return basis, d_loss

def plot_learning_curves(d_loss, draw_styles = ['r-','g-','b-','k-']):
    plt.clf()
    ax = plt.axes()
    for name, curve in d_loss.items():
        x = range(len(curve))
        ax.plot(x, curve / curve.mean(), label=name)
        
    plt.ylim(0, 3)
    plt.title('Normalized Losses per CG Minibatch')
    plt.legend(loc='upper right')

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




if __name__ == '__main__':
    sirf.script(main)
