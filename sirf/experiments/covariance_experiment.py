import copy
import itertools
import plac
import cPickle as pickle
import numpy
import grid_world
import scipy.sparse
from dbel import Model
from bellman_basis import BellmanBasis as Basis
from scipy.optimize import fmin_cg
from bellman_basis import plot_features
import matplotlib.pyplot as plt
import util

# add tracking of reward and model loss along with bellman error
# mark shifts and best theta

@plac.annotations(
    k=('number of features', 'option', 'k', int),
    env_size=('size of the grid world', 'option', 's', int),
    lam=('lambda for TD-lambda training', 'option', None, float),
    gam=('discount factor', 'option', 'g', float),
    beta=('covariance loss paramter', 'option', None, float),
    eps=('epsilon for computing TD-lambda horizon', 'option', None, float),
    patience=('train until patience runs out', 'option', 'p', int),
    max_iter=('train for at most this many iterations', 'option', 'i', int),
    weighting=('method for sampling from grid world', 'option', None, str, ['policy', 'uniform']),
    loss_types=('train on these losses in this order', 'option', None, str),
    grad_vars=('compute gradient using these variables', 'option', None, str),
    nonlin=('feature nonlinearity', 'option', 'n', str, ['sigmoid', 'relu']),
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
         weighting = 'policy', 
         reg = ('l1-theta', 5e-2), 
         loss_types = 'covariance bellman',
         grad_vars = 'model',
         nonlin = None,
         nonzero = None,
         ):

    beta_ratio = beta/gam 

    if partition is None:
        partition = {'reward':1, 'model':k-1}        

    print 'building environment'
    mdp = grid_world.MDP(walls_on = True, size = env_size)
    n = env_size**2
    m = Model(mdp.env.R, mdp.env.P, gam = gam)
    m.gam = gam

    print 'constructing basis'
    bb = Basis(n, k, beta_ratio, partition = partition, reg_tuple = reg, nonlin = nonlin, nonzero = nonzero)
    bb.theta[:,-1] = m.R # initialize the last column as the reward function

    R = numpy.array([])
    X = scipy.sparse.eye(n, n) # build dataset
    P = scipy.sparse.eye(n, n)
    n_steps = bb._calc_n_steps(lam, gam, eps)
    for i in xrange(n_steps): # decay epsilon 
        R = numpy.append(R, P * m.R)
        P = m.P * P
        X = scipy.sparse.vstack((X, P)) # todo insert other basis here
    R = R[:,None]
        
    # build bellman operator matrices
    print 'making mixing matrices'
    Mphi, Mrew = bb.get_mixing_matrices(n, lam, gam, sampled = False, eps = eps)
    #PHI_full = bb.encode(X)
    #PHIlam = numpy.dot(Mphi, PHI_full)
    #PHI0 = PHI_full[0:n,:]
    #Rlam = numpy.dot(Mrew, R) # sparse dot?

    #print 'PHIlam: ', PHIlam
    #print 'PHI_full', PHI_full
    #print 'Rlam: ', Rlam

    phases = zip(loss_types.split(), itertools.cycle(grad_vars.split()))

    test_loss = numpy.array([])
    test_be = numpy.array([])
    true_be = numpy.array([])
    true_lsq = numpy.array([])

    for ph in phases:
        lt, wrt = ph
        try:
            print 'training with %s loss with respect to %s vars' % (lt,wrt)
            bb.set_loss(loss_type=lt, wrt=wrt)
            bb.set_regularizer(reg)
            
            theta_old = copy.deepcopy(bb.theta)
            bb, tel, teb, trb, trq = train_basis(bb, m, X, R, X, R, 
                        Mphi, Mrew, patience, max_iter, weighting)

            delta = numpy.linalg.norm(bb.theta - theta_old)
            print 'delta: ', delta
            
            test_loss = numpy.append(test_loss, tel)
            test_be = numpy.append(test_be, teb)
            true_be = numpy.append(true_be, trb)
            true_lsq = numpy.append(true_lsq, trq)

            #plot_features(b.theta)
            #plt.show()

        except KeyboardInterrupt:
            print '\n user stopped current training loop'

    # save results!
    out_path = 'covariance_results.k=%i.reg=%s.lam=%s.%s.size=%i.pickle.gz' % (k, str(reg), str(lam), weighting, env_size)
    with util.openz(out_path, "wb") as out_file:
        pickle.dump((test_loss, test_be, true_be, true_lsq), out_file, protocol = -1)

    # plot basis functions
    plot_features(bb.theta)
    plt.savefig('basis.k=%i.reg=%s.lam=%s.b=%s.%s.pdf' % (k, str(reg), str(lam), str(beta), weighting)) # add gam etc to string?
    
    # plot learning curves
    plot_learning_curves(numpy.array([test_loss, test_be, true_be, true_lsq]), 
                         ['test loss', 'test BE', 'true BE', 'true lsq'],
                         ['r-','g-','b-','k-'])
    plt.savefig('loss.k=%i.reg=%s.lam=%s.b=%s.%s.pdf' % (k, str(reg), str(lam), str(beta), weighting))
    
    # plot value functions
    plot_value_functions(env_size, m, bb)
    plt.savefig('value.k=%i.reg=%s.lam=%s.b=%s.%s.pdf' % (k, str(reg), str(lam), str(beta), weighting))


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
    w_be = m.get_lstd_weights(b.theta) # TODO add lambda parameter here
    v_be = numpy.dot(b.theta, w_be)
    ax.imshow(numpy.reshape(v_be, (size, size)), cmap = 'gray', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # least squares solution with true value function
    ax = f.add_subplot(313)
    w_lsq = numpy.linalg.lstsq(b.theta, m.V)[0]
    v_lsq = numpy.dot(b.theta, w_lsq)
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
        ax.plot(x, l / l.max(), draw_styles[i], label=labels[i])

    plt.ylim(0, 1)
    plt.title('Normalized Losses per CG Minibatch')
    plt.legend()
    #handles, labels = ax.get_legend_handles_labels()
    #ypos = numpy.array([numpy.zeros_like(switch), numpy.ones_like(switch)])
    #ax.plot(numpy.array([switch,switch]), ypos, 'k-') # draw switch positions and best theta
    

def train_basis(basis, model, S, R, S_test, R_test, Mphi, Mrew, patience, 
                    max_iter, weighting):
    
    test_loss = numpy.append(numpy.array([]), basis.loss(basis.theta, S, R, Mphi, Mrew))
    test_be = numpy.append(numpy.array([]), basis.loss_be(basis.theta, S, R, Mphi, Mrew))
    true_be = numpy.append(numpy.array([]), model.bellman_error(basis.theta, weighting = weighting))
    true_lsq = numpy.append(numpy.array([]), model.value_error(basis.theta, weighting = weighting))

    n_test_inc = 0
    best_test_loss = numpy.inf
    try:
        while (n_test_inc < patience):
            
            old_theta = copy.deepcopy(basis.theta)
            basis.set_theta( fmin_cg(basis.loss, basis.theta.flatten(), basis.grad,
                                args = (S, R, Mphi, Mrew),
                                full_output = False,
                                maxiter = max_iter, 
                                gtol = 1e-8) )
            delta = numpy.linalg.norm(old_theta-basis.theta)
            print 'delta theta: ', delta

            print 'normalizing columns; pre norms: ', numpy.apply_along_axis(numpy.linalg.norm, 0, basis.theta).mean()
            basis.theta = numpy.apply_along_axis(lambda v: v/numpy.linalg.norm(v), 0, basis.theta)
            
            err = basis.loss(basis.theta, S_test, R_test, Mphi, Mrew) #basis.loss_be(basis.theta, S_test, Sp_test, R_test)  
            test_loss = numpy.append(test_loss, err)
            test_be = numpy.append(test_be, basis.loss_be(basis.theta, S, R, Mphi, Mrew))
            true_be = numpy.append(true_be, model.bellman_error(basis.theta, weighting = weighting))
            true_lsq = numpy.append(true_lsq, model.value_error(basis.theta, weighting = weighting))
            
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
