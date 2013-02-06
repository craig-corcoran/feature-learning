import copy
import cPickle as pickle
import numpy
import grid_world
from dbel import Model
from bellman_basis import BellmanBasis as Basis
from scipy.optimize import fmin_cg
from bellman_basis import plot_features
import matplotlib.pyplot as plt
import util

def main(k = 16, env_size = 15, lam = 0., gam = 1-1e-4, beta = 0.999, eps = 1e-5, 
    partition = None, patience = 15, max_iter = 15, weighting = 'policy'):
        
    beta = beta/gam # just trust me

    if partition is None:
        partition = {'reward':1, 'model':k-1}        

    print 'building environment'
    mdp = grid_world.MDP(walls_on = True, size = env_size)
    n = env_size**2
    m = Model(mdp.env.R, mdp.env.P, gam = gam)
    m.gam = gam

    print 'constructing basis'
    bb = Basis(n, k, beta, partition = partition)
    bb.theta[:,0] = m.R # initialize the first column as the reward function
    
    R = numpy.array([])
    X = numpy.eye(n) # build dataset
    P = numpy.eye(n)
    n_steps = bb._calc_n_steps(lam, gam, eps)
    for i in xrange(n_steps): # decay epsilon 
        R = numpy.append(R, numpy.dot(P, m.R))
        P = numpy.dot(m.P, P)
        X = numpy.vstack((X, numpy.dot(P, numpy.eye(n)))) # todo insert other basis here
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

    loss_types = ['covariance']
    grad_vars = ['model']
    phases = zip(loss_types, grad_vars)

    test_loss = numpy.array([])
    test_be = numpy.array([])
    true_be = numpy.array([])
    true_lsq = numpy.array([])

    for ph in phases:
        lt, wrt = ph
        try:
            print 'training with %s loss with respect to %s vars' % (lt,wrt)
            bb.set_loss(loss_type=lt, wrt='model')
            
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
    out_path = 'covariance_results.k=%i.lam=%s.%s.size=%i.pickle.gz' % (k, str(0.), weighting, env_size)
    with util.openz(out_path, "wb") as out_file:
        pickle.dump((test_loss, test_be, true_be, true_lsq), out_file, protocol = -1)
    
    # plot basis functions
    plt.clf()
    plot_features(bb.theta)
    plt.savefig('basis.k=%i.lam=%s.b=%s.%s.pdf' % (k, str(lam), str(beta), weighting)) # add gam etc to string?
    
    # plot learning curves
    plot_learning_curves(numpy.array([test_loss, test_be, true_be, true_lsq]), 
                         ['test loss', 'test BE', 'true BE', 'true lsq'],
                         ['r-','g-','b-','k-'])
    plt.savefig('loss.k=%i.lam=%s.b=%s.%s.pdf' % (k, str(lam), str(beta), weighting))
    
    # plot value functions
    plot_value_functions(env_size, m, bb)
    plt.savefig('value.k=%i.lam=%s.b=%s.%s.pdf' % (k, str(lam), str(beta), weighting))


def plot_value_functions(size, m, b):
    # plot value functions, true and approx
    plt.clf()
    f = plt.figure()
    
    # todo something wrong with model vf?
    # true model value fn
    f.add_subplot(311)
    plt.imshow(numpy.reshape(m.V, (size, size)), cmap = 'gray', interpolation = 'nearest')

    # bellman error estimate (using true model)
    f.add_subplot(312)
    w_be = m.get_lstd_weights(b.theta) # TODO add lambda parameter here
    v_be = numpy.dot(b.theta, w_be)
    plt.imshow(numpy.reshape(v_be, (size, size)), cmap = 'gray', interpolation = 'nearest')
    
    # least squares solution with true value function
    f.add_subplot(313)
    w_lsq = numpy.linalg.lstsq(b.theta, m.V)[0]
    v_lsq = numpy.dot(b.theta, w_lsq)
    plt.imshow(numpy.reshape(v_lsq, (size, size)), cmap = 'gray', interpolation = 'nearest')

    print 'bellman error norm from v: ', numpy.linalg.norm(m.V - v_be)
    print 'lsq error norm from v: ', numpy.linalg.norm(m.V - v_lsq)

def plot_learning_curves(losses, labels, draw_styles = ['r-','g-','b-','k-']):
    plt.clf()
    ax = plt.axes()
    n_losses = len(losses)
    
    print losses, labels, draw_styles
    for i in xrange(n_losses):
        x = range(len(losses[i]))
        ax.plot(x, losses[i]/numpy.mean(losses[i]), draw_styles[i], label=labels[i])

    ax.ylim(ymax=3)
    ax.title('Loss normalized by mean')
    ax.legend()
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

            print 'normalizing columns; pre norms: ', numpy.apply_along_axis(numpy.linalg.norm, 0, basis.theta)
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
    main()



    

    


