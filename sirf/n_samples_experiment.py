import copy
import cPickle as pickle
import numpy
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
import grid_world
from dbel import Model
from bellman_basis import plot_features, SIBasis
import util

# these losses nonconvex?
# combine basis classes 
# use reward, model, Pw^*, bellman schedule
# error bars
# run on cluster - distribute across condor
# predictive/reconstructive loss
# go data

def main(n = 81, k = 16, patience = 10, 
            max_iter = 10, mb_size = 200, env_size = 9, weighting = 'uniform'):
    assert n == env_size**2

    mdp = grid_world.MDP(walls_on = True, size = env_size)    
    m = Model(mdp.env.R, mdp.env.P)
    #m.gam = gam # set for larger envs

    b = SIBasis(n,k, m.gam)

    R = m.R[:,None]
    
    # build initialization targets as P^i * R
    # TODO sample version
    C = m.R[:,None]
    P = m.P
    for i in xrange(k-1):
        C = numpy.hstack((C, numpy.dot(P, m.R)[:,None]))
        P = numpy.dot(P, m.P)

    # sample the hold out test set
    S_test, Sp_test, R_test, _, = mdp.sample_grid_world(2*mb_size, distribution = weighting)

    sample_be = numpy.append(numpy.array([]), b.loss_be(b.theta, S_test, Sp_test, R_test))
    true_be = numpy.append(numpy.array([]), m.bellman_error(b.theta, weighting = weighting))
    true_lsq = numpy.append(numpy.array([]), m.value_error(b.theta, weighting = weighting))

    print 'initial test be, true be, and true lsq losses: ', sample_be[0], true_be[0], true_lsq[0]

    switch = numpy.array([])
    loss_types = ['model','bellman']
    
    # sample once at the beginning
    S, Sp, R, _, = mdp.sample_grid_world(mb_size, distribution = weighting)

    best_test_loss = numpy.inf
    for lt in loss_types:
        try:
            b.loss_type = lt
            n_test_inc = 0
            delta = numpy.inf
            while (n_test_inc < patience):
         
                theta_old = copy.deepcopy(b.theta)
                
                # sample for each minibatch
                #S, Sp, R, _, = mdp.sample_grid_world(mb_size, distribution = weighting)
                #S = numpy.eye(n)
                #Sp = m.P

                b.set_theta( fmin_cg(b.loss, b.theta.flatten(), b.grad,
                                    args = (S, Sp, R, C),
                                    full_output = False,
                                    maxiter = max_iter, 
                                    gtol = 1e-8) )
                delta = numpy.linalg.norm(b.theta - theta_old)
                print 'delta: ', delta
                print 'loss type: ', lt
                
                err = b.loss_be(b.theta, S_test, Sp_test, R_test) 
                sample_be = numpy.append(sample_be, err)
                true_be = numpy.append(true_be, m.bellman_error(b.theta, weighting = weighting))
                true_lsq = numpy.append(true_lsq, m.value_error(b.theta, weighting = weighting))
                
                if err < best_test_loss:
                    best_test_loss = b.loss(b.theta, S_test, Sp_test, R_test)
                    best_theta = b.theta
                    n_best = len(sample_be)-1
                    n_test_inc = 0
                    print 'new best: ', best_test_loss
                else:
                    n_test_inc += 1
                    print 'iters without better loss: ', n_test_inc
                
                #plot_features(b.theta)
                #plt.show()

        except KeyboardInterrupt:
            print '\n user stopped current training loop'

        switch = numpy.append(switch, len(sample_be)-1)

    b.theta = best_theta
    err_test = b.loss_be(b.theta, S_test, Sp_test, R_test) 
    err_true = m.bellman_error(b.theta, weighting = weighting)
    
    print 'final bellman error (test, true): ', err_test,', ', err_true
    
    # TODO move plotting to separate script
    # plot basis functions
    plot_features(b.theta)
    plt.savefig('basis.k=%i.lam=%s.%s.pdf' % (k, str(0.), weighting))
    
    # plot learning curves
    plt.clf()
    ax = plt.axes()
    x = range(len(sample_be))
    ax.plot(x, sample_be/max(sample_be), 'r-', x, true_be/max(true_be), 'g-', x, true_lsq / max(true_lsq), 'b-')

    ypos = numpy.array([numpy.zeros_like(switch), numpy.ones_like(switch)])
    ax.plot(numpy.array([switch,switch]), ypos, 'k-')
    
    ax.plot(n_best, 0, 'go')
    ax.legend(['Test BE','True BE','True RMSE'], loc = 'lower left')
    plt.savefig('loss.k=%i.lam=%s.%s.pdf' % (k, str(0.), weighting))
    
    
    # plot value functions, true and approx
    plt.clf()
    f = plt.figure()
    
    # true model value fn
    f.add_subplot(311)
    plt.imshow(numpy.reshape(m.V, (env_size, env_size)), cmap = 'gray', interpolation = 'nearest')

    # bellman error estimate (using true model)
    f.add_subplot(312)
    w_be = m.get_lstd_weights(b.theta) # TODO add lambda parameter here
    v_be = numpy.dot(b.theta, w_be)
    plt.imshow(numpy.reshape(v_be, (env_size, env_size)), cmap = 'gray', interpolation = 'nearest')
    
    # least squares solution with true value function
    f.add_subplot(313)
    w_lsq = numpy.linalg.lstsq(b.theta, m.V)[0]
    v_lsq = numpy.dot(b.theta, w_lsq)
    plt.imshow(numpy.reshape(v_lsq, (env_size, env_size)), cmap = 'gray', interpolation = 'nearest')
    plt.savefig('value.k=%i.lam=%s.%s.pdf' % (k, str(0.), weighting))

    print 'bellman error norm from v: ', numpy.linalg.norm(m.V - v_be)
    print 'lsq error norm from v: ', numpy.linalg.norm(m.V - v_lsq)

def experiment(k = 100, n_runs = 8, n_samples = None, env_size = 31, gam = 1-1e-4,
                    weighting = 'uniform', patience = 15, max_iter = 10):
    
    training_methods = ['model', 'bellman', 'model-bellman']

    if n_samples is None:
        n_samples = numpy.round(numpy.logspace(2,4,8)).astype(int) # 50 to 5000 samples

    mdp = grid_world.MDP(walls_on = True, size = env_size)    
    m = Model(mdp.env.R, mdp.env.P)
    m.gam = gam # set for larger envs
    dim = env_size**2
    b = SIBasis(dim,k, m.gam)

    test_be = numpy.zeros((len(n_samples), len(training_methods)))
    true_be = numpy.zeros((len(n_samples), len(training_methods)))
    true_lsq = numpy.zeros((len(n_samples), len(training_methods)))

    for i,n in enumerate(n_samples):
        
        for r in xrange(n_runs):
            
            # initialize features sparsely
            theta_init = 1e-7 * numpy.random.standard_normal((dim, k))
            #sparsity = 0.8
            #for c in xrange(k):
                #z = numpy.random.random(dim)
                #theta_init[:,c][z < sparsity] = 0.

            # sample data and hold-out test set
            S, Sp, R, _, = mdp.sample_grid_world(n, distribution = weighting)
            S_test, Sp_test, R_test, _, = mdp.sample_grid_world(n, distribution = weighting)

            for j,tm in enumerate(training_methods):
                
                print 'training with %i samples and %s loss' % (n,tm)
                b.set_theta(theta_init)

                if (tm == 'model') or (tm == 'bellman'):
                    b.loss_type = tm
                    b = train_basis(b, S, Sp, R, S_test, Sp_test, R_test, 
                                                patience, max_iter, weighting)
                elif tm is 'model-bellman':
                    b.loss_type = 'model'
                    b = train_basis(b, S, Sp, R, S_test, Sp_test, R_test, 
                                                patience, max_iter, weighting)
                    b.loss_type = 'bellman'
                    b = train_basis(b, S, Sp, R, S_test, Sp_test, R_test, 
                                                patience, max_iter, weighting)
                else:
                    print tm
                    assert False
                
                # TODO what about variance / error bars?
                test_be[i,j] += b.loss_be(b.theta, S_test, Sp_test, R_test) / float(n) 
                true_be[i,j] += m.bellman_error(b.theta, weighting = weighting)
                true_lsq[i,j] += m.value_error(b.theta, weighting = weighting)
        
        map(lambda x: x/float(n_runs), test_be[i,:])
        map(lambda x: x/float(n_runs), true_be[i,:])
        map(lambda x: x/float(n_runs), true_lsq[i,:])
    
    # save results!
    out_path = 'n_sample_results.k=%i.lam=%s.%s.size=%i.r=%i.pickle.gz' % (k, 
                str(0.), weighting, env_size, n_runs)
    with util.openz(out_path, "wb") as out_file:
        pickle.dump((test_be, true_be, true_lsq), out_file, protocol = -1)
    
    x = range(len(n_samples))
    f = plt.figure()
    ax = f.add_subplot(311)
    ax.plot(x, test_be[:,0], 'r-', x, test_be[:,1], 'b-', x, test_be[:,2], 'g-')
    #ax.legend(training_methods, loc = 3)

    ax = f.add_subplot(312)
    ax.plot(x, true_be[:,0], 'r-', x, true_be[:,1], 'b-', x, true_be[:,2], 'g-')
    #ax.legend(training_methods, loc = 3)

    ax = f.add_subplot(313)
    ax.plot(x, true_lsq[:,0], 'r-', x, true_lsq[:,1], 'b-', x, true_lsq[:,2], 'g-')
    #ax.legend(training_methods, loc = 3)

    plt.savefig('n_samples.k=%i.lam=%s.%s.size=%i.r=%i.pdf' % (k, str(0.),
                                                weighting, env_size, n_runs))       
                
                
def train_basis(basis, S, Sp, R, S_test, Sp_test, R_test, patience, 
                    max_iter, weighting):
    
    n_test_inc = 0
    best_test_loss = numpy.inf
    while (n_test_inc < patience):

        basis.set_theta( fmin_cg(basis.loss, basis.theta.flatten(), basis.grad,
                            args = (S, Sp, R, None), # TODO change args, use Basis
                            full_output = False,
                            maxiter = max_iter, 
                            gtol = 1e-8) )
        
        err = basis.loss(basis.theta, S_test, Sp_test, R_test, None) #basis.loss_be(basis.theta, S_test, Sp_test, R_test) 
        
        if err < best_test_loss:
            best_test_loss = err
            best_theta = copy.deepcopy(basis.theta)
            n_test_inc = 0
            print 'new best %s loss: ' % basis.loss_type, best_test_loss
        else:
            n_test_inc += 1
            print 'iters without better %s loss: ' % basis.loss_type, n_test_inc
    
    basis.theta = best_theta

    return basis
    
        

            
if __name__ == '__main__':
    #main()
    experiment()
