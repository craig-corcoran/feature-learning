import copy
import cPickle as pickle
import numpy
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
import grid_world
from rl import Model
import scipy.sparse
from bellman_basis import BellmanBasis
import util

def experiment(n_runs = 10, k = 16, env_size = 15, gam = 0.995, lam = 0., eps = 1e-5,  
    partition = None, patience = 15, max_iter = 15, weighting = 'policy',
    n_samples = None,
    training_methods = ['covariance', 'bellman', 'covariance-bellman']):


    if n_samples is None:
        #n_samples = [100,500]
        n_samples = numpy.round(numpy.linspace(50,5000,8)).astype(int) # 50 to 5000 samples

    if partition is None:
        partition = {'model':k-1, 'reward':1}

    mdp = grid_world.MDP(walls_on = True, size = env_size)    
    m = Model(mdp.env.R, mdp.env.P, gam = gam)
    dim = env_size**2
    bb = BellmanBasis(dim, k, 1., partition = partition)
    n_extra = bb._calc_n_steps(lam, gam, eps)
    print 'n extra sampled needed: ', n_extra

    print 'making mixing matrices'
    # todo dict here
    test_be = numpy.zeros((len(n_samples), n_runs, len(training_methods)))
    test_re = numpy.zeros((len(n_samples), n_runs, len(training_methods)))
    test_me = numpy.zeros((len(n_samples), n_runs, len(training_methods)))
    true_be = numpy.zeros((len(n_samples), n_runs, len(training_methods)))
    true_lsq = numpy.zeros((len(n_samples), n_runs, len(training_methods)))
    figures = [test_be, test_re, test_me, true_be, true_lsq]

    for i,n in enumerate(n_samples):
        
        Mphi, Mrew = bb.get_mixing_matrices(n, lam, gam, sampled = True, eps = eps, dim = dim)

        for r in xrange(n_runs):
            
            # initialize features (sparsely?)
            theta_init = 1e-7 * numpy.random.standard_normal((dim, k))
            theta_init[:,-1] = m.R # set last column to reward
            #sparsity = 0.8
            #for c in xrange(k):
                #z = numpy.random.random(dim)
                #theta_init[:,c][z < sparsity] = 0.

            # sample data and hold-out test set
            S, Sp, R, _, = mdp.sample_grid_world(n, distribution = weighting); 
            S = numpy.vstack((S, Sp[-1,:]))
            S_test, Sp_test, R_test, _, = mdp.sample_grid_world(n, distribution = weighting)
            S_test = scipy.sparse.vstack((S_test, Sp_test[-1,:]))

            for j,tm in enumerate(training_methods):
                
                print 'training with %i samples and %s loss' % (n,tm)
                bb.set_theta(theta_init)

                if (tm == 'covariance') or (tm == 'bellman'):
                    print tm
                    bb.set_loss(tm, 'all')
                    bb = train_basis(bb, S, R, S_test, R_test, Mphi, Mrew,
                                                patience, max_iter, weighting)
                elif tm == 'covariance-bellman':
                    bb.set_loss('covariance', 'model')
                    bb = train_basis(bb, S, R, S_test, R_test, Mphi, Mrew,
                                                patience, max_iter, weighting)
                    bb.set_loss('bellman', 'model')
                    bb = train_basis(bb, S, R, S_test, R_test, Mphi, Mrew,
                                                patience, max_iter, weighting)
                else:
                    print 'unrecognized training method string: ', tm
                    assert False
                
                test_be[i,r,j] = bb.loss_be(bb.theta, S_test, R_test, Mphi, Mrew) / float(n) 
                test_re[i,r,j] = bb.loss_m(bb.theta, S_test, R_test, Mphi, Mrew) / float(n)
                test_me[i,r,j] = bb.loss_r(bb.theta, S_test, R_test, Mphi, Mrew) / float(n)
                true_be[i,r,j] = m.bellman_error(bb.theta, weighting = weighting)
                true_lsq[i,r,j] = m.value_error(bb.theta, weighting = weighting)
    


    
    # save results!
    out_path = 'n_sample_results.k=%i.l=%s.g=%s.%s.size=%i.r=%i..pickle.gz' % (k, str(lam), str(gam), weighting, env_size, n_runs)
    with util.openz(out_path, "wb") as out_file:
        pickle.dump(figures, out_file, protocol = -1)
    
    x = range(len(n_samples))
    f = plt.figure()
    
    names = ['Test Bellman', 'Test Reward', 'Test Model', 'True Bellman', 'True Least Sq.']
    plot_styles = ['r-', 'b-', 'g-']
    for i in xrange(len(figures)):
        #r = i % 3
        #c = i % 2
        ax = f.add_subplot(2,3,i+1)
        
        for h,tm in enumerate(training_methods):                

            std = numpy.std(figures[i][:,:,h], axis=1)
            mn = numpy.mean(figures[i][:,:,h], axis=1)
            ax.fill_between(x, mn-std, mn+std)
            ax.plot(x, mn, plot_styles[h], label = tm)
            plt.title(names[i])
            plt.axis('off')
            #plt.legend(loc = 3) # lower left

    plt.savefig('n_samples.k=%i.l=%s.g=%s.%s.size=%i.r=%i.pdf' % (k, str(lam), 
                        str(gam), weighting, env_size, n_runs))       
                

def train_basis(basis, S, R, S_test, R_test, Mphi, Mrew, patience, 
                    max_iter, weighting):
    
    try:
        n_test_inc = 0
        best_test_loss = numpy.inf
        while (n_test_inc < patience):

            basis.set_theta( fmin_cg(basis.loss, basis.theta.flatten(), basis.grad,
                                args = (S, R, Mphi, Mrew), 
                                full_output = False,
                                maxiter = max_iter, 
                                gtol = 1e-8) )
            
            err = basis.loss(basis.theta, S_test, R_test, Mphi, Mrew) 
            
            if err < best_test_loss:
                best_test_loss = err
                best_theta = copy.deepcopy(basis.theta)
                n_test_inc = 0
                print 'new best %s loss: ' % basis.loss_type, best_test_loss
            else:
                n_test_inc += 1
                print 'iters without better %s loss: ' % basis.loss_type, n_test_inc

        basis.theta = best_theta

    except KeyboardInterrupt:
        print '\n user stopped current training loop'

    return basis
     
if __name__ == '__main__':
    #main()
    experiment()
