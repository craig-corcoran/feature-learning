import copy
import cPickle as pickle
import numpy
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
import scipy.sparse
import condor
import util
import grid_world
from rl import Model
from bellman_basis import BellmanBasis
import theano


def experiment(workers = 1, n_runs = 10, k = 16, env_size = 15, gam = 0.995, lam = 0., eps = 1e-5,  
    partition = None, patience = 15, max_iter = 15, weighting = 'uniform',
    n_samples = None, beta_ratio = 1.,
    training_methods = ['covariance', 'bellman', 'covariance-bellman']):
    
    theano.config.warn.sum_div_dimshuffle_bug = False

    if n_samples is None:
        #n_samples = [100,500]
        n_samples = numpy.round(numpy.linspace(50,5000,8)).astype(int) # 50 to 5000 samples

    if partition is None:
        partition = {'theta-model':k-1, 'theta-reward':1}

    mdp = grid_world.MDP(walls_on = True, size = env_size)    
    m = Model(mdp.env.R, mdp.env.P, gam = gam)
    dim = env_size**2
    
    #n_extra = bb._calc_n_steps(lam, gam, eps)
    #print 'n extra sampled needed: ', n_extra
    bb = BellmanBasis(dim, k, beta_ratio, partition = partition)
    d_loss_funcs = {'test-bellman': bb.loss_be , 'test-reward': bb.loss_r,
                    'test-model': bb.loss_m, 'true-bellman': m.bellman_error,
                    'true-lsq': m.value_error}
    d_loss_data = {}
    for key in d_loss_funcs.iterkeys():
        d_loss_data[key] = numpy.zeros((len(n_samples), n_runs, len(training_methods)))

    def yield_jobs():

        for i,n in enumerate(n_samples):
            
            Mphi, Mrew = bb.get_mixing_matrices(n, lam, gam, sampled = True, eps = eps, dim = dim)

            for r in xrange(n_runs):
                
                # initialize features (sparsely?)
                print dim, k
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
                    
                    yield (condor_job,[(i,r,j), d_loss_funcs, tm, theta_init, dim, k,
                                        beta_ratio, partition, S, R, S_test, R_test,
                                        Mphi, Mrew, patience, max_iter, weighting])

    # aggregate the condor data
    for (_, result) in condor.do(yield_jobs(), workers):
            d_batch_loss, ind_tuple = result
            for name in d_batch_loss.keys():
                d_loss_data[name][ind_tuple] = d_batch_loss[name]

    # save results!
    out_path = 'n_sample_results.k=%i.l=%s.g=%s.%s.size=%i.r=%i..pickle.gz' % (k, str(lam), str(gam), weighting, env_size, n_runs)
    with util.openz(out_path, "wb") as out_file:
        pickle.dump(d_loss_data, out_file, protocol = -1)
    
    x = range(len(n_samples))
    f = plt.figure()
    
    plot_styles = ['r-', 'b-', 'g-']
    for i,(key,mat) in enumerate(d_loss_data.items()):

        ax = f.add_subplot(2,3,i+1) # todo generalize for arb length 
        
        for h,tm in enumerate(training_methods):                

            std = numpy.std(mat[:,:,h], axis=1)
            mn = numpy.mean(mat[:,:,h], axis=1)
            ax.fill_between(x, mn-std, mn+std)
            ax.plot(x, mn, plot_styles[h], label = tm)
            plt.title(key)
            plt.axis('off')
            #plt.legend(loc = 3) # lower left

    plt.savefig('n_samples.k=%i.l=%s.g=%s.%s.size=%i.r=%i.pdf' % (k, str(lam), 
                        str(gam), weighting, env_size, n_runs))  

def condor_job(ind_tuple, d_loss_funcs, method, theta_init, dim, k, beta_ratio,
    partition, S, R, S_test, R_test, Mphi, Mrew, patience, max_iter, weighting):
    
    bb = BellmanBasis(dim, k, beta_ratio, partition = partition, theta = theta_init)
    #print 'training with %i samples and %s loss' % (S.shape[0], method)

    if (method == 'covariance') or (method == 'bellman'):
        bb.set_loss(method, ['theta-all'])
        bb = train_basis(bb, S, R, S_test, R_test, Mphi, Mrew,
                                    patience, max_iter, weighting)
    elif method == 'covariance-bellman':
        bb.set_loss('covariance', ['theta-model'])
        bb = train_basis(bb, S, R, S_test, R_test, Mphi, Mrew,
                                    patience, max_iter, weighting)
        bb.set_loss('bellman', ['theta-model'])
        bb = train_basis(bb, S, R, S_test, R_test, Mphi, Mrew,
                                    patience, max_iter, weighting)
    else:
        print 'unrecognized training method string: ', method
        assert False
    
    d_batch_loss = {}
    for key,fun in d_loss_funcs.items():
        if 'test' in key:
            d_batch_loss[key] = fun(bb.theta, S_test, R_test, Mphi, Mrew) 
        elif 'true' in key:
            d_batch_loss[key] = fun(bb.theta, weighting = weighting) # TODO include lambda here/model 

    return d_batch_loss, ind_tuple
  

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
