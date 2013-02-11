import copy
import cPickle as pickle
import numpy
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
import scipy.sparse
import condor
import sirf
#import util
#import grid_world
import sirf.util as util
import sirf.grid_world as grid_world
from sirf.rl import Model
from sirf.bellman_basis import BellmanBasis
import theano

theano.gof.compilelock.set_lock_status(False)
theano.config.warn.sum_div_dimshuffle_bug = False
theano.config.on_unused_input = 'ignore'
        
logger = sirf.get_logger(__name__)

# weighting and loss measures - policy loss
# include reward learning
# nonlinear feature setup
def experiment(workers = 2, n_runs = 1, k = 16, env_size = 15, gam = 0.995, lam = 0., eps = 1e-5,  
    partition = None, patience = 1, max_iter = 3, weighting = 'uniform',
    n_samples = None, beta_ratio = 1.,
    training_methods = ['covariance', 'layered', 'covariance-layered']): 
    
    theano.gof.compilelock.set_lock_status(False)
    theano.config.on_unused_input = 'ignore'
    theano.config.warn.sum_div_dimshuffle_bug = False

    if n_samples is None:
        n_samples = [100,500]
        #n_samples = numpy.round(numpy.linspace(50,5000,8)).astype(int) # 50 to 5000 samples

    if partition is None:
        partition = {'theta-model':k-1, 'theta-reward':1}

    mdp = grid_world.MDP(walls_on = True, size = env_size)    
    m = Model(mdp.env.R, mdp.env.P, gam = gam)
    dim = env_size**2

    # tracked losses
    logger.
    losses = ['test-bellman', 'test-reward', 'test-model'] #, 'true-bellman', 'true-lsq']
    
    #n_extra = bb._calc_n_steps(lam, gam, eps)
    #print 'n extra sampled needed: ', n_extra
    d_loss_data = {}
    for key in losses:
        d_loss_data[key] = numpy.zeros((len(n_samples), n_runs, len(training_methods)))

    def yield_jobs():

        for i,n in enumerate(n_samples):
            
            Mphi, Mrew = BellmanBasis.get_mixing_matrices(n, lam, gam, sampled = True, eps = eps, dim = dim)

            for r in xrange(n_runs):
                
                # initialize features with unit norm
                theta_init = numpy.random.standard_normal((dim, k))
                theta_init[:,-1] = m.R # XXX set last column to reward
                theta_init /= numpy.sqrt((theta_init * theta_init).sum(axis=0))

                w_init = numpy.random.standard_normal((k,1)) 
                w_init = w_init / numpy.linalg.norm(w_init)

                # sample data and hold-out test set
                S, Sp, R, _, = mdp.sample_grid_world(n, distribution = weighting); 
                S = numpy.vstack((S, Sp[-1,:]))
                S_test, Sp_test, R_test, _, = mdp.sample_grid_world(n, distribution = weighting)
                S_test = scipy.sparse.vstack((S_test, Sp_test[-1,:]))

                bb = BellmanBasis(dim, k, beta_ratio, partition = partition, 
                    theta = theta_init, w = w_init, record_loss = losses)
                
                for j,tm in enumerate(training_methods):
                    
                    yield (condor_job,[(i,r,j), bb, tm, S, R, S_test, R_test,
                            Mphi, Mrew, patience, max_iter, weighting])

    # aggregate the condor data
    for (_, result) in condor.do(yield_jobs(), workers):
            d_batch_loss, ind_tuple = result
            for name in d_batch_loss.keys():
                d_loss_data[name][ind_tuple] = d_batch_loss[name]

    # save results!
    out_path = './sirf/output/pickle/n_sample_results.k=%i.l=%s.g=%s.%s.size=%i.r=%i..pickle.gz' \
                        % (k, str(lam), str(gam), weighting, env_size, n_runs)
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

    plt.savefig('./sirf/output/plots/n_samples.k=%i.l=%s.g=%s.%s.size=%i.r=%i.pdf' 
            % (k, str(lam), str(gam), weighting, env_size, n_runs))  

def condor_job(ind_tuple, bb, method, S, R, S_test, R_test, Mphi, 
            Mrew, patience, max_iter, weighting):
    
    logger.info( 'training with %i samples and %s method' % (S.shape[0], method))

    if (method == 'covariance'): 
        bb.set_loss(method, ['theta-all'])
        bb = train_basis(bb, S, R, S_test, R_test, Mphi, Mrew,
                                    patience, max_iter, weighting)
    elif (method == 'layered'):
        bb.set_loss(method, ['theta-all', 'w'])
        bb = train_basis(bb, S, R, S_test, R_test, Mphi, Mrew,
                                    patience, max_iter, weighting)
    elif method == 'covariance-layered':
        bb.set_loss('covariance', ['theta-model'])
        bb = train_basis(bb, S, R, S_test, R_test, Mphi, Mrew,
                                    patience, max_iter, weighting)
        bb.set_loss('layered', ['theta-model', 'w'])
        bb = train_basis(bb, S, R, S_test, R_test, Mphi, Mrew,
                                    patience, max_iter, weighting) # b= redundant?
    else:
        print 'unrecognized training method string: ', method
        assert False
    
    d_batch_loss = {}
    for key,fun in bb.d_loss_funcs.items():
        if 'test' in key:
            d_batch_loss[key] = fun(bb.theta, bb.w, S_test, R_test, Mphi, Mrew) 
        elif 'true' in key:
            d_batch_loss[key] = fun(bb.theta, weighting = weighting) # TODO include lambda here/model, also w 

    return d_batch_loss, ind_tuple
  

def train_basis(basis, S, R, S_test, R_test, Mphi, Mrew, patience, 
                    max_iter, weighting, min_imp = 1e-3):
    try:
        n_test_inc = 0
        best_test_loss = numpy.inf
        while (n_test_inc < patience):
            
            basis.set_params( fmin_cg(basis.loss, basis.flat_params, basis.grad,
                                args = (S, R, Mphi, Mrew), 
                                full_output = False,
                                maxiter = max_iter, 
                                gtol = 1e-8) )
            
            err = basis.loss(basis.flat_params, S_test, R_test, Mphi, Mrew) 
            
            if err < (best_test_loss - min_imp):
                n_test_inc = 0
                print 'new best %s loss: ' % basis.loss_type, best_test_loss
            else:
                n_test_inc += 1
                print 'iters without better %s loss: ' % basis.loss_type, n_test_inc
            if err < best_test_loss:
                best_test_loss = err
                best_theta = copy.deepcopy(basis.theta)

        basis.theta = best_theta

    except KeyboardInterrupt:
        print '\n user stopped current training loop'

    return basis
     
if __name__ == '__main__':
    #main()
    experiment()
