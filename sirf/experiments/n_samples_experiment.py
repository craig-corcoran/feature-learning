import os
import copy
import cPickle as pickle
import numpy
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
import scipy.sparse
import condor
import sirf
import sirf.util as util
import sirf.grid_world as grid_world
from sirf.rl import Model
from sirf.bellman_basis import BellmanBasis
import theano

theano.gof.compilelock.set_lock_status(False)
theano.config.warn.sum_div_dimshuffle_bug = False
theano.config.on_unused_input = 'ignore'
        
logger = sirf.get_logger(__name__)

# update model / perfect info loss
# weighting and loss measures - policy loss
# include reward learning
# nonlinear feature setup
def experiment(workers = 80, n_runs = 9, k = 16, env_size = 15, gam = 0.998, lam = 0., eps = 1e-5,  
    partition = None, patience = 8, max_iter = 8, weighting = 'uniform', reward_init = False,
    nonlin = 1e-8, n_samples = None, beta_ratio = 1.,
    training_methods = None):
    
    if training_methods is None:
        # note: for each loss string, you need a corresponding wrt list
        if reward_init:
            training_methods = [
            (['prediction'],[['theta-model']]),
            (['prediction', 'layered'], [['theta-model'],['theta-model','w']]),
            (['covariance'],[['theta-model']]), # with reward, without fine-tuning
            (['covariance', 'layered'], [['theta-model'],['theta-model','w']]), # theta-model here for 2nd wrt?
            (['layered'], [['theta-all','w']])] # baseline
        
        else:
            training_methods = [(['prediction'],[['theta-all']]),
            (['prediction', 'layered'], [['theta-all'],['theta-all','w']]),
            (['covariance'],[['theta-all']]), # with reward, without fine-tuning
            (['covariance', 'layered'], [['theta-all'],['theta-all','w']]), # theta-model here for 2nd wrt?
            (['layered'], [['theta-all','w']])] # baseline
    
    theano.gof.compilelock.set_lock_status(False)
    theano.config.on_unused_input = 'ignore'
    theano.config.warn.sum_div_dimshuffle_bug = False

    if n_samples is None:
        #n_samples = [100,500]
        n_samples = numpy.round(numpy.linspace(50,1500,6)).astype(int) 

    if partition is None:
        partition = {'theta-model':k-1, 'theta-reward':1}

    mdp = grid_world.MDP(walls_on = True, size = env_size)    
    m = Model(mdp.env.R, mdp.env.P, gam = gam)
    dim = env_size**2

    # tracked losses

    losses = ['test-bellman', 'test-reward', 'test-model', 'true-bellman', 'true-lsq'] 
    logger.info('losses tracked: '+ str(losses))
    
    #n_extra = bb._calc_n_steps(lam, gam, eps)
    #print 'n extra sampled needed: ', n_extra
    d_loss_data = {}
    for key in losses:
        d_loss_data[key] = numpy.zeros((len(n_samples), n_runs, len(training_methods)))

    def yield_jobs():

        for i,n in enumerate(n_samples):
            
            Mphi, Mrew = BellmanBasis.get_mixing_matrices(n, lam, gam, sampled = True, eps = eps)

            for r in xrange(n_runs):
                
                # initialize features with unit norm
                theta_init = numpy.random.standard_normal((dim+1, k))
                if reward_init:
                    theta_init[:-1,-1] = m.R # XXX set last column to reward
                    theta_init[-1,-1] = 0
                theta_init /= numpy.sqrt((theta_init * theta_init).sum(axis=0))

                w_init = numpy.random.standard_normal((k+1,1)) 
                w_init = w_init / numpy.linalg.norm(w_init)

                # sample data: training, validation, and test sets
                S, Sp, R, _, = mdp.sample_grid_world(n, distribution = weighting); 
                S = numpy.vstack((S, Sp[-1,:]))
                S_val, Sp_val, R_val, _, = mdp.sample_grid_world(n, distribution = weighting)
                S_val = scipy.sparse.vstack((S_val, Sp_val[-1,:]))
                S_test, Sp_test, R_test, _, = mdp.sample_grid_world(n, distribution = weighting)
                S_test = scipy.sparse.vstack((S_test, Sp_test[-1,:]))
                
                bb = BellmanBasis(dim+1, k, beta_ratio, partition = partition, 
                    theta = theta_init, w = w_init, record_loss = losses, nonlin = nonlin)
                
                for j,tm in enumerate(training_methods):
                    
                    yield (condor_job,[(i,r,j), bb, m, tm, 
                            S, R, S_val, R_val, S_test, R_test,
                            Mphi, Mrew, patience, max_iter, weighting])

    # aggregate the condor data
    for (_, result) in condor.do(yield_jobs(), workers):
            d_batch_loss, ind_tuple = result
            for name in d_batch_loss.keys():
                d_loss_data[name][ind_tuple] = d_batch_loss[name]

    # save results! 
    pi_root = 'n_samples_results_rinit' if reward_init else 'n_samples_results'    
    out_path = os.getcwd()+'/sirf/output/pickle/%s.no_r.k=%i.l=%s.g=%s.%s.size=%i.r=%i..pickle.gz' \
                    % (pi_root, k, str(lam), str(gam), weighting, env_size, n_runs)
    logger.info('saving results to %s' % out_path)
    with util.openz(out_path, "wb") as out_file:
        pickle.dump(d_loss_data, out_file, protocol = -1)
    
    x = numpy.array(n_samples, dtype = numpy.float64) #range(len(n_samples))
    f = plt.figure()
    logger.info('plotting')
    plot_styles = ['r-', 'b-', 'g-', 'k-', 'c-', 'm-']
    for i,(key,mat) in enumerate(d_loss_data.items()):

        ax = f.add_subplot(2,3,i+1) # todo generalize for arb length 
        
        for h,tm in enumerate(training_methods):                

            std = numpy.std(mat[:,:,h], axis=1)
            mn = numpy.mean(mat[:,:,h], axis=1)
            if 'test' in key:
                mn = mn/x
                std = std/x
            ax.fill_between(x, mn-std, mn+std, facecolor='yellow', alpha=0.15)
            ax.plot(x, mn, plot_styles[h], label = str(tm[0]))
            plt.title(key)
            #plt.axis('off')
            #plt.legend(loc = 3) # lower left
    
    pl_root = 'n_samples_rinit' if reward_init else 'n_samples'
    plt.savefig(os.getcwd()+'/sirf/output/plots/%s.n=%i-%i.k=%i.l=%s.g=%s.%s.size=%i.r=%i.pdf' 
        % (n_samples[0], n_samples[-1], pl_root, k, 
        str(lam), str(gam), weighting, env_size, n_runs))  

def condor_job(ind_tuple, bb, model, method, S, R, S_val, R_val, S_test, R_test, Mphi, 
            Mrew, patience, max_iter, weighting):
    
    logger.info( 'training with %i samples and %s method' % (S.shape[0], method))
    loss_list, wrt_list = method
    assert len(loss_list) == len(wrt_list)

    recordable = 'test-bellman test-reward test-model true-bellman true-lsq'.split()
    record_funs = [bb.loss_be , bb.loss_r, bb.loss_m, model.bellman_error, model.value_error] # xxx hardcoded here
    d_loss_funs = dict(zip(recordable, record_funs))

    for i, loss in enumerate(loss_list):
        bb.set_loss(loss, wrt_list[i])
        bb = train_basis(bb, S, R, S_val, R_val, Mphi, Mrew, patience, 
                    max_iter, weighting)
 
    d_batch_loss = {}
    for key,fun in d_loss_funs.items():
        if 'test' in key:
            d_batch_loss[key] = fun(bb.theta, bb.w, S_test, R_test, Mphi, Mrew) 
        elif 'true' in key:
            d_batch_loss[key] = fun(bb.theta[:-1], weighting = weighting) # todo include lambda here/model, also w 

    return d_batch_loss, ind_tuple
  

def train_basis(basis, S, R, S_test, R_test, Mphi, Mrew, patience, 
                    max_iter, weighting, min_imp = 1e-5):
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
                logger.info( 'iters without better %s loss: ' % basis.loss_type, n_test_inc)
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
