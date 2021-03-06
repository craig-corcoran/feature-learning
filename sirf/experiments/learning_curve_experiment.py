import os
import csv
import time
import copy
import numpy
import pickle
import plac
import scipy.optimize
import scipy.sparse as sp
import theano
import matplotlib.pyplot as plt

import condor
import sirf
import sirf.grid_world as grid_world
import sirf.util as util

from sirf.grid_world import ValueGreedyPolicy
from itertools import izip
from sirf.rl import Model, calc_discount_horizon
from sirf.plotting import *
from sirf.encoding import TabularFeatures, TileFeatures
from sirf.bellman_basis import plot_features, BellmanBasis
from sirf.aggregate import out_string, reorder_columns
from sirf.util import openz

# mark best theta
# init with datapoints

# vary: lambda, req_rew, encoding, nonlin, size, reg, shift, sample init, 
#       convergence params, col normalization

theano.gof.compilelock.set_lock_status(False)
theano.config.on_unused_input = 'ignore'
theano.config.warn.sum_div_dimshuffle_bug = False

logger = sirf.get_logger(__name__)

@plac.annotations(
    workers=('number of condor workers', 'option', None, int),
    k=('number of features', 'option', None, int),
    encoding=('feature encoding used (tabular, tiled)', 'option', None, str),
    env_size=('size of the grid world', 'option', 's', int),
    n_runs=('number of runs to average performance over', 'option', None, int),
    n_reward_samples=('number of samples taken during reward harvesting evalutation', 'option', None, int),
    lam=('lambda for TD-lambda training', 'option', None, float),
    gam=('discount factor', 'option', None, float),
    beta=('covariance loss parameter', 'option', None, float),
    alpha=('extra multiplier on reconstruction cost of rewards', 'option', None, float),
    eta=('coefficient for laplacian regularization', 'option', None, float),
    eps=('precision for computing TD-lambda horizon', 'option', None, float),
    patience=('train until patience runs out', 'option', None, int),
    max_iter=('train for at most this many iterations', 'option', 'i', int),
    l1theta=('regularize theta with this L1 parameter', 'option', None, float),
    l1code=('regularize feature code with this L1 parameter', 'option', None, float),
    l2code=('regularize feature code with this L2 parameter', 'option', None, float),
    n_samples=('sample this many state transitions', 'option', None, str),
    nonlin=('feature nonlinearity', 'option', None, str, ['sigmoid', 'relu']),
    nonzero=('penalty for zero theta vectors', 'option', None, float),
    training_methods=('list of tuples of loss fn list and wrt params list', 'option'),
    min_imp=('train until loss improvement percentage is less than this', 'option', None, float),
    min_delta=('train until change in parameters is less than this', 'option', None, float),
    fldir=('feature-learning directory that has sirf/ and output/ in it', 'option', None, str),
    req_rew=('Boolean switch to require nonzero reward to be in the sample set when sampling', 'option', None, str),
    record_runs=('Boolean switch for recording learning curve plots and pickles at the end of each run', 'option', None, str),
    )
def main(workers = 0,
         k = 36,
         encoding = 'tile',
         env_size = 9,
         n_runs = 1,
         n_reward_samples = 2500,
         n_reward_runs = 25,
         lam = 0.,
         gam = 0.995,
         beta = 0.995,
         alpha = 1.,
         eta = 0.5,
         eps = 1e-5, 
         patience = 16,
         max_iter = 8, 
         l1theta = None,
         l1code = None,
         l2code = None,
         n_samples = None,
         nonlin = None,
         nonzero = None,
         training_methods = None,
         min_imp = 0.0001,
         min_delta = 1e-6,
         fldir = '/scratch/cluster/ccor/feature-learning/',
         req_rew = False,
         record_runs = False,
         ):

    if n_samples:
        n_samples = map(int, n_samples.split(','))

    beta_ratio = beta/gam 
    # append reward to basis when using perfect info?
    if training_methods is None:
        training_methods = [
            #(['covariance', 'prediction', 'value_prediction', 'bellman'],[['theta-all'],['theta-all'],['theta-all'],['theta-all','w']]),
            #(['prediction', 'value_prediction', 'bellman'],[['theta-all'],['theta-all'],['theta-all','w']]),
            #(['value_prediction'],[['theta-all']]),
            #(['value_prediction', 'bellman'],[['theta-all'],['theta-all','w']]),
            #(['prediction'],[['theta-all']]),
            #(['prediction', 'bellman'], [['theta-all'],['theta-all','w']]),
            #(['covariance'], [['theta-all']]),
            (['prediction', 'bellman'], [['theta-all'],['theta-all','w']]),
            (['bellman', 'prediction', 'bellman'], [['theta-all','w'], ['theta-all'],['theta-all','w']]),
            (['full_covariance', 'bellman'], [['theta-all'],['theta-all','w']]),
            (['covariance', 'bellman'], [['theta-all'],['theta-all','w']]),
            (['full_laplacian'], [['theta-all', 'w']]), 
            (['laplacian'], [['theta-all', 'w']]), 
            (['bellman'], [['theta-all', 'w']]), # baseline
            ]  

    losses = ['sample-reward', 'test-lsbellman', 'test-bellman', 'test-reward',  'test-model', 'test-fullmodel', # test-training
              'true-policy', 'true-policy-uniform', 'true-bellman', 'true-lsbellman', 'true-reward', 'true-model', 'true-fullmodel', 'true-lsq'] \
                if n_samples else \
             ['sample-reward', 'true-policy-uniform', 'true-policy', 'true-bellman',  'true-lsbellman', 'true-reward', 'true-model', 'true-fullmodel', 'true-lsq'] 

    logger.info('building environment of size %i' % env_size)
    mdp = grid_world.MDP(walls_on = True, size = env_size)
    env = mdp.env
    n_states = env_size**2
    m = Model(mdp.R, mdp.P, gam = gam)
    
    # create raw data encoder (constant appended in encoder by default)
    if encoding is 'tabular':
        encoder = TabularFeatures(env_size, append_const = True)
    elif encoding is 'tile':
        encoder = TileFeatures(env_size, append_const = True)
    elif encoding is 'factored':
        raise NotImplementedError

    def sample(n):
        logger.info('sampling from a grid world')
        # currently defaults to on-policy sampling
        
        n_extra = calc_discount_horizon(lam, gam, eps) - 1 # mdp returns n+1 states and n rewards
        kw = dict(n_samples = n + n_extra, encoder = encoder, req_rew = req_rew) 
        R, X, _ = mdp.sample_encoding(**kw)
        
        if req_rew:
            logger.info('reward required')
            assert sum(R.todense()) > 0

        logger.info('reward sum: %.2f' % sum(R.todense()))

        R_val, X_val, _ = mdp.sample_encoding(**kw)
        R_test, X_test, _ = mdp.sample_encoding(**kw)
        #losses = ['test-bellman', 'test-reward',  'test-model', 
                #'true-bellman', 'true-reward', 'true-model', 'true-lsq'] # test-training
        weighting = 'policy'

        return (X, X_val, X_test), (R, R_val, R_test), weighting
       
    def full_info():
        logger.info('using perfect information')
        # gen stacked matrices of I, P, P^2, ...
        R = numpy.array([])
        S = sp.eye(n_states, n_states)
        P = sp.eye(n_states, n_states)
        for i in xrange(calc_discount_horizon(lam, gam, eps)): # decay epsilon 
            R = numpy.append(R, P * m.R)
            P = m.P * P
            S = sp.vstack((S, P))
        
        X = encoder.encode(S)   
        R = sp.csr_matrix(R[:,None])
        X_val = X_test = X
        R_val = R_test = R
        #losses =  ['true-bellman', 'true-reward', 'true-model'] 
        weighting = 'uniform'

        return (X, X_val, X_test), (R, R_val, R_test), weighting

    
    reg = None
    if l1theta is not None:
        reg = ('l1theta', l1theta)
    if l1code is not None:
        reg = ('l1code', l1code)
    if l2code is not None:
        reg = ('l2code', l2code)

    run_param_keys = ['k','method','encoding','samples', 'reward_samples', 'reward_runs', 'size','weighting',
                      'lambda','gamma','alpha', 'eta', 'regularization','nonlinear']
    def yield_jobs(): 
        
        for i,n in enumerate(n_samples or [n_states]):
            
            logger.info('creating job with %i samples/states' % n)
            
            # build bellman operator matrices
            logger.info('making mixing matrices')
            Mphi, Mrew = BellmanBasis.get_mixing_matrices(n, lam, gam, 
                                    sampled = bool(n_samples), eps = eps)
            
            for r in xrange(n_runs):

                n_features = encoder.n_features
                # initialize parameters
                theta_init = numpy.random.standard_normal((n_features, k))
                theta_init /= numpy.sqrt((theta_init * theta_init).sum(axis=0))
                w_init = numpy.random.standard_normal((k+1,1)) 
                w_init = w_init / numpy.linalg.norm(w_init)
                

                # sample or gather full info data
                X_data, R_data, weighting = sample(n) if n_samples else full_info()

                bb_params = [n_features, [k], beta_ratio]
                bb_dict = dict( alpha = alpha, reg_tuple = reg, nonlin = nonlin,
                                nonzero = nonzero, thetas = [theta_init])
        
                for j, tm in enumerate(training_methods):
                    loss_list, wrt_list = tm
                    assert len(loss_list) == len(wrt_list)
                    
                    run_param_values = [k, tm, encoder, n, 
                                n_reward_samples, n_reward_runs, 
                                env_size, weighting, 
                                lam, gam, alpha, eta, 
                                reg[0]+str(reg[1]) if reg else 'None',
                                nonlin if nonlin else 'None']

                    d_run_params = dict(izip(run_param_keys, run_param_values))
                     
                    yield (train_basis,[d_run_params, bb_params, bb_dict,
                                        env, m, losses, # environment, model and loss list
                                        X_data, R_data, Mphi, Mrew, # training data
                                        max_iter, patience, min_imp, min_delta, # optimization params 
                                        fldir, record_runs]) # recording params
    # create output file path
    date_str = time.strftime('%y%m%d.%X').replace(':','')
    out_dir = fldir + 'sirf/output/csv/'
    root =  '%s.%s_results' % (
               date_str, 
               'n_samples' if n_samples else 'full_info')
    
    

    d_experiment_params = dict(izip(['k','encoding','size',
                      'lambda','gamma','alpha','regularization','nl'], 
                      [k, encoder, env_size, lam, gam, alpha, 
                              reg[0]+str(reg[1]) if reg else 'None',
                              nonlin if nonlin else 'None']))
    save_path = out_string(out_dir, root, d_experiment_params, '.csv.gz')
    logger.info('saving results to %s' % save_path)
    
    # get column title list ordered params | losses using dummy dicts
    d_param = dict(izip(run_param_keys, numpy.zeros(len(run_param_keys))))
    d_loss = dict(izip(losses, numpy.zeros(len(run_param_keys))))
    col_keys_array,_ = reorder_columns(d_param, d_loss)

    with openz(save_path, "wb") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(col_keys_array)

        for (_, out) in condor.do(yield_jobs(), workers):
            keys, vals = out
            assert (keys == col_keys_array).all() # todo catch
            writer.writerow(vals)

def sample_policy_reward(env, v, n_samples_perrun, n_runs):
        logger.info('performing reward evalutation on value greedy policy') 
        curr_policy = ValueGreedyPolicy(env, v)
        mmdp = grid_world.MDP(env, curr_policy, walls_on = True)
        total = 0
        for i in xrange(n_runs):
            env.state = env.random_state(seed = i)
            r, _, _ = mmdp.sample_policy(n_samples_perrun)
            total += numpy.sum(r)
        return total

def test_sample_reward():
    mdp = grid_world.MDP(walls_on = True)
    env = mdp.env
    m = Model(mdp.R, mdp.P, gam = gam)
    mdp = grid_world.MDP(walls_on = True, size = env_size, 
            policy = grid_world.OptimalPolicy(env, m))
    

def train_basis(d_run_params, basis_params, basis_dict, 
                env, model, losses, 
                S_data, R_data, Mphi, Mrew, 
                max_iter, patience, min_imp, min_delta, 
                fl_dir, record_runs):

    method, weighting, encoder, n_reward_samples, n_reward_runs, env_size = map(lambda x: d_run_params[x], 
                                      'method weighting encoding reward_samples reward_runs size'.split())
    logger.info('training basis using training method: %s' % str(method))

    S, S_val, S_test = S_data
    R, R_val, R_test = R_data

    n_rows = float(S.shape[0])

    # initialize loss dictionary
    d_loss_learning = {}
    for key in losses:
        d_loss_learning[key] = numpy.array([])
     
    loss_list, wrt_list = method
    assert len(loss_list) == len(wrt_list)

    logger.info('constructing bellman basis')
    basis = BellmanBasis(*basis_params, **basis_dict)
    
    def record_loss(d_loss):
        # TODO automate/shorten this
        # record losses with test set
        for loss, arr in d_loss.items():
            if loss == 'sample-reward':
                val = sample_policy_reward(env, basis.estimated_value(encoder.B, False), n_reward_samples, n_reward_runs)
            elif loss == 'test-training':
                val = basis.loss(basis.flat_params, S_test, R_test, Mphi, Mrew) / n_rows
            elif loss == 'test-bellman':
                val = basis.loss_be(*(basis.params + [S_test, R_test, Mphi, Mrew])) / n_rows
            elif loss == 'test-lsbellman':
                val = basis.loss_lsbe(*(basis.params + [S_test, R_test, Mphi, Mrew])) / n_rows
            elif loss == 'test-reward':
                val = basis.loss_r(*(basis.params + [S_test, R_test, Mphi, Mrew])) / n_rows
            elif loss == 'test-model':
                val = basis.loss_m(*(basis.params + [S_test, R_test, Mphi, Mrew])) / n_rows
            elif loss == 'test-fullmodel':
                val = basis.loss_fm(*(basis.params + [S_test, R_test, Mphi, Mrew])) / n_rows
            elif loss == 'true-policy':
                val = model.policy_distance(Bs, weighting = 'policy')
            elif loss == 'true-policy-uniform':
                val = model.policy_distance(Bs, weighting = 'uniform')
            elif loss == 'true-bellman':
                val = model.bellman_error(Bs, w = basis.params[-1], weighting = weighting)
            elif loss == 'true-lsbellman':
                val = model.bellman_error(Bs, weighting = weighting)
            elif loss == 'true-reward':
                val = model.reward_error(Bs, weighting = weighting)
            elif loss == 'true-model':
                val = model.model_error(Bs, weighting = weighting)
            elif loss == 'true-fullmodel':
                val = model.fullmodel_error(Bs, weighting = weighting)
            elif loss == 'true-lsq':
                val = model.lsq_error(Bs, weighting = weighting)
            else: print loss; assert False
            
            d_loss[loss] = numpy.append(arr, val)
        return d_loss
    
    switch = [] # list of indices where a training method switch occurred
    it = 0
    
    # train once on w to initialize
    basis.set_loss('bellman', ['w'])
    basis.set_params(scipy.optimize.fmin_cg(
            basis.loss, basis.flat_params, basis.grad,
            args = (S, R, Mphi, Mrew),
            full_output = False,
            maxiter = max_iter,
            ))
    
    # TODO keep?
    IM = encoder.weights_to_basis(basis.thetas[-1])
    Bs = basis.encode(encoder.B, False)
    if len(basis.thetas) == 1:
        assert (IM == Bs).all()
    d_loss_learning = record_loss(d_loss_learning)

    for loss, wrt in zip(loss_list, wrt_list):
        
        waiting = 0
        best_params = None
        best_test_loss = 1e20
        
        if 'w' in wrt_list: # initialize w to the lstd soln given the current basis
            logger.info('initializing w to lstd soln')
            basis.params[-1] = BellmanBasis.lstd_weights(basis.encode(S), R, Mphi, Mrew) # TODO change to iteration of opt on w?
        
        try:
            while (waiting < patience):
                it += 1
                logger.info('*** iteration ' + str(it) + '***')
                
                old_params = copy.deepcopy(basis.flat_params)
                for loss_, wrt_ in ((loss, wrt), ('bellman', ['w'])):
                    basis.set_loss(loss_, wrt_)
                    basis.set_params(scipy.optimize.fmin_cg(
                            basis.loss, basis.flat_params, basis.grad,
                            args = (S, R, Mphi, Mrew),
                            full_output = False,
                            maxiter = max_iter,
                            ))
                basis.set_loss(loss, wrt) # reset loss back from bellman
                 
                delta = numpy.linalg.norm(old_params-basis.flat_params)
                logger.info('delta theta: %.2f' % delta)
                
                norms = numpy.apply_along_axis(numpy.linalg.norm, 0, basis.thetas[0])
                logger.info( 'column norms: %.2f min / %.2f avg / %.2f max' % (
                    norms.min(), norms.mean(), norms.max()))
                
                err = basis.loss(basis.flat_params, S_val, R_val, Mphi, Mrew)
                
                if err < best_test_loss:
                    
                    if ((best_test_loss - err) / best_test_loss > min_imp) and (delta > min_delta):
                        waiting = 0
                    else:
                        waiting += 1
                        logger.info('iters without better %s loss: %i' % (basis.loss_type, int(waiting)))

                    best_test_loss = err
                    best_params = copy.deepcopy(basis.flat_params)
                    logger.info('new best %s loss: %.2f' % (basis.loss_type, best_test_loss))
                    
                else:
                    waiting += 1
                    logger.info('iters without better %s loss: %i' % (basis.loss_type, int(waiting)))

                Bs = basis.encode(encoder.B, False)
                d_loss_learning = record_loss(d_loss_learning)

        except KeyboardInterrupt:
            logger.info( '\n user stopped current training loop')
        
        # set params to best params from last loss
        basis.set_params(vec = best_params)
        switch.append(it-1)
    
    sparse_eps = 1e-5
    Bs = basis.encode(encoder.B, False)
    d_loss_learning = record_loss(d_loss_learning)
    logger.info( 'final test bellman error: %.2f' % model.bellman_error(Bs, weighting = weighting))
    logger.info( 'final sparsity: ' + str( [(numpy.sum(abs(th) < sparse_eps) / float(len(th.flatten()))) for th in basis.params]))

    # edit d_run_params to not include wrt list in method
    d_run_params['method'] = '-'.join(d_run_params['method'][0])
    
    # TODO change to output log file and plot afterwards
    if record_runs:
        
        # save results!
        # plot basis functions
        plot_stacked_features(Bs[:, :36])
        figst = out_string(fl_dir+'sirf/output/plots/learning/', 'basis_stacked', d_run_params, '.pdf')
        plt.savefig(figst)

        # plot the basis functions again!
        plot_features(Bs)
        figst = out_string(fl_dir+'sirf/output/plots/learning/', 'basis_all', d_run_params, '.pdf')
        plt.savefig(figst)
        
        # plot learning curves
        pltd = plot_learning_curves(d_loss_learning, switch, filt = 'test')
        if pltd: # if we actually plotted
            plt.savefig(out_string(fl_dir+'sirf/output/plots/learning/', 'test_loss', d_run_params, '.pdf') )
        pltd = plot_learning_curves(d_loss_learning, switch, filt = 'true')
        if pltd:
            plt.savefig(out_string(fl_dir+'sirf/output/plots/learning/', 'true_loss', d_run_params, '.pdf'))        
        
        # plot value functions
        plot_value_functions(env_size, model, Bs)
        plt.savefig(out_string(fl_dir+'sirf/output/plots/learning/', 'value_funcs', d_run_params, '.pdf'))

        # plot spectrum of reward and features
        gen_spectrum(Bs, model.P, model.R)
        plt.savefig(out_string(fl_dir+'sirf/output/plots/learning/', 'spectrum', d_run_params, '.pdf'))
    
    d_loss_batch = dict(izip(d_loss_learning.keys(), map(lambda x: x[-1], 
                                                    d_loss_learning.values())))
    # returns keys_array, values_array
    return reorder_columns(d_run_params, d_loss_batch) 

if __name__ == '__main__':
    sirf.script(main)
