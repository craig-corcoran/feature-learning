import os
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
from itertools import izip
from sirf.rl import Model
from sirf.encoding import TabularFeatures, TileFeatures
from sirf.bellman_basis import plot_features, BellmanBasis 

# mark  best theta
# init with datapoints
# on policy learning with perfect info?
# initializing w to w* or faster gradient inner loop
# running until long convergence

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
    lam=('lambda for TD-lambda training', 'option', None, float),
    gam=('discount factor', 'option', None, float),
    beta=('covariance loss parameter', 'option', None, float),
    alpha=('extra multiplier on reconstruction cost of rewards', 'option', None, float),
    eps=('epsilon for computing TD-lambda horizon', 'option', None, float),
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
    fldir=('feature-learning directory that has sirf/output in it', 'option', None, str),
    movie=('Boolean switch for recording movie of basis functions during learning', 'option', None, str),
    req_rew=('Boolean switch to require nonzero reward to be in the sample set when sampling', 'option', None, str),
    record_runs=('Boolean switch for recording learning curve plots and pickles at the end of each run', 'option', None, str),
    )
def main(workers = 0,
         k = 36,
         encoding = 'tile',
         env_size = 9,
         n_runs = 1,
         lam = 0.,
         gam = 0.995,
         beta = 0.995,
         alpha = 1.,
         eps = 1e-5, 
         patience = 15,
         max_iter = 8, 
         l1theta = None,
         l1code = 0.0002,
         l2code = None,
         n_samples = None,
         nonlin = None,
         nonzero = None,
         training_methods = None,
         min_imp = 0.0002,
         min_delta = 1e-6,
         fldir = '/scratch/cluster/ccor/feature-learning/',
         movie = False,
         req_rew = True,
         record_runs = False,
         ):

    if n_samples:
        n_samples = map(int, n_samples.split(','))

    beta_ratio = beta/gam 
    # append reward to basis when using perfect info?
    if training_methods is None:
        training_methods = [
            (['prediction'],[['theta-all']]),
            (['prediction', 'layered'], [['theta-all'],['theta-all','w']]),
            (['layered'], [['theta-all', 'w']]) # baseline
            ]  

    losses = ['test-bellman', 'test-reward',  'test-model', 'true-bellman',
                'true-reward', 'true-model', 'true-lsq'] if n_samples else \
             ['true-bellman', 'true-reward', 'true-model'] 

    logger.info('building environment of size %i' % env_size)
    mdp = grid_world.MDP(walls_on = True, size = env_size)
    n_states = env_size**2

    m = Model(mdp.env.R, mdp.env.P, gam = gam)
    
    # constant appended in encoder by default
    if encoding is 'tabular':
        encoder = TabularFeatures(env_size, append_const = True)
    elif encoding is 'tile':
        encoder = TileFeatures(env_size, append_const = True)
    elif encoding is 'factored':
        raise NotImplementedError

    def sample(n):
        logger.info('sampling from a grid world')
        # currently defaults to on-policy sampling
        
        kw = dict(n_samples = n, encoder = encoder, req_rew = req_rew) 
        R, X, _ = mdp.sample_encoding(**kw)
        
        if req_rew:
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
        for i in xrange(BellmanBasis._calc_n_steps(lam, gam, eps)): # decay epsilon 
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
    

    logger.info('constructing basis')
    reg = None
    if l1theta is not None:
        reg = ('l1theta', l1theta)
    if l1code is not None:
        reg = ('l1code', l1code)
    if l2code is not None:
        reg = ('l2code', l2code)

    # initialize loss dictionary with numpy array for each run, method, and number of samples
    d_loss_data = {}
    for key in losses:
        d_loss_data[key] = numpy.zeros((len(n_samples), n_runs, len(training_methods)))
    
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
                theta_init = 1e-2*numpy.random.standard_normal((n_features, k))
                #theta_init /= numpy.sqrt((theta_init * theta_init).sum(axis=0))
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
                    
                    # generate output string with relevant parameters
                    out_string = '%s.k=%i.reg=%s.a=%s.lam=%s.gam=%s.%s.%s%s.' % (
                        str(tm),
                        k,
                        str(reg) if reg is None else reg[0] + str(reg[1]),
                        str(alpha),
                        lam, gam, weighting, #'+'.join(losses),
                        '.samples=%d' % n if n_samples else '',
                        '.nonlin=%s' % nonlin if nonlin else '')
                    
                    yield (train_basis, [(i,r,j), bb_params, bb_dict, tm, m, losses, encoder,
                        X_data, R_data, Mphi, Mrew, patience, max_iter, weighting, 
                        out_string, min_imp, min_delta, env_size, fldir, movie, record_runs])

    # aggregate the condor data
    for (_, result) in condor.do(yield_jobs(), workers):
            d_batch_loss, ind_tuple = result
            for name in d_batch_loss.keys():
                d_loss_data[name][ind_tuple] = d_batch_loss[name]

    def out_string(pref, root, suff = ''):
        return '%s%s.k=%i.reg=%s.a=%s.lam=%s.gam=%s%s%s%s' % (
        pref,
        root,
        k,
        str(reg) if reg is None else reg[0] + str(reg[1]),
        str(alpha),
        lam, gam, #'+'.join(losses),
        '.samples=%s' % n_samples if n_samples else '',
        '.nonlin=%s' % nonlin if nonlin else '',
        suff)

    # save results!
    out_dir = fldir + 'sirf/output/'
    pickle_path = out_string(out_dir, 'pickle/%s_results' % 
                    ('n_samples' if n_samples else 'full_info'), '.pickle.gz')

    logger.info('saving results to %s' % pickle_path)
    with util.openz(pickle_path, "wb") as out_file:
        pickle.dump(d_loss_data, out_file, protocol = -1)

    labels = map(lambda x: str(x[0]), training_methods)
    plot_aggregate_data(n_samples, d_loss_data, labels)
    plot_path = out_string(out_dir, 'plots/%s_results' % 
                            ('n_samples' if n_samples else 'full_info'), '.pdf')
    plt.savefig(plot_path) 

def plot_aggregate_data(n_samples, d_loss_data, labels):

    x = numpy.array(n_samples, dtype = numpy.float64) if n_samples else range(len(n_samples))
    f = plt.figure()
    logger.info('plotting aggregate run performance data')

    num = len(d_loss_data)
    cols = numpy.ceil(numpy.sqrt(num))
    rows = numpy.ceil(num/cols)
    for i,(key,mat) in enumerate(d_loss_data.items()):

        ax = f.add_subplot(rows,cols,i+1) 
        
        for h,lb in enumerate(labels):                

            std = numpy.std(mat[:,:,h], axis=1)
            ste = std / numpy.sqrt(x)
            mn = numpy.mean(mat[:,:,h], axis=1)
            
            ax.fill_between(x, mn-ste, mn+ste, alpha=0.15, linewidth = 0)
            ax.plot(x, mn, label = lb)
            plt.title(key)
            plt.axis('off')
            #plt.legend(loc = 3) # lower left


def train_basis(ind_tuple, basis_params, basis_dict, method, model, losses, 
            encoder, S_data, R_data, Mphi, Mrew, patience, max_iter, weighting, 
            out_string, min_imp, min_delta, env_size, fl_dir, movie, record_runs):

    logger.info('training basis using training method: %s' % str(method))

    S, S_val, S_test = S_data
    R, R_val, R_test = R_data

    # initialize loss dictionary
    d_loss_learning = {}
    for key in losses:
        d_loss_learning[key] = numpy.array([])
    
    if movie:
        logger.info('clearing movie directory of pngs')
        movie_path = fl_dir + 'sirf/output/plots/movie/'
        os.system("rm %s*.png" % (movie_path)) 
    
    loss_list, wrt_list = method
    assert len(loss_list) == len(wrt_list)
    basis = BellmanBasis(*basis_params, **basis_dict)
    
    def record_loss(d_loss):

        # record losses with test set
        for loss, arr in d_loss.items():
            if loss == 'test-training':
                val = basis.loss(basis.flat_params, S_test, R_test, Mphi, Mrew)
            elif loss == 'test-bellman':
                val = basis.loss_be(*(basis.params + [S_test, R_test, Mphi, Mrew]))
            elif loss == 'test-reward':
                val = basis.loss_r(*(basis.params + [S_test, R_test, Mphi, Mrew]))
            elif loss == 'test-model':
                val = basis.loss_m(*(basis.params + [S_test, R_test, Mphi, Mrew]))
            elif loss == 'true-bellman':
                val = model.bellman_error(IM, weighting = weighting)
            elif loss == 'true-reward':
                val = model.reward_error(IM, weighting = weighting)
            elif loss == 'true-model':
                val = model.model_error(IM, weighting = weighting)
            elif loss == 'true-lsq':
                val = model.value_error(IM, weighting = weighting)
            else: assert False

            d_loss[loss] = numpy.append(arr, val)
        return d_loss
    
    vmin = -0.25
    vmax = 0.25
    switch = [] # list of indices where a training method switch occurred
    it = 0
    IM = encoder.weights_to_basis(basis.thetas[-1])
    d_loss_learning = record_loss(d_loss_learning)
    for i,loss in enumerate(loss_list):
        
        basis.set_loss(loss, wrt_list[i])
    
        waiting = 0
        best_params = None
        best_test_loss = 1e20
        
        if movie:
            # save a blank frame before/between training losses
            plot_features(numpy.zeros_like(basis.thetas[-1][:-1]), vmin = vmin, vmax = vmax)
            plt.savefig(fl_dir + 'sirf/output/plots/movie/img_%03d.png' % it)
        
        if 'w' in wrt_list: # initialize w to the lstd soln given the current basis
            logger.info('initializing w to lstd soln')
            basis.params[-1] = BellmanBasis.lstd_weights(basis.encode(S), R, Mphi, Mrew)
        
        try:
            while (waiting < patience):
                it += 1
                logger.info('*** iteration ' + str(it) + '***')
                
                if movie:
                    # record learning movie frame
                    plot_features(basis.thetas[-1][:-1], vmin = vmin, vmax = vmax)
                    plt.savefig(fl_dir + 'sirf/output/plots/movie/img_%03d.png' % it)

                old_params = copy.deepcopy(basis.flat_params)
                basis.set_params( vec = scipy.optimize.fmin_cg(basis.loss, basis.flat_params, basis.grad,
                                  args = (S, R, Mphi, Mrew),
                                  full_output = False,
                                  maxiter = max_iter,
                                  gtol = 1e-8
                                  ) )
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

                # check reward loss gradient
                #print 'norm of reward prediction gradient: ', numpy.linalg.norm(
                    #basis.grad_rew_pred(basis.flat_params, S, R, Mphi, Mrew))
                IM = encoder.weights_to_basis(basis.thetas[-1])
                d_loss_learning = record_loss(d_loss_learning)


        except KeyboardInterrupt:
            logger.info( '\n user stopped current training loop')
        
        basis.set_params(vec = best_params)
        switch.append(it-1)
    
    sparse_eps = 1e-5
    IM = encoder.weights_to_basis(basis.thetas[-1])
    d_loss_learning = record_loss(d_loss_learning)
    logger.info( 'final test bellman error: %.2f' % model.bellman_error(IM, weighting = weighting))
    logger.info( 'final sparsity: ' + str( [(numpy.sum(abs(th) < sparse_eps) / float(len(th.flatten()))) for th in basis.params]))

    # store final errors in a batch loss dictionary
    d_loss_batch = dict(izip(d_loss_learning.keys(), map(lambda x: x[-1], d_loss_learning.values())))
    with util.openz('%ssirf/output/pickle/d_loss_batch%s%s%s' % (fl_dir, ind_tuple, out_string,'.pickle.gz'), "wb") as out_file:
            pickle.dump(d_loss_learning, out_file, protocol = -1)


    if record_runs:
        # save results!
        with util.openz(fl_dir + 'sirf/output/pickle/learning_curve_results' + out_string + '.pickle.gz', "wb") as out_file:
            pickle.dump(d_loss_learning, out_file, protocol = -1)

        # plot basis functions
        plot_features(IM[:, :36])
        plt.savefig(fl_dir + 'sirf/output/plots/basis' + out_string+ '.pdf')

        # plot the basis functions again!
        _plot_features(IM)
        plt.savefig(fl_dir + 'sirf/output/plots/basis0' + out_string + '.pdf')
        
        # plot learning curves
        pltd = plot_learning_curves(d_loss, switch, filt = 'test')
        if pltd:
            plt.savefig(fl_dir + 'sirf/output/plots/test_loss' + out_string + '.pdf')
        pltd = plot_learning_curves(d_loss, switch, filt = 'true')
        if pltd:
            plt.savefig(fl_dir + 'sirf/output/plots/true_loss' + out_string + '.pdf')    
        
        # plot value functions
        plot_value_functions(env_size, model, IM)
        plt.savefig(fl_dir + 'sirf/output/plots/value' + out_string + '.pdf')

        # plot spectrum of reward and features
        gen_spectrum(IM, model.P, model.R)
        plt.savefig(fl_dir + 'sirf/output/plots/spectrum' + out_string + '.pdf')

        # make movie from basis files saved # currenty not working on cs machines
        #if movie:
        #make_learning_movie(movie_path, out_string)

    return d_loss_batch, ind_tuple # basis

def gen_spectrum(Inp, P, R):
    plt.clf()
    Inp /= numpy.sqrt((Inp * Inp).sum(axis=0))
    w,v = numpy.linalg.eig( P )#- beta * numpy.eye(m.P.shape[0]))
    v = v[:, numpy.argsort(abs(w))[::-1]]
    v = numpy.real(v)
    mag = abs(numpy.dot(Inp.T, v))
    r_mag = abs(numpy.dot(R.T, v))
    
    order = numpy.argsort(r_mag)[::-1]
    mag = mag[:, order]
    r_mag = r_mag[:, order]

    #r_mag = r_mag[r_mag > 1e-10] 
    
    # plot feature spectrums
    x = range(len(w))
    for sp in mag:
        plt.semilogy(x, sp, '.')
    plt.semilogy(range(len(r_mag)), r_mag, 'ko')
    plt.ylim(1e-6, 1)

def make_learning_movie(movie_path, out_string):
    
    logger.info('Making movie animation.mpg - this make take a while')
    mod_out_str = out_string.replace('(','').replace(')','').replace('[','').replace(']','')
    cmd = "ffmpeg -qscale 2 -r 8 -b 10M  -i %simg_%s.png  %slearning_mov.%s.mp4" % (movie_path, '%03d', movie_path, mod_out_str)
    os.system(cmd)
    #os.system("ffmpeg -qscale 2 -r 4 -b 10M  -i %simg_%03d.png  %slearning_mov.%s.mp4" % (movie_path, movie_path, out_string))
    #os.system("mencoder %s -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation.mpg" % path)

def plot_learning_curves(d_loss, switch, filt = '', ylim = None, mean_norm = False):
    plt.clf()
    ax = plt.axes()
    plotted = False
    for name, curve in d_loss.items():
        if filt in name:
            plotted = True
            x = range(len(curve))
            ax.plot(x, curve, label = name)
    
    if len(switch) > 1:
        for i in xrange(len(switch)-1):
            ax.plot([switch[i], switch[i]], [0, plt.gca().get_ylim()[1]], 'k--', label = 'training switch')
    
    if plotted:
        plt.title('Losses per CG Minibatch')
        if ylim is not None:
            plt.ylim(ylim)
        ax.legend() #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return plotted

def _plot_features(phi, r = None, c = None, vmin = None, vmax = None):
    plt.clf()
    j,k = phi.shape
    if r is None:
        r = c = numpy.round(numpy.sqrt(j))
        assert r*c == j
        
    m = numpy.floor(numpy.sqrt(k))
    n = numpy.ceil(k/float(m))
    assert m*n >= k 
    
    f = plt.figure()
    for i in xrange(k):
            
        ax = f.add_subplot(m,n,i+1)
        im = numpy.reshape(phi[:,i], (r,c))
        ax.imshow(im, cmap = 'RdBu', interpolation = 'nearest', vmin = vmin, vmax = vmax)
        ax.set_xticks([])
        ax.set_yticks([])

def plot_value_functions(size, m, IM):
    # TODO append bias when solving for value function
    # plot value functions, true and approx
    plt.clf()
    f = plt.figure()
    
    # true model value fn
    ax = f.add_subplot(311)
    ax.imshow(numpy.reshape(m.V, (size, size)), cmap = 'gray', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    # bellman error estimate (using true model)
    ax = f.add_subplot(312)
    w_be = m.get_lstd_weights(IM) # todo add lambda parameter here
    v_be = numpy.dot(IM, w_be)
    ax.imshow(numpy.reshape(v_be, (size, size)), cmap = 'gray', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # least squares solution with true value function
    ax = f.add_subplot(313)
    w_lsq = numpy.linalg.lstsq(IM, m.V)[0]
    v_lsq = numpy.dot(IM, w_lsq)
    ax.imshow(numpy.reshape(v_lsq, (size, size)), cmap = 'gray', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    logger.info('bellman error norm from v: %.2f' % numpy.linalg.norm(m.V - v_be))
    logger.info('lsq error norm from v: %.2f' % numpy.linalg.norm(m.V - v_lsq))

if __name__ == '__main__':
    sirf.script(main)
