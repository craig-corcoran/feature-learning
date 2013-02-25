import copy
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
    lam=('lambda for TD-lambda training', 'option', None, float),
    gam=('discount factor', 'option', None, float),
    beta=('covariance loss parameter', 'option', None, float),
    alpha=('extra multiplier on reconstruction cost of rewards', 'option', None, float),
    eps=('epsilon for computing TD-lambda horizon', 'option', None, float),
    patience=('train until patience runs out', 'option', None, int),
    max_iter=('train for at most this many iterations', 'option', 'i', int),
    weighting=('method for sampling from grid world', 'option', None, str, ['policy', 'uniform']),
    l1theta=('regularize theta with this L1 parameter', 'option', None, float),
    l1code=('regularize feature code with this L1 parameter', 'option', None, float),
    l2code=('regularize feature code with this L2 parameter', 'option', None, float),
    state_rep=('represent world states this way', 'option', None, str, ['tabular', 'factored']),
    n_samples=('sample this many state transitions', 'option', None, int),
    nonlin=('feature nonlinearity', 'option', None, str, ['sigmoid', 'relu']),
    nonzero=('penalty for zero theta vectors', 'option', None, float),
    training_methods=('list of tuples of loss fn list and wrt params list', 'option'),
    min_imp=('train until loss improvement percentage is less than this', 'option', None, float),
    min_delta=('train until change in parameters is less than this', 'option', None, float),
    fl_dir=('feature-learning directory that has sirf/output in it', 'option', None, str)
    )
def main(workers = 0,
         k = 36,
         encoding = 'tabular',
         env_size = 15,
         n_runs = 1,
         lam = 0.,
         gam = 0.999,
         beta = 0.999,
         alpha = 1.,
         eps = 1e-5, 
         patience = 15,
         max_iter = 15,
         weighting = 'uniform', 
         l1theta = None,
         l1code = 0.0002,
         l2code = None,
         state_rep = 'tabular',
         n_samples = None,
         nonlin = None,
         nonzero = None,
         training_methods = None,
         min_imp = 0.0002,
         min_delta = 1e-6,
         fl_dir = '/scratch/cluster/ccor/feature-learning/',
         movie = False
         ):

    beta_ratio = beta/gam 
    # append reward to basis when using perfect info?
    if training_methods is None:
        training_methods = [
            (['prediction'],[['theta-all']]),
            (['prediction', 'layered'], [['theta-all'],['theta-all','w']]),
            #(['covariance'],[['theta-all']]), # with reward, without fine-tuning
            #(['covariance', 'layered'], [['theta-all'],['theta-all','w']]), 
            (['layered'], [['theta-all', 'w']]) # baseline
            ]    

    print 'building environment'
    mdp = grid_world.MDP(walls_on = True, size = env_size)
    n = env_size**2
    m = Model(mdp.env.R, mdp.env.P, gam = gam)

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
        X = scipy.sparse.hstack((X, numpy.ones((X.shape[0],1))))
        X_val = X_test = X
        R_val = R_test = R
        losses =  ['true-bellman', 'true-reward', 'true-model'] #, 'test-training'] #, 'true-lsq']

    # build bellman operator matrices
    print 'making mixing matrices'
    Mphi, Mrew = BellmanBasis.get_mixing_matrices(n_samples or n, lam, gam, sampled = bool(n_samples), eps = eps)

    print 'constructing basis'
    reg = None
    if l1theta is not None:
        reg = ('l1theta', l1theta)
    if l1code is not None:
        reg = ('l1code', l1code)
    if l2code is not None:
        reg = ('l2code', l2code)

    
    # initialize features sparsely and with unit norm
    theta_init = numpy.random.standard_normal((n+1, k))
    #sparsity = 0.5
    #for i in xrange(k):
        #z = numpy.random.random(n+1)
        #theta_init[:,i][z < sparsity] = 0.
    theta_init /= numpy.sqrt((theta_init * theta_init).sum(axis=0))
    #theta_init = X.todense()[numpy.round(X.shape[0]*numpy.random.random(k)).astype('int'),:].T # init with samples # :/ not so good?

    #w_init = numpy.random.standard_normal((k+1,1)) 
    #w_init = w_init / numpy.linalg.norm(w_init) 
    
    x = X.todense()
    vec = numpy.ones((x.shape[0],1))
    y = numpy.hstack((x, vec))

    bb_params = [n, [k], beta_ratio]
    bb_dict = dict( alpha = alpha, reg_tuple = reg, nonlin = nonlin,
        nonzero = nonzero, thetas = [theta_init])

    # initialize loss dictionary
    d_loss_data = {}
    for key in losses:
        d_loss_data[key] = numpy.array([])
    
    def yield_jobs():
        for tm in training_methods:
            loss_list, wrt_list = tm
            assert len(loss_list) == len(wrt_list)
            
            out_string = '%s.k=%i.reg=%s.a=%s.lam=%s.gam=%s.%s.%s%s.' % (
                str(tm),
                k,
                str(reg) if reg is None else reg[0] + str(reg[1]),
                str(alpha),
                lam, gam, weighting, #'+'.join(losses),
                '.samples=%d' % n_samples if n_samples else '',
                '.nonlin=%s' % nonlin if nonlin else '')

            yield (train_basis, [bb_params, bb_dict, tm, m, d_loss_data, X, R,  
                X_val, R_val, X_test, R_test, Mphi, Mrew, patience, max_iter, 
            weighting, out_string, min_imp, min_delta, env_size, fl_dir, movie])

    # launch condor jobs
    for _ in condor.do(yield_jobs(), workers):
        pass

def train_basis(basis_params, basis_dict, method, model, d_loss, S, R,  
            S_val, R_val, S_test, R_test, Mphi, Mrew, patience, max_iter, 
            weighting, out_string, min_imp, min_delta, env_size, fl_dir, movie):

    print 'training basis using training method: ', str(method)
    
    if movie:
        print 'clearing movie directory of pngs'
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
                val = basis.loss_be(basis.params, S_val, R_val, Mphi, Mrew)
            elif loss == 'test-reward':
                val = basis.loss_r(basis.params, S_val, R_val, Mphi, Mrew)
            elif loss == 'test-model':
                val = basis.loss_m(basis.params, S_val, R_val, Mphi, Mrew)
            elif loss == 'true-bellman':
                val = model.bellman_error(basis.thetas[-1][:-1], weighting = weighting)
            elif loss == 'true-reward':
                val = model.reward_error(basis.thetas[-1][:-1], weighting = weighting)
            elif loss == 'true-model':
                val = model.model_error(basis.thetas[-1][:-1], weighting = weighting)
            elif loss == 'true-lsq':
                val = model.value_error(basis.thetas[-1][:-1], weighting = weighting)
            else: assert False

            d_loss[loss] = numpy.append(arr, val)
        return d_loss
    
    vmin = -0.25
    vmax = 0.25
    switch = [] # list of indices where a training method switch occurred
    it = 0
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
            print 'initializing w to lstd soln'
            basis.params[-1] = BellmanBasis.lstd_weights(basis.encode(S), R, Mphi, Mrew)
        
        try:
            while (waiting < patience):
                it += 1
                print '*** iteration', it, '***'
                
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
                print 'delta theta: ', delta
                
                norms = numpy.apply_along_axis(numpy.linalg.norm, 0, basis.thetas[0])
                print 'column norms: %.2f min / %.2f avg / %.2f max' % (
                    norms.min(), norms.mean(), norms.max())
                
                err = basis.loss(basis.flat_params, S_val, R_val, Mphi, Mrew)
                
                if err < best_test_loss:
                    
                    if ((best_test_loss - err) / best_test_loss > min_imp) and (delta > min_delta):
                        waiting = 0
                    else:
                        waiting += 1
                        print 'iters without better %s loss: ' % basis.loss_type, waiting

                   
                    best_test_loss = err
                    best_params = copy.deepcopy(basis.flat_params)
                    print 'new best %s loss: ' % basis.loss_type, best_test_loss
                    
                else:
                    waiting += 1
                    print 'iters without better %s loss: ' % basis.loss_type, waiting

                # check reward loss gradient
                print 'norm of reward prediction gradient: ', numpy.linalg.norm(
                    basis.grad_rew_pred(basis.flat_params, S, R, Mphi, Mrew))

                d_loss = record_loss(d_loss)


        except KeyboardInterrupt:
            print '\n user stopped current training loop'
        
        basis.set_params(vec = best_params)
        switch.append(it-1)
    
    sparse_eps = 1e-4
    print 'final test bellman error: ', model.bellman_error(basis.thetas[-1][:-1], weighting = weighting)
    print 'final sparsity: ', [sum(abs(th) < sparse_eps) / float(len(th.flatten())) for th in basis.params]

    # save results!
    with util.openz(fl_dir + 'sirf/output/pickle/learning_curve_results' + out_string + 'pickle.gz', "wb") as out_file:
        pickle.dump(d_loss, out_file, protocol = -1)

    # plot basis functions
    plot_features(basis.thetas[-1][:-1, :36])
    plt.savefig(fl_dir + 'sirf/output/plots/basis' + out_string+ '.pdf')
    
    # plot learning curves
    plot_learning_curves(d_loss, switch)
    plt.savefig(fl_dir + 'sirf/output/plots/loss' + out_string + '.pdf')
    
    # plot value functions
    plot_value_functions(env_size, model, basis)
    plt.savefig(fl_dir + 'sirf/output/plots/value' + out_string + '.pdf')

    # plot the basis functions again!
    _plot_features(basis.thetas[-1][:-1, :36])
    plt.savefig(fl_dir + 'sirf/output/plots/basis0' + out_string + '.pdf')

    # plot spectrum of reward and features
    gen_spectrum(IM, model.P, model.R)

    # make movie from basis files saved
    #make_learning_movie(movie_path, out_string)

    #return basis, d_loss

def gen_spectrum(Inp, P, R):
    Inp /= numpy.sqrt((Inp * Inp).sum(axis=0))
    w,v = numpy.linalg.eig( P )#- beta * numpy.eye(m.P.shape[0]))
    v = v[:, numpy.argsort(abs(w))]
    v = numpy.real(v)
    mag = abs(numpy.dot(Inp.T, v))
    r_mag = abs(numpy.dot(R.T, v))
    
    # plot feature spectrums
    x = range(len(w))
    for sp in mag:
        plt.plot(x, sp, 'o')
    plt.plot(x, r_mag, 'ko')


def make_learning_movie(movie_path, out_string):
    
    print 'Making movie animation.mpg - this make take a while'
    mod_out_str = out_string.replace('(','').replace(')','').replace('[','').replace(']','')
    cmd = "ffmpeg -qscale 2 -r 8 -b 10M  -i %simg_%s.png  %slearning_mov.%s.mp4" % (movie_path, '%03d', movie_path, mod_out_str)
    print 'command: ', cmd
    os.system(cmd)
    #os.system("ffmpeg -qscale 2 -r 4 -b 10M  -i %simg_%03d.png  %slearning_mov.%s.mp4" % (movie_path, movie_path, out_string))
    #os.system("mencoder %s -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation.mpg" % path)

def plot_learning_curves(d_loss, switch):
    plt.clf()
    ax = plt.axes()
    #mx = 0
    for name, curve in d_loss.items():
        print 'plotting %s curve' % name
        x = range(len(curve))
        #ax.semilogy(x, curve, label = name)
        if name == 'test-training':        
            ax.plot(x, curve/curve.mean(), label = name)
        else:
            ax.plot(x, curve, label = name)
       # mx = max(mx, numpy.max(curve))
     
    if len(switch) > 1:
        for i in xrange(len(switch)-1):
            ax.plot([switch[i], switch[i]], [0, 1], 'k--', label = 'training switch')
    plt.title('Losses per CG Minibatch')
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.ylim(0, 2.)
    ax.legend()

#def _plot_features(phi, r = None, c = None):
    #plt.clf()
    #j,k = phi.shape
    #if r is None:
        #r = c = numpy.round(numpy.sqrt(j))
        #assert r*c == j
        
    #m = numpy.floor(numpy.sqrt(k))
    #n = numpy.ceil(k/float(m))
    #assert m*n >= k 

    #f = plt.figure()
    #for i in xrange(k):
        
        #u = numpy.floor(i / m) 
        #v = i % n
        
        #im = numpy.reshape(phi[:,i], (r,c))
        #ax = f.add_axes([float(u)/m, float(v)/n, 1./m, 1./n])
        #ax.imshow(im, cmap = 'RdBu', interpolation = 'nearest')
        #plt.axis('off')

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
        

def plot_value_functions(size, m, b):
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
    w_be = m.get_lstd_weights(b.thetas[-1][:-1]) # TODO add lambda parameter here
    v_be = numpy.dot(b.thetas[-1][:-1], w_be)
    ax.imshow(numpy.reshape(v_be, (size, size)), cmap = 'gray', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # least squares solution with true value function
    ax = f.add_subplot(313)
    w_lsq = numpy.linalg.lstsq(b.thetas[-1][:-1], m.V)[0]
    v_lsq = numpy.dot(b.thetas[-1][:-1], w_lsq)
    ax.imshow(numpy.reshape(v_lsq, (size, size)), cmap = 'gray', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    print 'bellman error norm from v: ', numpy.linalg.norm(m.V - v_be)
    print 'lsq error norm from v: ', numpy.linalg.norm(m.V - v_lsq)

if __name__ == '__main__':
    sirf.script(main)
