import copy
import numpy
import scipy.sparse
import theano
import theano.sparse
import theano.sandbox.linalg
import grid_world
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg

# TODO 
# on-policy v uniform weighting for true be, value error
# lstsq v solve, alternative normal equations
# R and P sparse in the model/env
# sparse matrix lstsq?
# L_q
# reconstuctive versions of the above
# theano version and gradients for all
# alternate solutions for v

class Model:
    ''' RL model, including reward, transition function, and functions for 
    getting the bellman error, etc'''

    def __init__(self, R, P, gam = (1-4e-3)):
        self.R = R
        self.P = P
        self.gam = gam
        self.V = self._value_func(self.R, self.P, self.gam)
        self.mu = self._stationary_dist(self.R, self.P, self.gam)

    def _value_func(self, R, P, gam):
        assert scipy.sparse.issparse(R) is False
        assert scipy.sparse.issparse(P) is False

        a = numpy.eye(P.shape[0]) - gam * P
        return numpy.linalg.solve(a, R)[:,None]

    def _stationary_dist(self, R, P, gam, eps = 1e-8):
    
        d = numpy.random.random(R.shape)
        d = d / numpy.linalg.norm(d)
        delta = 1e4
        while numpy.linalg.norm(delta) > eps:
            d = d / numpy.linalg.norm(d)
            d_new = numpy.dot(d.T, P)
            delta = d - d_new
            d = d_new

        return d/numpy.linalg.norm(d)

    def get_lstd_weights(self, PHI):

        a = numpy.dot(PHI.T, (PHI - self.gam * numpy.dot(self.P, PHI)))
        b = numpy.dot(PHI.T, self.R)
        if a.ndim > 0:
            return numpy.linalg.solve(a,b) 
        return numpy.array(b/a)

    def bellman_error(self, PHI, w = None, weighting = 'uniform'):
        
        if w is None:
            w = self.get_lstd_weights(PHI)
        
        A = (PHI - self.gam * numpy.dot(self.P, PHI))
        # diagonal weight matrix
        D = numpy.diag(self.mu) if weighting is 'stationary' else numpy.eye(PHI.shape[0])
        return numpy.linalg.norm(numpy.dot(D, (self.R - numpy.dot(A, w))))

    def value_error(self, PHI, w = None, weighting = 'uniform'):

        if PHI.ndim == 1:
            PHI = PHI[:,None]
        
        if w is None:
            w = numpy.linalg.lstsq(PHI, self.V)[0]

        # diagonal weight matrix            
        D = numpy.diag(self.mu) if weighting is 'stationary' else numpy.eye(PHI.shape[0])
        return numpy.linalg.norm(numpy.dot(D, self.V - numpy.dot(PHI, w)))

class OptimalPolicy:
    ''' acts according to the value function of a random agent - should be 
    sufficient in grid world'''

    def __init__(self, env, m):
        self.env = env
        self.m = m
        self.v = m.V
    

    def choose_action(self, actions):
        
        max_val = -numpy.inf
        act = None
        for a in actions:
            next_s = self.env.next_state(a)
            next_s = self.env.state_to_index(next_s)
            val = self.v[next_s]
            if val > max_val:
                act = a
                max_val = val
        assert act is not None

        return act


class Basis:
    ''' k dimensional basis over discrete state space of size n 
    given tabular features x, phi = theta.T * x'''
    
    def __init__(self, k, n, theta = None):
        
        self.k = k
        self.theta = theta
        if self.theta is None:
            self.theta = numpy.random.standard_normal((n,k))
    
    def get_phi(self, X):
        if scipy.sparse.issparse(X):
            return X.dot(self.theta)
        return numpy.dot(X, self.theta)

    def sample_lstd_weights(self, PHI, PHI_p, R, gam):
        ''' given samples of features PHI, next step features PHI_p, and 
        rewards R, solve for lstd weights'''

        a = numpy.dot(PHI.T, (PHI - gam * PHI_p)) 
        if scipy.sparse.issparse(R):
            b = R.T.dot(PHI).T
        else:
            b = numpy.dot(PHI.T, R)
        
        if a.ndim > 0:
            return numpy.linalg.solve(a,b) 
        return b/a

    def loss_be(self, PHI, PHI_p, R, gam, w = None):

        if w is None: # solve lstd for given samples
            w = self.sample_lstd_weights(PHI, PHI_p, R, gam)

        if PHI.ndim == 1:
            PHI = PHI[:,None]
            PHI_p = PHI_p[:,None]
            
        return numpy.linalg.norm(R - numpy.dot((PHI - gam * PHI_p), w))

    def loss_r(self, PHI, R):
        R = R.todense()
        return numpy.linalg.norm(R - numpy.dot(PHI, numpy.linalg.lstsq(PHI, R)[0]))

    def loss_P(self, PHI, PHI_p, gam):
        
        return numpy.linalg.norm(PHI_p - numpy.dot(PHI, numpy.linalg.lstsq(PHI, PHI_p)[0]))

class TheanoBasis:
    
    def __init__(self, n, k_r, k_m, theta = None, loss_type = 'bellman'):

        self.k_r = k_r        
        self.k_m = k_m

        self.n = n
        self.theta = theta

        num_params = n*(k_m + k_r)
        if self.theta is None:
            self.theta = numpy.random.standard_normal(num_params)
        else:
            assert self.theta.shape == (num_params,)

        self.t_theta_m = theano.tensor.dmatrix('theta_m')
        self.t_theta_r = theano.tensor.dmatrix('theta_r')
        
        self.t_S = theano.sparse.csr_matrix('S')
        self.t_S_p = theano.sparse.csr_matrix('S_p')
        self.t_R = theano.sparse.csr_matrix('R')
        self.t_gam = theano.tensor.dscalar('gamma')
        self.t_reg = theano.tensor.dscalar('regularization')
        
        # shared variables
        t_n = theano.tensor.sharedvar.scalar_constructor(n) #theano.tensor.iscalar('n')
        t_k_r = theano.tensor.sharedvar.scalar_constructor(k_r) #theano.tensor.iscalar('k')
        t_k_m = theano.tensor.sharedvar.scalar_constructor(k_m) #theano.tensor.iscalar('k')
        
        # encode s and s' using the full feature set, theta_m and theta_r
        #t_theta_r_rect = theano.tensor.reshape(self.t_theta_r, (t_n, t_k_r))
        t_PHI_r = theano.sparse.structured_dot(self.t_S, self.t_theta_r)
        #t_PHI_p_r = theano.sparse.structured_dot(t_S_p, t_theta_r)

        #t_theta_m_rect = theano.tensor.reshape(self.t_theta_m, (t_n, t_k_m))
        #t_PHI_m = theano.sparse.structured_dot(t_S, t_theta_m)
        #t_PHI_p_m = theano.sparse.structured_dot(t_S_p, t_theta_m)
        self.t_theta_full = theano.tensor.horizontal_stack(self.t_theta_r, self.t_theta_m)
        t_PHI = theano.sparse.structured_dot(self.t_S, self.t_theta_full)
        t_PHI_p = theano.sparse.structured_dot(self.t_S_p, self.t_theta_full)

        # lstd weights for bellman error using normal eqns
        a = theano.tensor.dot(t_PHI.T, (t_PHI - self.t_gam * t_PHI_p))
        b = theano.sparse.structured_dot(self.t_R.T, t_PHI).T
        #t_w_lstd = theano.sandbox.linalg.solve(a,b) # solve currently has no gradient implemented
        t_w_lstd = theano.tensor.dot(theano.sandbox.linalg.matrix_inverse(a), b)
        
        # bellman error loss
        e_be = self.t_R - theano.tensor.dot((t_PHI - self.t_gam * t_PHI_p), t_w_lstd) # error vector
        self.L_be = theano.tensor.sqrt(theano.tensor.sum(theano.tensor.sqr(e_be))) # TODO need sqrt?
        self.loss_be = theano.function([self.t_theta_r, self.t_theta_m, \
                        self.t_S, self.t_S_p, self.t_R, self.t_gam], self.L_be)

        # reward loss: ||PHI_r (PHI_r.T * PHI_r)^-1 PHI_r.T * R - R||
        c = theano.tensor.dot(t_PHI_r.T, t_PHI_r)
        d = theano.sparse.structured_dot(self.t_R.T, t_PHI_r).T
        w_r = theano.tensor.dot(theano.sandbox.linalg.matrix_inverse(c), d)
        e_r = self.t_R - theano.tensor.dot(t_PHI_r, w_r)
        self.L_r = theano.tensor.sqrt(theano.tensor.sum(theano.tensor.sqr(e_r)))
        #self.loss_r = theano.function([t_theta_r, t_S, t_S_p, t_R], L_r)

        # model loss: ||PHI (PHI.T * PHI)^-1 PHI.T * PHI_p # should model loss include reward features? yes i think so
        aa = theano.tensor.dot(t_PHI.T, t_PHI)
        bb = theano.tensor.dot(t_PHI_p.T, t_PHI).T
        w_m = theano.tensor.dot(theano.sandbox.linalg.matrix_inverse(aa), bb) # least squares weight matrix
        e_m = t_PHI_p - theano.tensor.dot(t_PHI, w_m) # model error matrix
        self.L_m = theano.tensor.sqrt(theano.tensor.sum(theano.tensor.sqr(e_m))) # frobenius norm
        #self.loss_m = theano.function([t_theta_r, t_theta_m, t_S, t_S_p], L_m)

        # l1 norm regularization
        #self.el1 = (theano.tensor.sum(abs(self.t_theta_r)) + \
               #theano.tensor.sum(abs(self.t_theta_m))) / (t_n * (t_k_r + t_k_m))
        
        self.el1 = (theano.tensor.sum(abs(self.t_PHI))) / (t_n * (t_k_r + t_k_m))
        #self.reg = theano.function([t_theta_r, t_theta_m], el1) 
        
        ## combined loss function L + reg * R
        #loss = self.L_be + self.t_reg * self.el1
        #self.loss = theano.function([self.t_theta_r, self.t_theta_m, self.t_S, self.t_S_p, self.t_R, self.t_gam, self.t_reg], loss)

        ## overall loss gradient
        #lgrad = theano.grad(loss, [self.t_theta])[0]
        #self.grad_loss = theano.function([self.t_theta_r, self.t_theta_m, self.t_S, self.t_S_p, self.t_R, self.t_gam, self.t_reg], lgrad)

        self.set_loss(loss_type) 

    def set_loss(self, loss_type):

        self.loss_type = loss_type
        
        if loss_type is 'bellman':
            loss = self.L_be + self.t_reg * self.el1
            grad_var = self.t_theta_full # TODO try this
            #grad_var = self.t_theta_m
        elif loss_type is 'reward':
            loss = self.L_r + self.t_reg * self.el1
            grad_var = self.t_theta_r
        elif loss_type is 'model':
            loss = self.L_m + self.t_reg * self.el1
            grad_var = self.t_theta_m
        else:
            print 'invalid loss type string'
            assert False

        self.loss_func = theano.function([self.t_theta_r, self.t_theta_m, \
                self.t_S, self.t_S_p, self.t_R, self.t_gam, self.t_reg], loss,\
                on_unused_input='ignore')
        
        gloss = theano.grad(loss, [grad_var])[0]
        self.loss_grad = theano.function([self.t_theta_r, self.t_theta_m, \
                self.t_S, self.t_S_p, self.t_R, self.t_gam, self.t_reg], \
                gloss, \
                on_unused_input='ignore')

    
    def reshape_theta(self, theta):
        theta = numpy.reshape(theta, (self.n, self.k_r + self.k_m))
        theta_r = theta[:,:self.k_r]
        theta_m = theta[:,self.k_r:(self.k_r + self.k_m)]
        return theta_r, theta_m

    def flat_loss(self, theta, S, S_p, R, gam, reg):
        ''' loss function to be called by scipy.optimize, which requires that 
        the parameter being optimized (theta) is a flat vector and is the first
        argument passed into the function. Also expects sparse vectors of states
        and rewards S, S_p, R. gam: discounting rate gamma. reg: regularization
        parameter'''
        
        theta_r, theta_m = self.reshape_theta(theta)

        return self.loss_func(theta_r, theta_m, S, S_p, R, gam, reg)

    def flat_grad(self, theta, S, S_p, R, gam, reg):

        theta_r, theta_m = self.reshape_theta(theta)

        grad = self.loss_grad(theta_r, theta_m, S, S_p, R, gam, reg)
        if self.loss_type is 'bellman':
            assert grad.shape == (self.n, self.k_m+self.k_r)
        elif self.loss_type is 'model':
            grad = numpy.hstack((numpy.zeros((self.n, self.k_r)), grad))
        elif self.loss_type is 'reward':
            grad = numpy.hstack((grad, numpy.zeros((self.n, self.k_m))))
        else: 
            print 'invalid loss type string'
            assert False
        return grad.flatten()

    def flat_be(self, theta, S, S_p, R, gam):
        theta_r, theta_m = self.reshape_theta(theta)

        return self.loss_be(theta_r, theta_m, S, S_p, R, gam)

    def get_phi(self, X):
        theta = self.get_theta_matrix()
        if scipy.sparse.issparse(X):
            return X.dot(theta)
        return numpy.dot(X, theta)

    def get_theta_matrix(self):
        
        return numpy.reshape(self.theta, (self.n, self.k_m + self.k_r))

    def lstd_weights(self, PHI, PHI_p, R, gam):

        a = numpy.dot(PHI.T, (PHI - gam * PHI_p)) 
        if scipy.sparse.issparse(R):
            b = R.T.dot(PHI).T
        else:
            b = numpy.dot(PHI.T, R)
        
        if a.ndim > 0:
            return numpy.linalg.solve(a,b) 
        return b/a

    #def loss_r(self, PHI, R):
        #R = R.todense()
        #return numpy.linalg.norm(R - numpy.dot(PHI, numpy.linalg.lstsq(PHI, R)[0]))

    #def loss_P(self, PHI, PHI_p, gam):
        
        #return numpy.linalg.norm(PHI_p - numpy.dot(PHI, numpy.linalg.lstsq(PHI, PHI_p)[0]))


def test_theano_basis(n = 81, k_r = 1, k_m=8, mb_size = 40000,
        reg_param = 100., max_iter = 2, weighting = 'uniform', threshold = 1e-2, alpha = 0.8):
    
    mdp = grid_world.MDP(walls_on = True)
    m = Model(mdp.env.R, mdp.env.P) 
    #mdp.policy = OptimalPolicy(mdp.env, m)

    theta = 1e-4*numpy.random.standard_normal(n*(k_r+k_m))
    t = TheanoBasis(n, k_r, k_m, theta)
    
    # sample the hold out test set
    X = mdp.sample_grid_world(mb_size, distribution = weighting)
    numpy.random.shuffle(X)
    X = scipy.sparse.csr_matrix(X[:,:-2], dtype = numpy.float64) # throw out the actions

    n_vars = (X.shape[1]-1)/2.
    s_test = X[:,:n_vars]
    s_p_test = X[:,n_vars:-1]
    r_test = X[:,-1]

    sample_be = numpy.zeros(1, dtype = numpy.float64)
    true_be = numpy.zeros(1, dtype = numpy.float64)
    true_lsq = numpy.zeros(1, dtype = numpy.float64)

    sample_be[0] = t.flat_be(t.theta, s_test, s_p_test, r_test, m.gam)
    true_be[0] = m.bellman_error(t.get_theta_matrix(), weighting = weighting)
    true_lsq[0] = m.value_error(t.get_theta_matrix(), weighting = weighting)

    loss_types = ['reward', 'model', 'bellman']
    #loss_types = ['bellman']
 
    cnt = 1
    for lt in loss_types:
        
        print 'using %s loss' % lt
        t.set_loss(lt)
        delta = 1
        running_avg = 1
    
        while running_avg > threshold:
            
            print 'minibatch: ', cnt

            X = mdp.sample_grid_world(mb_size, distribution = weighting)
            numpy.random.shuffle(X)
            X = scipy.sparse.csr_matrix(X[:,:-2], dtype = numpy.float64) # throw out the actions

            s = X[:,:n_vars]
            s_p = X[:,n_vars:-1]
            r = X[:,-1]

            if cnt == 1:
                print 'initial sample loss: ', t.flat_be(t.theta, s, s_p, r, m.gam)
                print 'initial true bellman error: ', m.bellman_error(t.get_theta_matrix())
            
            theta_old = copy.deepcopy(t.theta)
            t.theta= fmin_cg(t.flat_loss, t.theta, t.flat_grad,
                                    args = (s, s_p, r, m.gam, reg_param),
                                    full_output = False,
                                    maxiter = max_iter, 
                                    gtol = 1e-8) # gtol

            delta = numpy.linalg.norm(t.theta - theta_old)
            running_avg = running_avg * (1-alpha) + delta * alpha
            print 'change in theta: ', delta
            print 'running average of delta', running_avg

            sample_be = numpy.hstack((sample_be, t.flat_be(t.theta, s_test, s_p_test, r_test, m.gam)))
            true_be = numpy.hstack((true_be, m.bellman_error(t.get_theta_matrix(), weighting = weighting)))
            true_lsq = numpy.hstack((true_lsq,  m.value_error(t.get_theta_matrix(), weighting = weighting)))

            print 'sample BE loss: ', sample_be[cnt]
            print 'true BE : ', true_be[cnt]

            cnt += 1

        #plot_features(t.get_theta_matrix())
        #plt.show()
    
    # plot basis functions
    plot_features(t.get_theta_matrix())
    plt.savefig('basis.k=%i.%s.pdf' % (k_r+k_m, weighting))
    
    # plot learning curve
    #plt.figure()
    plt.clf()
    ax = plt.axes()
    x = range(len(sample_be))
    ax.plot(x, sample_be/max(sample_be), 'r-', x, true_be/max(true_be), 'g-', x, true_lsq / max(true_lsq), 'b-')
    ax.legend(['Test BE','True BE','True RMSE'])
    plt.savefig('loss.k=%i.%s.pdf' % (k_r+k_m, weighting))

    # plot value functions, true and approx
    plt.clf()
    f = plt.figure()
    side = numpy.sqrt(n)
    
    # true model value fn
    f.add_subplot(311)
    plt.imshow(numpy.reshape(m.V, (side, side)), cmap = 'gray', interpolation = 'nearest')

    # bellman error estimate (using true model)
    f.add_subplot(312)
    basis = t.get_theta_matrix()
    w_be = m.get_lstd_weights(basis)
    v_be = numpy.dot(basis, w_be)
    plt.imshow(numpy.reshape(v_be, (side, side)), cmap = 'gray', interpolation = 'nearest')
    
    # least squares solution with true value function
    f.add_subplot(313)
    w_lsq = numpy.linalg.lstsq(basis, m.V)[0]
    v_lsq = numpy.dot(basis, w_lsq)
    plt.imshow(numpy.reshape(v_lsq, (side, side)), cmap = 'gray', interpolation = 'nearest')
    plt.savefig('value.k=%i.%s.pdf' % (k_r+k_m, weighting))

# TODO 
# effect of walls
# remove grid
# plot value function - lstd and lsqr
# normalize regularization loss by number of samples
# true bellman error weighted by on-policy distribution
# test effects of alternate normal eqn formulation
# stop at convergence instead of iters
# try using perfect information - uniform and policy weighted

def plot_features(phi, r = None, c = None):
 
    j,k = phi.shape
    if r is None:
        r = c = numpy.round(numpy.sqrt(j))
        assert r*c == j
        
    m = numpy.ceil(numpy.sqrt(k))
    n = numpy.ceil(k/float(m))

    f = plt.figure()
    assert m*n == k # this may not work for any k
    
    for i in xrange(k):
        
        u = numpy.floor(i / m) 
        v = i % n
        
        im = numpy.reshape(phi[:,i], (r,c))
        ax = f.add_axes([float(u)/m, float(v)/n, 1./m, 1./n])
        ax.imshow(im, cmap = 'gray', interpolation = 'nearest')

def test_basis(k=4, n=81, n_samples = 5000):
    
    mdp = grid_world.MDP()
    m = Model(mdp.env.R, mdp.env.P) 
    b = Basis(k,n,numpy.random.standard_normal((n,k)))
    c = copy.deepcopy(b)
    c.theta = c.theta[:,0] # c has only one random feature
    
    
    # test true model loss functions
    print 'exact (unif. weighted) bellman error: ', m.bellman_error(b.theta)
    print 'exact (unif. weighted) value error: ', m.value_error(b.theta)

    # check that the value function has zero bellman error
    assert m.bellman_error(m.V) < 1e-8
    assert m.value_error(m.V) < 1e-8
    
    X = mdp.sample_grid_world(n_samples)
    numpy.random.shuffle(X)
    X = scipy.sparse.csr_matrix(X[:,:-2], dtype = numpy.float64) # throw out the actions

    n_vars = (X.shape[1]-1)/2.
    assert n_vars % 1 == 0 # assert integer
    assert n_vars == n

    S = X[:,:n_vars]
    S_p = X[:,n_vars:-1]
    R = X[:,-1]
    
    # make sure we saw a reward in the sample set
    assert len(R.nonzero()[0]) > 0

    PHI = b.get_phi(S)
    PHI_p = b.get_phi(S_p)
    
    # make sure shapes match
    assert PHI.shape == PHI_p.shape
    assert PHI.shape[0] == R.shape[0]

    # make sure feature vectors are dense
    assert scipy.sparse.issparse(PHI) is False
    assert scipy.sparse.issparse(PHI_p) is False
    
    l_be = b.flat_be(PHI, PHI_p, R, m.gam)
    

    vphi = S.dot(m.V)
    vphi_p = S_p.dot(m.V)
    v_be = b.flat_be(vphi, vphi_p, R, m.gam)

    cphi = c.get_phi(S)
    cphi_p = c.get_phi(S_p)
    one_be = c.flat_be(cphi, cphi_p, R, m.gam)

    
    print 'sampled bellman error loss with k features: ', l_be
    print 'sampled bellman error loss with only one feature: ', one_be
    print 'sampled bellman error loss with value function feature: ', v_be

    print 'sampled reward loss: ', b.loss_r(PHI, R)
    print 'sampled model loss: ', b.loss_P(PHI, PHI_p, m.gam)

    
    # more features are better
    assert one_be > l_be
    

    # the sample BE is less for the value function feature than random
    if l_be == v_be:
        assert l_be == 0
    else:
        assert v_be < one_be


    

if __name__ == '__main__':
    test_theano_basis()
