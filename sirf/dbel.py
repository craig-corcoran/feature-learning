import copy
import time
import numpy
import scipy.sparse
import theano
import theano.sparse
import theano.sandbox.linalg
import grid_world
import matplotlib.pyplot as plt


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

    def __init__(self, R, P, gam = (1-2e-2)):
        self.R = R
        self.P = P
        self.gam = gam
        self.V = self.get_value_func(self.R, self.P, self.gam)

    def get_value_func(self, R, P, gam):
        assert scipy.sparse.issparse(R) is False
        assert scipy.sparse.issparse(P) is False

        a = numpy.eye(P.shape[0]) - gam * P
        return numpy.linalg.solve(a, R)[:,None]

        #v_p = self.get_stationary(R, P, gam)
        #assert numpy.linalg.norm(v_p - v[:,None]) < 1e-6
        #return v
        
    #def get_stationary(self, R, P, gam, eps = 1e-8):
        #V = numpy.ones((P.shape[0], 1), dtype = numpy.float64) / P.shape[0]
        #if R.ndim == 1:
            #R = R[:,None]
        
        #delta = 1e4
        #while numpy.linalg.norm(delta) > eps:
            #delta = R + gam * numpy.dot(P, V) - V
            #V = V + delta
        #return V

    def get_lstd_weights(self, PHI):

        a = numpy.dot(PHI.T, (PHI - self.gam * numpy.dot(self.P, PHI)))
        b = numpy.dot(PHI.T, self.R)
        if a.ndim > 0:
            return numpy.linalg.solve(a,b) 
        return numpy.array(b/a)

    def bellman_error(self, PHI, w = None):
        
        if w is None:
            w = self.get_lstd_weights(PHI)
        
        A = (PHI - self.gam * numpy.dot(self.P, PHI))
        return numpy.linalg.norm(self.R - numpy.dot(A, w))

    def value_error(self, PHI, w=None):

        if PHI.ndim == 1:
            PHI = PHI[:,None]
        
        if w is None:
            w = numpy.linalg.lstsq(PHI, self.V)[0]

        return numpy.linalg.norm(self.V - numpy.dot(PHI, w))


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
    
    def __init__(self, k, n, theta = None):
        
        self.k = k
        self.theta = theta
        if self.theta is None:
            self.theta = numpy.random.standard_normal((n,k))

        t_theta = theano.tensor.dmatrix('theta')
        t_S = theano.sparse.csr_matrix('S')
        t_S_p = theano.sparse.csr_matrix('S_p')
        t_R = theano.sparse.csr_matrix('R')
        t_gam = theano.tensor.dscalar('gamma')

        # encode s and s'
        t_PHI = theano.sparse.structured_dot(t_S, t_theta)
        t_PHI_p = theano.sparse.structured_dot(t_S_p, t_theta)
        
        # lstd weights for BE using normal eqns
        a = theano.tensor.dot(t_PHI.T, (t_PHI - t_gam * t_PHI_p))
        b = theano.sparse.structured_dot(t_R.T, t_PHI).T
        #t_w_lstd = theano.sandbox.linalg.solve(a,b) # solve currently has no gradient implemented
        t_w_lstd = theano.tensor.dot(theano.sandbox.linalg.matrix_inverse(a), b)
        
        # bellman error loss
        e = t_R - theano.tensor.dot((t_PHI - t_gam * t_PHI_p), t_w_lstd) # error vector
        L_be = theano.tensor.sqrt(theano.tensor.sum(theano.tensor.sqr(e))) # TODO need sqrt?
        self.loss_be = theano.function([t_theta, t_S, t_S_p, t_R, t_gam], L_be)

        # regularization
        el1 = theano.tensor.sum(abs(t_theta)) # l1 norm
        self.reg = theano.function([t_theta], el1) 
        greg = theano.grad(el1, [t_theta])[0]
        self.grad_reg = theano.function([t_theta], greg)

        # bellman error gradient
        grad_L_be = theano.grad(L_be, [t_theta])[0]
        self.grad_be = theano.function([t_theta, t_S, t_S_p, t_R, t_gam], grad_L_be)

    
    def get_phi(self, X):
        if scipy.sparse.issparse(X):
            return X.dot(self.theta)
        return numpy.dot(X, self.theta)

    def sample_lstd_weights(self, PHI, PHI_p, R, gam):

        a = numpy.dot(PHI.T, (PHI - gam * PHI_p)) 
        if scipy.sparse.issparse(R):
            b = R.T.dot(PHI).T
        else:
            b = numpy.dot(PHI.T, R)
        
        if a.ndim > 0:
            return numpy.linalg.solve(a,b) 
        return b/a

    #def loss_be(self, PHI, PHI_p, R, gam, w = None):

        #if w is None: # solve lstd for given samples
            #w = self.sample_lstd_weights(PHI, PHI_p, R, gam)

        #if PHI.ndim == 1:
            #PHI = PHI[:,None]
            #PHI_p = PHI_p[:,None]
            
        #return numpy.linalg.norm(R - numpy.dot((PHI - gam * PHI_p), w))

    def loss_r(self, PHI, R):
        R = R.todense()
        return numpy.linalg.norm(R - numpy.dot(PHI, numpy.linalg.lstsq(PHI, R)[0]))

    def loss_P(self, PHI, PHI_p, gam):
        
        return numpy.linalg.norm(PHI_p - numpy.dot(PHI, numpy.linalg.lstsq(PHI, PHI_p)[0]))

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
    
    l_be = b.loss_be(PHI, PHI_p, R, m.gam)
    

    vphi = S.dot(m.V)
    vphi_p = S_p.dot(m.V)
    v_be = b.loss_be(vphi, vphi_p, R, m.gam)

    cphi = c.get_phi(S)
    cphi_p = c.get_phi(S_p)
    one_be = c.loss_be(cphi, cphi_p, R, m.gam)

    
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

def test_theano_basis(k=16, n=81, n_samples = 5000, alpha = 1.e-1,
        reg_param = 1.e-2, decay = 1.-1e-4, n_iters = 10000):
    
    size = numpy.sqrt(n)

    mdp = grid_world.MDP(walls_on = True)
    m = Model(mdp.env.R, mdp.env.P) 
    theta = numpy.random.standard_normal((n,k))
    b = Basis(k, n, theta)
    t = TheanoBasis(k, n, theta)
    
    R = numpy.zeros(1)
    # make sure we saw a reward in the sample set
    while not (len(R.nonzero()[0]) > 0):
        
        X = mdp.sample_grid_world(n_samples)
        numpy.random.shuffle(X)
        X = scipy.sparse.csr_matrix(X[:,:-2], dtype = numpy.float64) # throw out the actions

        n_vars = (X.shape[1]-1)/2.

        S = X[:,:n_vars]
        S_p = X[:,n_vars:-1]
        R = X[:,-1]
    

    PHI = b.get_phi(S)
    PHI_p = b.get_phi(S_p)

    t_init = time.time()
    be_loss_np = b.loss_be(PHI, PHI_p, R, m.gam)
    print 'time to compute bellman error, numpy: ',  time.time() - t_init

    t_init = time.time()
    be_loss_th = t.loss_be(t.theta, S, S_p, R, m.gam)
    print 'time to compute bellman error, theano: ',  time.time() - t_init

    t_init = time.time()
    be_grad = t.grad_be(t.theta, S, S_p, R, m.gam).shape
    print 'time to compute bellman error gradient: ',  time.time() - t_init

    print 'bellman error loss, numpy functions: ',
    print 'bellman error loss, theano functions: ', be_loss_th
    print 'gradient shape: ', be_grad
    print 'gradient l1 norm: ', numpy.sum(numpy.abs(be_grad))

    # follow gradient of bellman error loss
    for i in xrange(n_iters):
        

        t.theta -= alpha * t.grad_be(t.theta, S, S_p, R, m.gam) # loss min
        t.theta -= alpha * reg_param * t.grad_reg(t.theta)
        if i % (n_iters / 10) == 0:
            print i
            print t.loss_be(t.theta, S, S_p, R, m.gam)
            b.theta = t.theta
            PHI = b.get_phi(S)
            PHI_p = b.get_phi(S_p)
            print b.loss_be(PHI, PHI_p, R, m.gam)
            alpha = alpha * decay
    
    plot_features(t.theta)
    plt.show()

# TODO loss on perfect model BE and stsq
# add walls

def plot_features(phi, r = None, c = None):
 
    j,k = phi.shape
    if r is None:
        r = c = numpy.round(numpy.sqrt(j))
        assert r*c == j
        
    m = numpy.ceil(numpy.sqrt(k))
    n = numpy.ceil(k/float(m))

    print m, n

    f = plt.figure()
    assert m*n == k # this may not work for any k
    
    for i in xrange(k):
        
        u = numpy.floor(i / m) 
        v = i % n
        
        im = numpy.reshape(phi[:,i], (r,c))
        ax = f.add_axes([float(u)/m, float(v)/n, 1./m, 1./n])
        ax.imshow(im, cmap = 'gray', interpolation = 'nearest')
    

if __name__ == '__main__':
    test_theano_basis()
