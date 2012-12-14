import copy
import numpy
import scipy.sparse
import grid_world

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
        
    def get_stationary(self, R, P, gam, eps = 1e-8):
        V = numpy.ones((P.shape[0], 1), dtype = numpy.float64) / P.shape[0]
        if R.ndim == 1:
            R = R[:,None]
        
        delta = 1e4
        while numpy.linalg.norm(delta) > eps:
            delta = R + gam * numpy.dot(P, V) - V
            V = V + delta
        return V

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
    given tabular features x, phi = THETA.T * x'''
    
    def __init__(self, k, n, THETA = None):
        
        self.k = k
        self.THETA = THETA
        if self.THETA is None:
            self.THETA = numpy.random.standard_normal((n,k))
    
    def get_phi(self, X):
        if scipy.sparse.issparse(X):
            return X.dot(self.THETA)
        return numpy.dot(X, self.THETA)

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

def test_basis(k=1, n=81, n_samples = 5000):
    
    mdp = grid_world.MDP()
    m = Model(mdp.env.R, mdp.env.P) 
    b = Basis(k,n,numpy.random.standard_normal((n,k)))
    c = copy.deepcopy(b)
    c.THETA = c.THETA[:,0] # c has only one random feature
    
    
    # test true model loss functions
    print 'exact (unif. weighted) bellman error: ', m.bellman_error(b.THETA)
    print 'exact (unif. weighted) value error: ', m.value_error(b.THETA)

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
    print 'bellman error with value only one feature: ', one_be
    print 'bellman error with value function feature: ', v_be

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
    test_basis()
