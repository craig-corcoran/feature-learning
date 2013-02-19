import numpy
import scipy.sparse

#TODO lambda versions of losses

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

    def append_bias(self, PHI):
        return numpy.hstack((PHI, numpy.ones((PHI.shape[0], 1))))
    
    def bellman_error(self, PHI, w = None, weighting = 'uniform'):
        PHI = self.append_bias(PHI)
        if w is None:
            w = self.get_lstd_weights(PHI)
        
        A = (PHI - self.gam * numpy.dot(self.P, PHI))
        # diagonal weight matrix
        D = numpy.diag(self.mu) if weighting is 'stationary' else numpy.eye(PHI.shape[0])
        return numpy.linalg.norm(numpy.dot(D, (self.R - numpy.dot(A, w))))

    def model_error(self, PHI, W = None, weighting = 'uniform'):
        
        A = numpy.dot(self.P, PHI)
        PHI = self.append_bias(PHI) # include bias in A to be reconstructed?
        if W is None:
            W = numpy.linalg.lstsq(PHI, A)[0] 
                
        # diagonal weight matrix
        D = numpy.diag(self.mu) if weighting is 'stationary' else numpy.eye(PHI.shape[0])
        return numpy.linalg.norm(numpy.dot(D, (numpy.dot(PHI, W) - A)))

    def reward_error(self, PHI, w = None, weighting = 'uniform'):
        
        PHI = self.append_bias(PHI)

        if w is None:
            w = numpy.linalg.lstsq(PHI, self.R)[0]

        D = numpy.diag(self.mu) if weighting is 'stationary' else numpy.eye(PHI.shape[0])

        a = numpy.linalg.norm(numpy.dot(D, (numpy.dot(PHI, w) - self.R)))
        return a

    def value_error(self, PHI, w = None, weighting = 'uniform'):

        if PHI.ndim == 1:
            PHI = PHI[:,None]

        PHI = self.append_bias(PHI)
        
        if w is None:
            w = numpy.linalg.lstsq(PHI, self.V)[0]

        # diagonal weight matrix            
        D = numpy.diag(self.mu) if weighting is 'stationary' else numpy.eye(PHI.shape[0])
        return numpy.linalg.norm(numpy.dot(D, self.V - numpy.dot(PHI, w)))




