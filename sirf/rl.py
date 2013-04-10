import numpy
import scipy.sparse

class Model:
    ''' RL model, including reward, transition function, and functions for 
    getting the bellman error, etc'''

    def __init__(self, R, P, gam = (1-4e-3), lam = 0.):
        self.R = R
        self.P = P
        self.gam = gam
        self.lam = lam
        assert lam < 1

        self.Plam = numpy.zeros_like(P)
        Pi = numpy.eye(self.P.shape[0])
        n = calc_discount_horizon(lam, gam)
        for i in xrange(n):
            Pi = numpy.dot(Pi, self.P)
            self.Plam += self.lam**i * self.gam**(i+1) * Pi
        self.Plam *= (1. - self.lam)
        
        self.V = self._value_func(self.R, self.P, self.gam)
        self.mu = self._policy_dist(self.R, self.P, self.gam)
        self.d_policy = self._optimal_policy(self.V)

    def _value_func(self, R, P, gam):
        assert scipy.sparse.issparse(R) is False
        assert scipy.sparse.issparse(P) is False

        a = numpy.eye(P.shape[0]) - gam * P
        return numpy.linalg.solve(a, R)[:,None]

    def _policy_dist(self, R, P, gam, eps = 1e-8):
    
        d = numpy.random.random(R.shape)
        d = d / numpy.linalg.norm(d)
        delta = 1e4
        while numpy.linalg.norm(delta) > eps:
            d = d / numpy.linalg.norm(d)
            d_new = numpy.dot(d.T, P)
            delta = d - d_new
            d = d_new

        return d/numpy.linalg.norm(d)

    def _optimal_policy(self, val, epsilon = 0.):
        ''' creates a dictionary whose keys are state indexes and whose values 
        are a set of integer values corresponding to the optimal actions 
        (adjacent state).'''

        A = self.P * val.T    
        assert A.ndim == 2
        d_policy = {}
        maxs = numpy.max(A, axis = 1)
        for i, ma in enumerate(maxs):
            next_best = set((A[i,:] - ma >= epsilon).nonzero()[0])
            d_policy[i] = next_best        
        return d_policy


    def get_lstd_weights(self, PHI, shift = 1e-7): 

        A = numpy.dot(PHI.T, (PHI - numpy.dot(self.Plam, PHI)))
        A += shift * numpy.eye(A.shape[0])
        b = numpy.dot(PHI.T, self.R)
        if A.ndim > 0:
            return numpy.linalg.solve(A,b) 
        return numpy.array(b/A)

    def append_bias(self, PHI):
        return numpy.hstack((PHI, numpy.ones((PHI.shape[0], 1))))

    def policy_distance(self, PHI, w = None, weighting = 'uniform'):
        ''' Pass val in as a column vector returns (hamming) distance to 
        the optimal policy from 0 to 1.
        '''
        # TODO should weighting be according to optimal policy distribution?
        PHI = self.append_bias(PHI)
        if w is None:
            w = self.get_lstd_weights(PHI)
        val = PHI.dot(w)

        n_states = len(val)
        assert n_states == len(self.V)
        d_policy = self._optimal_policy(val)
        
        num_right = 0
        for i, next_set in d_policy.items():
            # if one of the best action(s) of the tested policy is in the action set for the optimal policy
            if len(next_set.intersection(self.d_policy[i])) > 0: 
                num_right += self.mu[i] if weighting is 'policy' else 1
        norm = numpy.sum(self.mu) if weighting is 'policy' else n_states
        return float(norm - num_right) / norm
    
    def bellman_error(self, PHI, w = None, weighting = 'uniform'):
        PHI = self.append_bias(PHI)
        if w is None:
            w = self.get_lstd_weights(PHI)
        
        A = (PHI - numpy.dot(self.Plam, PHI))
        # diagonal weight matrix

        D = numpy.diag(self.mu) if weighting is 'policy' else numpy.eye(PHI.shape[0])
        return numpy.linalg.norm(numpy.dot(D, (self.R - numpy.dot(A, w))))


    def fullmodel_error(self, PHI, W = None, weighting = 'uniform'):
        
        A = numpy.dot(self.Plam, PHI)
        PHI = self.append_bias(PHI) # include bias in A to be reconstructed
        if W is None:
            W = numpy.linalg.lstsq(PHI, A)[0] 
                
        # diagonal weight matrix
        D = numpy.diag(self.mu) if weighting is 'policy' else numpy.eye(PHI.shape[0])
        return numpy.linalg.norm(numpy.dot(D, (numpy.dot(PHI, W) - A)))

    def model_error(self, PHI, w = None, q = None, weighting = 'uniform'):
        PHI = self.append_bias(PHI)
        if w is None:
            w = self.get_lstd_weights(PHI)

        a = numpy.dot(self.Plam, numpy.dot(PHI, w))
        if q is None:
            q = numpy.linalg.lstsq(PHI, a)[0]

        # diagonal weight matrix
        D = numpy.diag(self.mu) if weighting is 'policy' else numpy.eye(PHI.shape[0])
        return numpy.linalg.norm(numpy.dot(D, (numpy.dot(PHI, q) - a)))

    def reward_error(self, PHI, w = None, weighting = 'uniform'):
        
        PHI = self.append_bias(PHI)
        if w is None:
            w = numpy.linalg.lstsq(PHI, self.R)[0]

        D = numpy.diag(self.mu) if weighting is 'policy' else numpy.eye(PHI.shape[0])
        a = numpy.linalg.norm(numpy.dot(D, (numpy.dot(PHI, w) - self.R)))
        return a

    def lsq_error(self, PHI, w = None, weighting = 'uniform'):

        if PHI.ndim == 1:
            PHI = PHI[:,None]

        PHI = self.append_bias(PHI)
        
        if w is None:
            w = numpy.linalg.lstsq(PHI, self.V)[0]

        # diagonal weight matrix            
        D = numpy.diag(self.mu) if weighting is 'policy' else numpy.eye(PHI.shape[0])
        return numpy.linalg.norm(numpy.dot(D, self.V - numpy.dot(PHI, w)))

def calc_discount_horizon(lam, gam, eps = 1e-6):
    return 1 if lam == 0 else int(numpy.ceil(numpy.log(eps) / numpy.log(min(lam, gam))))




