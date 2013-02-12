import copy
import numpy
import scipy.sparse
import theano
import theano.sparse
import theano.sandbox.linalg
import grid_world
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg

#TODO lambda version and distribution weighting

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




