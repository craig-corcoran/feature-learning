import copy
import numpy
import scipy.sparse
import theano.tensor
import theano.tensor as TT
import theano
import theano.sparse
import theano.sparse as TS
import theano.sandbox.linalg
import matplotlib.pyplot as plt
import grid_world
from scipy.optimize import fmin_cg
from rl import Model

class BellmanBasis:

    LOSSES = 'bellman model reward covariance nonzero l1code l1theta'.split()

    def __init__(self, n, k, beta, loss_type = 'bellman', theta = None,
                 reg_tuple = None, partition = None, wrt = 'all', shift = 1e-6,
                 nonlin = None, nonzero = None):
        
        self.n = n # dim of data
        self.k = k # num of features/columns
        self.loss_type = loss_type
        self.nonzero = nonzero  # set this to some positive float to penalize zero theta vectors.

        if theta is None: 
            theta = 1e-6 * numpy.random.standard_normal((self.n, self.k))
            theta /= numpy.sqrt((theta * theta).sum(axis=0))
            # initialize features sparsely
            #sparsity = 0.8
            #for i in xrange(self.k):
                #z = numpy.random.random(self.n)
                #theta[:,i][z < sparsity] = 0.
        else: assert (theta.shape == (k*n,)) or (theta.shape == (self.n, self.k))
        self.theta = self._reshape_theta(theta)
    
        # partition the features for gradients
        self.d_partition = {}
        if partition is not None:       
            self.partition_theta(partition)
        else:
            self.theta_t = TT.dmatrix('theta')
            self.d_partition['all'] = self.theta_t

        # primitive theano vars
        #self.theta_t = TT.dmatrix('theta')
        self.S_t = theano.sparse.csr_matrix('S')
        self.Rfull_t = theano.sparse.csr_matrix('R')
        self.Mphi_t = TT.dmatrix('M_phi') # mixing matrix for PHI_lam
        self.Mrew_t = TT.dmatrix('M_rew') # mixing matrix for reward_lambda
        self.beta_t = theano.shared(beta) # multiplier on gamma set by env
        self.shift_t = theano.shared(shift)

        # encode s and mix lambda components
        d_nonlin = dict(sigmoid=TT.nnet.sigmoid, relu=lambda z: TT.maximum(0, z))
        self.PHI_full_t = d_nonlin.get(nonlin, lambda z: z)(TS.structured_dot(self.S_t, self.theta_t))
        self.PHIlam_t = TT.dot(self.Mphi_t, self.PHI_full_t)
        self.PHI0_t = self.PHI_full_t[0:self.PHIlam_t.shape[0],:]
        self.Rlam_t = TS.structured_dot(self.Rfull_t.T, self.Mrew_t.T).T #xxx

        self.cov = TT.dot(self.PHI0_t.T, self.PHI0_t) 
        self.cov_inv = theano.sandbox.linalg.matrix_inverse(self.cov)
        #self.cov_inv = theano.sandbox.linalg.matrix_inverse(TS.structured_add(
        #        self.cov, self.shift_t * TS.square_diagonal(TT.ones((k,))))) # l2 reg to avoid sing matrix

        # precompile theano functions and gradients.
        self.losses = {k: self.compile_loss(k) for k in self.LOSSES}

        self.set_loss(loss_type, wrt)
        self.set_regularizer(reg_tuple)

    @property
    def loss_be(self):
        return self.losses['bellman'][0]

    @property
    def theano_vars(self):
        return [self.theta_t, self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t]

    def compile_loss(self, loss):
        kw = dict(on_unused_input='ignore')
        loss_t = getattr(self, '%s_funcs' % loss)()
        loss = theano.function(self.theano_vars, loss_t, **kw)
        return loss, [
            theano.function(self.theano_vars, theano.grad(loss_t, var), **kw)
            for var in self.d_partition.itervalues()]

    def partition_theta(self, partition):
        ''' creates partition of the parameters according to the partition sizes
        given in the partition tuple. ex: partition = {'reward':1, 'model':k-1} 
        '''
        
        print 'partitioning theta'
        names, indices = partition.keys(), partition.values()
        assert sum(indices) == self.k
        self.part_inds = numpy.insert(numpy.cumsum(indices), 0, 0)
        
        n_parts = len(names)
        for i in xrange(n_parts):
            self.d_partition[names[i]] = theano.shared(
                        self.theta[:, self.part_inds[i]: self.part_inds[i+1]]) 
        
        part_list = map( lambda name: self.d_partition[name], names)
        self.theta_t = TT.horizontal_stack(*part_list) 
        self.d_partition['all'] = self.theta_t

    def set_regularizer(self, reg_tuple):
        self.reg_type = reg_tuple
        if reg_tuple is not None:
            self.reg_type, self.reg_param = reg_tuple

    def set_loss(self, loss_type, wrt):
        self.loss_type = loss_type
        self.wrt = wrt
        self.part_num = self.d_partition.keys().index(wrt)
        self.loss_func, self.loss_grads = self.losses[loss_type]
        self.loss_grad = self.loss_grads[self.part_num]

    def l1code_funcs(self):
        '''Minimize the size of the feature values.'''
        return TT.sum(TT.abs_(self.PHI_full_t))

    def l1theta_funcs(self):
        '''Minimize the size of the feature weights.'''
        return TT.sum(TT.abs_(self.theta_t))

    def nonzero_funcs(self):
        '''Try to make sure feature weights are nonzero.'''
        return (1. / ((self.theta_t * self.theta_t).sum(axis=0) + 1e-10)).sum()

    def bellman_funcs(self):

        # lstd weights for bellman error using normal eqns
        b = TT.dot(self.PHI0_t.T, self.Rlam_t)
        a = TT.dot(self.PHI0_t.T, (self.PHI0_t - self.PHIlam_t)) 
        #w_lstd = theano.sandbox.linalg.solve(a,b) # solve currently has no gradient implemented
        #A = theano.sandbox.linalg.matrix_inverse( TS.structured_add( \
        #         a, self.shift_t * TS.square_diagonal(TT.ones((self.k,))))) # l2 reg to avoid sing matrix
        #w_lstd = TT.dot(A, b)
        w_lstd = TT.dot(theano.sandbox.linalg.matrix_inverse(a), b)
        e_be = self.Rlam_t - TT.dot((self.PHI0_t - self.PHIlam_t), w_lstd) # error vector
        return TT.sqrt(TT.sum(TT.sqr(e_be))) # need sqrt?

    def reward_funcs(self):
        
        # reward loss: ||(PHI0 (PHI0.T * PHI0))^-1 PHI0.T * Rlam - Rlam||
        d = TT.dot(self.PHI0_t.T, self.Rlam_t)
        w_r = TT.dot(self.cov_inv, d)
        e_r = self.Rlam_t - TT.dot(self.PHI0_t, w_r)
        return TT.sqrt(TT.sum(TT.sqr(e_r)))

    def model_funcs(self):
        
        # model loss: ||PHI0 (PHI0.T * PHI0)^-1 PHI0.T * PHIlam - PHIlam|| 
        bb = TT.dot(self.PHI0_t.T, self.PHIlam_t)
        w_m = TT.dot(self.cov_inv, bb) # least squares weight matrix
        e_m = self.PHIlam_t - TT.dot(self.PHI0_t, w_m) # model error matrix
        return TT.sqrt(TT.sum(TT.sqr(e_m))) # frobenius norm

    def covariance_funcs(self):
        
        # todo weight by stationary distribution if unsampled?
        A = self.cov - self.beta_t * TT.dot(self.PHI0_t.T,self.PHIlam_t)
        return TT.sum(TT.abs_(A))

    def loss(self, theta, S, R, Mphi, Mrew):
        
        theta = self._reshape_theta(theta)
        loss =  self.loss_func(theta, S, R, Mphi, Mrew)
        #print 'loss pre reg: ', loss
        if self.reg_type is 'l1-code':
            l1_loss, _ = self.losses['l1code']
            loss += self.reg_param * l1_loss(theta, S, R, Mphi, Mrew)
        if self.reg_type is 'l1-theta':
            l1_loss, _ = self.losses['l1theta']
            loss += self.reg_param * l1_loss(theta, S, R, Mphi, Mrew)
        if self.nonzero:
            nz_loss, _ = self.losses['nonzero']
            loss += self.nonzero * nz_loss(theta, S, R, Mphi, Mrew)
        return loss / (S.shape[0] * self.n) # norm loss by num samples and dim of data
        
    def grad(self, theta, S, R, Mphi, Mrew):
        
        theta = self._reshape_theta(theta)
        grad = self.loss_grad(theta, S, R, Mphi, Mrew)
        if self.reg_type is 'l1-code':
            _, l1_grads = self.losses['l1code']
            grad += self.reg_param * l1_grads[self.part_num](theta, S, R, Mphi, Mrew)
        if self.reg_type is 'l1-theta':
            _, l1_grads = self.losses['l1theta']
            grad += self.reg_param * l1_grads[self.part_num](theta, S, R, Mphi, Mrew)
        if self.nonzero:
            _, nz_grads = self.losses['nonzero']
            grad += self.reg_param * nz_grads[self.part_num](theta, S, R, Mphi, Mrew)
        grad = grad / (S.shape[0] * self.n) # norm grad by num samples and dim of data

        if self.wrt is 'all':
            return grad.flatten()
        else: # pad with zeros for partition
            full_grad = numpy.zeros_like(theta)
            full_grad[:, self.part_inds[self.part_num]:self.part_inds[self.part_num+1]] = grad
            return full_grad.flatten()

    def _calc_n_steps(self, lam, gam, eps):

        # calculate number of steps to perform lambda-averaging over
        if lam == 0:
            n_time_steps = 1
        else:
            n_time_steps = int(numpy.ceil(numpy.log(eps) / numpy.log(min(lam,gam))))

        return n_time_steps

    def get_mixing_matrices(self, m, lam, gam, sampled = True, eps = 1e-5, dim = None):
        ''' returns a matrix for combining feature vectors from different time
        steps for use in TD(lambda) algorithms. m is the number of final
        rows/samples, dim is the dimension (equal to m and not needed if not
        sampled), and n_steps is the number of extra time steps. Here we use
        only all m+n_step updates; slightly different than recursive TD
        algorithm, which uses all updates from all samples '''

        n_steps = self._calc_n_steps(lam, gam, eps = eps)
        vec = map(lambda i: (lam*gam)**i, xrange(n_steps)) # decaying weight vector
        if sampled:
            assert dim is not None
            M = numpy.zeros((m, m + n_steps-1))
            for i in xrange(m): # for each row
                M[i, i:i+n_steps] = vec
            m_phi = numpy.hstack((numpy.zeros((m,1)), M))
            #print 'mphi shape: ', m_phi.shape

        else:
            M = numpy.zeros((m, m * n_steps))
            for i in xrange(m): # xxx optimize
                for j in xrange(n_steps):
                    M[i, i + j*m] = vec[j]
            m_phi = numpy.hstack((numpy.zeros((m,m)), M))
        
        m_rew = M
        m_phi = (1-lam) * gam * m_phi
        return m_phi, m_rew

    def encode(self, S):
        return numpy.dot(S, self.theta)

    def lstd_weights(self, S, R, lam, gam, eps, sampled = True):

        if sampled is False:
            raise NotImplementedError

        n_steps = self._calc_n_steps(lam, gam, eps)
        m = S.shape[0]-n_steps-1
        assert m > 0
        
        Mphi, Mrew = self.get_mixing_matrices(m, lam, gam, sampled = sampled)
        PHI_full = self.encode(S)
        PHIlam = numpy.dot(Mphi, PHI_full)
        PHI0 = PHI_full[0:m,:]
        Rlam = numpy.dot(Mrew, R)

        a = numpy.dot(PHI0.T, (PHI0 - gam * PHIlam)) 
        if scipy.sparse.issparse(Rlam):
            b = Rlam.T.dot(PHI0).T
        else:
            b = numpy.dot(PHI0.T, Rlam)
        
        if a.ndim > 0:
            return numpy.linalg.solve(a,b) 
        return b/a

    def set_theta(self, theta):
        self.theta = self._reshape_theta(theta)

    def _reshape_theta(self, theta = None):
        if theta is None:
            theta = self.theta
        return numpy.reshape(theta, (self.n, self.k))


def plot_features(phi, r = None, c = None):
    
    plt.clf()
    j,k = phi.shape
    if r is None:
        r = c = numpy.round(numpy.sqrt(j))
        assert r*c == j
        
    m = int(numpy.floor(numpy.sqrt(k)))
    n = int(numpy.ceil(k/float(m)))
    assert m*n >= k 
    
    print 'n: ', n
    F = None
    for i in xrange(m):
        
        slic = phi[:,n*i:n*(i+1)]
        if i == 0:
            F = _stack_feature_row(slic, r, c)
        else:
            F = numpy.vstack((F, _stack_feature_row(slic, r, c)))
        
    
    F = F.astype(numpy.float64)
    v = 2*numpy.mean(numpy.abs(F)) # numpy.max(abs(F)) #
    plt.imshow(F, cmap='gray', interpolation = 'nearest', vmin = -v, vmax = v)
    plt.axis('off')    
    plt.colorbar()
        

def _stack_feature_row(phi_slice, r, c):
    
    for i in xrange(phi_slice.shape[1]):
    
        im = numpy.reshape(phi_slice[:,i], (r,c))
        I = 0.3*numpy.ones((r+2,c+2)) # pad with white value
        I[1:-1,1:-1] = im
        if i == 0:
            F = I
        else:
            F = numpy.hstack((F,I))
    return F
    
            

if __name__ == '__main__':
    test_theano_basis()

# vary: 
#   -regularization type/param
#   -partitions, basis size
#   -initialization (norming cols)
#   -weighting for loss and samples
#   -lambda parameter, set to zero
#   -test
#   -model data
#   -experiment suite - average over runs
#   -minibatch size, optimizer

# how to drive wall values to zero?
# regularization on the weights or the code?
# compare values of lambda
# use exact model learning instead of samples
# sgd, other fmins
# is weighting on errors correct?
