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

    def __init__(self, n, k, beta, loss_type = 'bellman', theta = None,
            reg_tuple = None, partition = None, wrt = 'all', shift = 1e-6, nonlin = None):
        
        self.n = n # dim of data
        self.k = k # num of features/columns
        self.loss_type = loss_type

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
        self.w_t = TT.dmatrix('w')
        self.d_partition['w'] = self.w_t # for derivatives wrt w

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

        # dictionary mapping loss funcs/grads to names
        self.d_losses = {
            'bellman': self.bellman_funcs, 'model': self.model_funcs, 
            'reward': self.reward_funcs, 'covariance':self.covariance_funcs}

        # init functions for tracking bellman error components
        self.grad_var = self.d_partition['all']
        self.loss_be, self.grad_be = self.d_losses['bellman']()
        self.loss_r, self.grad_r = self.d_losses['reward']()
        self.loss_m, self.grad_m = self.d_losses['model']()

        self.set_loss(loss_type, wrt)
        self.set_regularizer(reg_tuple)

    def unpack_params(self, vec):
        n_theta_vars = self.k*self.n
        self.set_theta(vec[:n_theta_vars])
        self.w = vec[n_theta_vars:]

    def pack_params(self, theta, w):        
        return numpy.append(theta.flatten(), w.flatten())
        

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
        if reg_tuple is not None:
            self.reg_type, self.reg_param = reg_tuple
            if self.reg_type == 'l1-code': # l1 loss on encoded features
                Lreg = TT.sum(TT.abs_(self.PHI_full_t)) 
            elif self.reg_type == 'l1-theta': # l1 loss on weights
                Lreg = TT.sum(TT.abs_(self.theta_t)) 
            else: 
                print self.reg_type 
                assert False
            
            Lreg_grad = theano.grad(Lreg, [self.grad_var])[0]
            self.reg_func = theano.function([self.theta_t, self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t], Lreg, on_unused_input='ignore')
            self.reg_grad = theano.function([self.theta_t, self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t], Lreg_grad, on_unused_input='ignore')
        else: self.reg_type = None

    def set_loss(self, loss_type, wrt):
        self.loss_type = loss_type
        self.wrt = wrt
        self.grad_var = self.d_partition[wrt]
        self.part_num = self.d_partition.keys().index(wrt)
        self.loss_be, self.grad_be = self.d_losses['bellman']() # rebuild bellman error?
        if (loss_type =='bellman'):
            self.loss_func = self.loss_be
            self.loss_grad = self.grad_be
        else:
            self.loss_func, self.loss_grad = self.d_losses[loss_type]()

    def closed_form_bellman_funcs(self):

        # lstd weights for bellman error using normal eqns
        b = TT.dot(self.PHI0_t.T, self.Rlam_t)
        a = TT.dot(self.PHI0_t.T, (self.PHI0_t - self.PHIlam_t)) 
        #w_lstd = theano.sandbox.linalg.solve(a,b) # solve currently has no gradient implemented
        #A = theano.sandbox.linalg.matrix_inverse( TS.structured_add( \
        #         a, self.shift_t * TS.square_diagonal(TT.ones((self.k,))))) # l2 reg to avoid sing matrix
        #w_lstd = TT.dot(A, b)
        w_lstd = TT.dot(theano.sandbox.linalg.matrix_inverse(a), b)
        e_be = self.Rlam_t - TT.dot((self.PHI0_t - self.PHIlam_t), w_lstd) # error vector
        Lbe = TT.sqrt(TT.sum(TT.sqr(e_be)))

        return Lbe

    def two_layer_bellman_funcs(self):
        e_be = self.Rlam_t - TT.dot((self.PHI0_t - self.PHIlam_t), self.w_t) # error vector
        Lbe = TT.sqrt(TT.sum(TT.sqr(e_be))) 
        return Lbe
        
        

    def reward_funcs(self):
        
        # reward loss: ||(PHI0 (PHI0.T * PHI0))^-1 PHI0.T * Rlam - Rlam||
        d = TT.dot(self.PHI0_t.T, self.Rlam_t)
        w_r = TT.dot(self.cov_inv, d)
        e_r = self.Rlam_t - TT.dot(self.PHI0_t, w_r)
        Lr = TT.sqrt(TT.sum(TT.sqr(e_r)))
        Lr_grad = theano.grad(Lr, [self.grad_var])[0]

        loss_r = theano.function([self.theta_t, self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t], Lr, on_unused_input='ignore')
        grad_r = theano.function([self.theta_t, self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t], Lr_grad, on_unused_input='ignore')

        return loss_r, grad_r

    def model_funcs(self):
        
        # model loss: ||PHI0 (PHI0.T * PHI0)^-1 PHI0.T * PHIlam - PHIlam|| 
        bb = TT.dot(self.PHI0_t.T, self.PHIlam_t)
        w_m = TT.dot(self.cov_inv, bb) # least squares weight matrix
        e_m = self.PHIlam_t - TT.dot(self.PHI0_t, w_m) # model error matrix
        Lm = TT.sqrt(TT.sum(TT.sqr(e_m))) # frobenius norm
        Lm_grad = theano.grad(Lm, [self.grad_var])[0]

        loss_m = theano.function([self.theta_t, self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t], Lm, on_unused_input='ignore')
        grad_m = theano.function([self.theta_t, self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t], Lm_grad, on_unused_input='ignore')

        return loss_m, grad_m

    def covariance_funcs(self):
        
        # todo weight by stationary distribution if unsampled?
        A = self.cov - self.beta_t * TT.dot(self.PHI0_t.T,self.PHIlam_t)
        Lc = TT.sum(TT.abs_(A))
        Lc_grad = theano.grad(Lc, [self.grad_var])[0]
        
        loss_c = theano.function([self.theta_t, self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t], Lc, on_unused_input='ignore')
        grad_c = theano.function([self.theta_t, self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t], Lc_grad, on_unused_input='ignore')

        return loss_c, grad_c

    def loss(self, theta, S, R, Mphi, Mrew):
        
        theta = self._reshape_theta(theta)
        loss =  self.loss_func(theta, S, R, Mphi, Mrew)
        #print 'loss pre reg: ', loss
        if self.reg_type is not None:
            loss += (self.reg_param * self.reg_func(theta, S, R, Mphi, Mrew))
            #print 'loss post reg: ', loss
        return loss / (S.shape[0] * self.n) # norm loss by num samples and dim of data
        
    def grad(self, theta, S, R, Mphi, Mrew):
        
        theta = self._reshape_theta(theta)
        grad = self.loss_grad(theta, S, R, Mphi, Mrew)
        if self.reg_type is not None:
            grad += (self.reg_param * self.reg_grad(theta, S, R, Mphi, Mrew))
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
