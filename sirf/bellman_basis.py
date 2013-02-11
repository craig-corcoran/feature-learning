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
import sirf
import condor
import grid_world
from scipy.optimize import fmin_cg
from rl import Model

theano.gof.compilelock.set_lock_status(False)
theano.config.warn.sum_div_dimshuffle_bug = False
theano.config.on_unused_input = 'ignore'
        
logger = sirf.get_logger(__name__)

class BellmanBasis:

    LOSSES = 'bellman layered model reward covariance nonzero l1code l1theta'.split()
    RECORDABLE = 'test-bellman test-reward test-model'.split()# true-bellman true-lsq'.split()

    def __init__(self, n, k, beta, loss_type = 'bellman', theta = None, w = None,
                 reg_tuple = None, partition = None, wrt = ['theta-all','w'],
                 nonlin = None, nonzero = None, record_loss = None, shift = 1e-6):

        theano.gof.compilelock.set_lock_status(False)
        theano.config.warn.sum_div_dimshuffle_bug = False
        theano.config.on_unused_input = 'ignore'

        logger.info('building bellman basis')
        
        self.n = n # dim of data
        self.k = k # num of features/columns
        self.loss_type = loss_type
        self.nonzero = nonzero  # set this to some positive float to penalize zero theta vectors.
        self.shift = shift

        if theta is None: 
            theta = 1e-6 * numpy.random.standard_normal((self.n, self.k))
            theta /= numpy.sqrt((theta * theta).sum(axis=0))
            # initialize features sparsely
            #sparsity = 0.8
            #for i in xrange(self.k):
                #z = numpy.random.random(self.n)
                #theta[:,i][z < sparsity] = 0.
        if w is None:
            w = numpy.random.standard_normal((self.k,1))
            w = w / numpy.linalg.norm(w)
        else: assert (theta.shape == (k*n,)) or (theta.shape == (self.n, self.k))
        self.set_params(theta = theta, w = w)

        # partition the features for gradients
        self.d_partition = {}
        if partition is not None:       
            self.partition_theta(partition)
        else:
            self.theta_t = TT.dmatrix('theta')
            self.d_partition['theta-all'] = self.theta_t

        # primitive theano vars
        self.w_t = TT.dmatrix('w')
        self.d_partition['w'] = self.w_t # for derivatives wrt w

        self.S_t = theano.sparse.csr_matrix('S')
        self.Rfull_t = theano.sparse.csr_matrix('R')
        self.Mphi_t = TT.dmatrix('Mphi') # mixing matrix for PHI_lam
        self.Mrew_t = TT.dmatrix('Mrew') # mixing matrix for reward_lambda
        self.beta_t = theano.shared(beta) # multiplier on gamma set by env xxx

        # encode s and mix lambda components
        d_nonlin = dict(sigmoid=TT.nnet.sigmoid, relu=lambda z: TT.maximum(0, z))
        self.PHI_full_t = d_nonlin.get(nonlin, lambda z: z)(TS.structured_dot(self.S_t, self.theta_t))
        self.PHIlam_t = TT.dot(self.Mphi_t, self.PHI_full_t)
        self.PHI0_t = self.PHI_full_t[0:self.PHIlam_t.shape[0],:]
        self.Rlam_t = TS.structured_dot(self.Rfull_t.T, self.Mrew_t.T).T 

        self.cov = TT.dot(self.PHI0_t.T, self.PHI0_t) + TS.square_diagonal(TT.ones((k,)) * self.shift) 
        self.cov_inv = theano.sandbox.linalg.matrix_inverse(self.cov) # l2 reg to avoid sing matrix
        
        logger.info('compiling theano losses')

        # precompile theano functions and gradients.
        #self.losses = {lo: self.compile_loss(lo) for lo in self.LOSSES} # older pythons do not like
        self.losses = dict(zip(self.LOSSES, [self.compile_loss(lo) for lo in self.LOSSES]))

        self.set_loss(loss_type, wrt)
        self.set_regularizer(reg_tuple)
        self.set_recorded_loss(record_loss)
        
    @property
    def loss_be(self):
        return self.losses['bellman'][0]

    @property
    def loss_r(self):
        return self.losses['reward'][0]

    @property
    def loss_m(self):
        return self.losses['model'][0]

    @property
    def theano_vars(self):
        return [self.theta_t, self.w_t, self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t]
    
    #@property
    #def m_bellman_error(self):
        #return self.model.bellman_error
    
    #@property
    #def m_value_error(self):
        #return self.model.bellman_error

    @property
    def record_funs(self):
        return [self.loss_be , self.loss_r, self.loss_m] #, self.m_bellman_error, self.m_value_error]

    def set_recorded_loss(self, losses):   
        self.d_recordable = dict(zip(self.RECORDABLE, self.record_funs))
        self.d_loss_funcs = {}
        for lo in losses:
            self.d_loss_funcs[lo] = self.d_recordable[lo]
                

    def compile_loss(self, loss):
        kw = dict(on_unused_input='ignore')
        loss_t = getattr(self, '%s_funcs' % loss)()
        loss = theano.function(self.theano_vars, loss_t, **kw)

        grad_list = []
        for var in self.d_partition.itervalues():
            try: # theano doesn't like it when you take the gradient of wrt an irrelevant var
                grad_list.append(theano.function(self.theano_vars, theano.grad(loss_t, var), **kw))
            except ValueError: # function not differentiable wrt var, just return zeros 
                grad_list.append(theano.function(self.theano_vars, TT.zeros_like(var), **kw))    

        return loss, grad_list
            #theano.function(self.theano_vars, theano.grad(loss_t, var), **kw)
            #for var in self.d_partition.itervalues()]

    def partition_theta(self, partition):
        ''' creates partition of the parameters according to the partition sizes
        given in the partition tuple. ex: partition = {'reward':1, 'model':k-1} 
        '''
        
        print 'partitioning theta: ', partition.keys()
        names = partition.keys()
        indices = partition.values()

        assert sum(indices) == self.k
        part_inds = numpy.insert(numpy.cumsum(indices), 0, 0)
        self.d_part_inds = {}
        for i,n in enumerate(names):
            self.d_part_inds[n] = (part_inds[i], part_inds[i+1])

            self.d_partition[n] = theano.shared(
                        self.theta[:, part_inds[i]: part_inds[i+1]]) 
        
        part_list = map( lambda name: self.d_partition[name], names)
        self.theta_t = TT.horizontal_stack(*part_list) 
        self.d_partition['theta-all'] = self.theta_t

    def set_regularizer(self, reg_tuple):
        self.reg_type = reg_tuple
        if reg_tuple is not None:
            self.reg_type, self.reg_param = reg_tuple

    def set_loss(self, loss_type, wrt):
        self.loss_type = loss_type
        self.wrt = wrt
        self.loss_func, self.loss_grads = self.losses[loss_type]
        

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
        ''' uses matrix inverse to solve for w'''
        # lstd weights for bellman error using normal eqns
        b = TT.dot(self.PHI0_t.T, self.Rlam_t)
        
        a = TT.dot(self.PHI0_t.T, (self.PHI0_t - self.PHIlam_t)) + TS.square_diagonal(TT.ones((self.k,)) * self.shift) 
        #w_lstd = theano.sandbox.linalg.solve(a,b) # solve currently has no gradient implemented
        #A = theano.sandbox.linalg.matrix_inverse( TS.structured_add( \
        #         a, self.shift_t * TS.square_diagonal(TT.ones((self.k,))))) # l2 reg to avoid sing matrix
        #w_lstd = TT.dot(A, b)
        w_lstd = TT.dot(theano.sandbox.linalg.matrix_inverse(a), b)
        e_be = self.Rlam_t - TT.dot((self.PHI0_t - self.PHIlam_t), w_lstd) # error vector
        return TT.sqrt(TT.sum(TT.sqr(e_be)))


    def layered_funcs(self):
        ''' uses self.w_t when measuring loss'''
        e_be = self.Rlam_t - TT.dot((self.PHI0_t - self.PHIlam_t), self.w_t) # error vector
        return TT.sqrt(TT.sum(TT.sqr(e_be))) 
        
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

    def loss(self, vec, S, R, Mphi, Mrew):
        
        theta, w = self._unpack_params(vec)
        loss =  self.loss_func(theta, w, S, R, Mphi, Mrew)
        #print 'loss pre reg: ', loss
        if self.reg_type is 'l1code':
            l1_loss, _ = self.losses['l1code']
            loss += self.reg_param * l1_loss(theta, w, S, R, Mphi, Mrew)
        if self.reg_type is 'l1theta':
            l1_loss, _ = self.losses['l1theta']
            loss += self.reg_param * l1_loss(theta, w, S, R, Mphi, Mrew)
        if self.nonzero:
            nz_loss, _ = self.losses['nonzero']
            loss += self.nonzero * nz_loss(theta, w, S, R, Mphi, Mrew)
        return loss / (S.shape[0] * self.n) # norm loss by num samples and dim of data
        
    def grad(self, vec, S, R, Mphi, Mrew):
        
        theta, w = self._unpack_params(vec)
        
        th_grad = numpy.zeros_like(self.theta)
        w_grad = numpy.zeros_like(self.w)
        for i,v in enumerate(self.d_partition.keys()):
            if v in self.wrt:
                grad = self.loss_grads[i](theta, w, S, R, Mphi, Mrew)
                if self.reg_type is 'l1code':
                    _, l1_grads = self.losses['l1code']
                    grad += self.reg_param * l1_grads[i](theta, w, S, R, Mphi, Mrew)
                if self.reg_type is 'l1theta':
                    _, l1_grads = self.losses['l1theta']
                    grad += self.reg_param * l1_grads[i](theta, w, S, R, Mphi, Mrew)
                if self.nonzero:
                    _, nz_grads = self.losses['nonzero']
                    grad += self.reg_param * nz_grads[self.part_num](theta, w, S, R, Mphi, Mrew)

                if v == 'theta-all':
                    th_grad = grad
                    # cant take grad wrt to theta-all and theta-part at the same time
                    wrt_temp = copy.copy(self.wrt)
                    wrt_temp.remove(v)
                    assert not any(['theta' in s for s in wrt_temp])
                elif v == 'w':
                    w_grad = grad
                else:
                    j,k = self.d_part_inds[v]
                    th_grad[:,j:k] = grad

        grad = grad / (S.shape[0] * self.n) # norm grad by num samples and dim of data
        return numpy.append(th_grad.flatten(), w_grad.flatten())
    
    @classmethod
    def _calc_n_steps(self, lam, gam, eps):

        # calculate number of steps to perform lambda-averaging over
        if lam == 0:
            n_time_steps = 1
        else:
            n_time_steps = int(numpy.ceil(numpy.log(eps) / numpy.log(min(lam,gam))))

        return n_time_steps

    @classmethod
    def get_mixing_matrices(self, m, lam, gam, sampled = True, eps = 1e-5, dim = None):
        ''' returns a matrix for combining feature vectors from different time
        steps for use in TD(lambda) algorithms. m is the number of final
        rows/samples, dim is the dimension (equal to m and not needed if not
        sampled), and n_steps is the number of extra time steps. Here we use
        only all m+n_step updates; slightly different than recursive TD
        algorithm, which uses all updates from all samples '''

        n_steps = BellmanBasis._calc_n_steps(lam, gam, eps = eps)
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
        return numpy.dot(S, self.theta) # TODO add nonlin here

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

    def set_params(self, vec = None, theta = None, w = None):
        if vec is not None:
            theta, w = self._unpack_params(vec)
        if theta is not None:
            self.theta = self._reshape_theta(theta)
        if w is not None:
            self.w = numpy.reshape(w, (self.k, 1))

    def _reshape_theta(self, theta = None):
        if theta is None:
            theta = self.theta
        return numpy.reshape(theta, (self.n, self.k))

    def _unpack_params(self, vec):
        n_theta_vars = self.k*self.n
        theta = self._reshape_theta(vec[:n_theta_vars])
        w = numpy.reshape(vec[n_theta_vars:], (self.k, 1))
        return theta, w
    
    @property
    def flat_params(self):        
        return numpy.append(self.theta.flatten(), self.w.flatten())


def plot_features(phi, r = None, c = None):
    
    logger.info('plotting features')

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
