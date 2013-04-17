import numpy
import theano
import theano.tensor as TT
import theano.sparse as TS
import theano.sandbox.linalg as TL
import matplotlib.pyplot as plt
import scipy.sparse as sp
import sirf

theano.gof.compilelock.set_lock_status(False)
theano.config.warn.sum_div_dimshuffle_bug = False
theano.config.on_unused_input = 'ignore'

logger = sirf.get_logger(__name__)

# add method for setting mixing weights according to bellman gradient - only positive weights? how to initialize?
# add tests for new bases
# create environment object
# toy smoothness example - can we avoid local minima


class PredictionBasis(BellmanBasis):
        
    def __init__(self, n, ks, thetas = None, w = None, 
                 reg_tuple = None, nonlin = None, 
                 losses = None, gradients = None,
                 mixing_weights = (1., 1., 1.)):
        
        super(PredictionBasis, self).__init__(self, n, ks, thetas, w, 
                                            reg_tuple, nonlin, 
                                            losses, gradients)

        self.mix_wts_t = theano.shared(numpy.array(mixing_weights, dtype = numpy.float64))

        # Thetas U V q w 
        self.param_names = ['theta-%d' % i for i in range(len(ks))] + ['U', 'V', 'q', 'w']
        self.params_t = [TT.dmatrix(n) for n in self.param_names]
    
    @property
    def shapes(self):
        return zip([self.n] + self.ks + [self.k]*3, self.ks + [self.k, self.n] + [1]*2)

    @property
    def U(self):
        return self.params[-4]

    @property
    def U_t(self):
        return self.params_t[-4]

    @property
    def V(self):
        return self.params[-3]

    @property
    def V_t(self):
        return self.params_t[-3]

    @property
    def q(self):
        return self.params[-2]

    @property
    def q_t(self):
        return self.params_t[-2]

    @property
    def thetas(self):
        return self.params[:-4]

    @property
    def thetas_t(self):
        return self.params_t[:-4]
    
    # TODO want (tied) reconstruction_calc

    def prediction_calc(self):
        return (self.mix_wts_t[0] * self.raw_prediction_calc() +
                self.mix_wts_t[1] * self.state_prediction_calc() + 
                self.mix_wts_t[2] * self.reward_prediction_calc())

    def tied_prediction_calc(self):
        return (self.mix_wts_t[0] * self.tied_raw_prediction_calc() +
                self.mix_wts_t[1] * self.state_prediction_calc() + 
                self.mix_wts_t[2] * self.reward_prediction_calc())

    def state_prediction_calc(self):
        # || Phi(X) U - Phi(X') ||, U is k(+1) by k
        return TT.sum(TT.abs_( TT.dot(self.PHI0_t, self.U_t ) - self.PHIlam_t))

    def reward_prediction_calc(self):
        # || Phi(X) q - Rlam ||, q is k(+1) by 1
        return TT.sum(TT.abs_( TT.dot(self.PHI0_t, self.q_t ) - self.PHIlam_t))

    def raw_prediction_calc(self):
        # || Phi(X) V - X' ||, V is k(+1) by n
        return TT.sum(TT.abs_( TT.dot(self.PHI0_t, self.V_t ) - self.Slam_t)) # TODO need Slam_t

    def tied_raw_prediction_calc(self):
        # || decode(encode(X)) - X' ||
        return TT.sum(TT.abs_(self.decode(self.PHI0_t) - self.Slam_t))

    def raw_reconstruction_calc(self):
        # || Phi(X) V - X ||
        return TT.sum(TT.abs_( TS.sub(self.S0_t, TT.dot(self.PHI0_t, self.V_t))))

    def tied_raw_reconstruction_calc(self):
        # || decode(encode(X)) - X ||
        return TT.sum(TT.abs_(TS.sub(self.S0_t, self.decode(self.PHI0_t))))



class BellmanBasis(object):

    def __init__(self, n, ks, thetas = None, w = None, 
                 reg_tuple = None, nonlin = None, 
                 losses = None, gradients = None):
        
        self.n = n - 1 # dim of input data, not encluding bias, so (a+1,b) can be used on all layers
        self.ks = ks # num of hidden layer features
        self.shift = 1e-6 # prevent singular matrix inversion with this pseudoinverse scalar.
        
        logger.info('building bellman basis: %s' %
                    ':'.join('%dx%d' % x for x in self.shapes))

        # params is a list of all learnable parameters in the model -- first the thetas, then w.
        params = []
        if thetas is None:
            thetas = [1e-6 * numpy.random.randn(a + 1, b) for a, b in self.shapes[:-1]]
        else: # make sure correctly shaped parameters are passed in 
            for i, theta in enumerate(thetas):
                a,b = self.shapes[i]
                assert theta.shape == (a+1, b)
        params.extend(thetas)
        
        # XXX fold w into initialization above
        if w is None:
            w = numpy.random.randn(self.k + 1, 1)
            #w = w / numpy.linalg.norm(w)
        else:
            assert w.shape == (self.k + 1, ) or w.shape == (self.k + 1, 1)
        params.append(w)
        # self.set_params(params = params)

        # primitive theano vars
        self.param_names = ['theta-%d' % i for i in range(len(thetas))] + ['w']
        self.params_t = [TT.dmatrix(n) for n in self.param_names]

        self.Sfull_t = TS.csr_matrix('S')
        self.Rfull_t = TS.csr_matrix('R')
        self.Mphi_t = TT.dmatrix('Mphi') # mixing matrix for PHI_lam
        self.Mrew_t = TT.dmatrix('Mrew') # mixing matrix for reward_lambda

        # pick out a numpy and corresponding theano nonlinearity
        relu_t = lambda z: TT.maximum(0, z)
        relu = lambda z: numpy.clip(z, 0, numpy.inf)
        sigmoid_t = lambda z: 1. / (1 + TT.exp(-z))
        sigmoid = lambda z: 1. / (1 + numpy.exp(-z))
        ident = lambda z: z
        self.nonlin_t, self.nonlin = dict(
            sigmoid=(sigmoid_t, sigmoid),
            relu=(relu_t, relu)
            ).get(nonlin, (ident, ident))

        # encode s and mix lambda components
        self.PHI_full_t = self.encode(self.S_t) 
        self.S0_t = self.Sfull_t[:self.MPhi_t.shape[0],:] # still sparse?
        self.Slam_t = TS.structured_dot(self.Sfull_t.T, self.Mphi_t.T).T
        self.Rlam_t = TS.structured_dot(self.Rfull_t.T, self.Mrew_t.T).T
        # 'c' appended to variable name implies constant bias column appended
        self.PHIlam_t = TT.dot(self.Mphi_t, self.PHI_full_t)
        self.PHIlamc_t = self.stack_bias(self.PHIlam_t)
        self.PHI0_t = self.PHI_full_t[:self.PHIlam_t.shape[0],:]
        self.PHI0c_t = self.stack_bias(self.PHI0_t)

        # create symbolic w_lstd, value fn
        self.A_t = self.stack_bias(self.PHI0_t - self.PHIlam_t) # (PHI0 - PHIlam)|e 
        self.b_t = TT.dot(self.PHI0c_t.T, self.Rlam_t)
        a = TT.dot(self.PHI0c_t.T, self.A_t) + TT.eye(self.k + 1) * self.shift
        self.w_lstd_t = TT.dot(TL.matrix_inverse(a), self.b_t) # includes bias param
        self.v_t = TT.dot(self.PHI0c_t, self.w_t)
    
        # create covariance matrices and inverse used in losses
        self.lam_cov_t = TT.dot(self.PHI0_t.T, self.PHIlam_t)  
        self.lam_covc_t = TT.dot(self.PHI0c_t.T, self.PHIlamc_t)  
        self.cov_t = TT.dot(self.PHI0_t.T, self.PHI0_t)  # no bias
        self.covc_t = TT.dot(self.PHI0c_t.T, self.PHI0c_t)  # includes bias
        self.covc_inv_t = TL.matrix_inverse(self.covc_t + self.shift * TT.eye(self.k + 1)) # l2 reg to avoid singular matrix
    
        # precompile theano functions and gradients.
        self.set_regularizer(reg_tuple)
        if losses:
            if self.reg_type:
                losses.append(self.reg_type)
            self.losses = dict(zip(losses, [self.compile_loss(lo) for lo in losses]))
        else:
            assert False # XXX add default losses

        if gradients:
            if self.reg_type:
                gradients.append(self.reg_type)
            self.gradients = dict(zip(gradients, [self.compile_gradient(gr) for gr in gradients]))
        else:
            assert False # XXX

        self.set_loss('bellman')
        

    @staticmethod
    def stack_bias_t(x): # TODO make sparse matrix compatible
        return TT.concatenate([x, TT.ones((x.shape[0], 1))], axis = 1)

    @staticmethod
    def stack_bias(x): # TODO make sparse matrix compatible
        if len(x.shape) == 1:
            x = x.reshape((1, len(x)))
        return numpy.hstack([x, numpy.ones((len(x), 1))]) # what if S is sparse? xxx

    @property
    def k(self):
        return self.ks[-1]

    @property
    def shapes(self):
        return zip([self.n] + self.ks, self.ks + [1])

    @property
    def loss_be(self):
        return self.losses['bellman'][0]

    @property
    def loss_lsbe(self):
        return self.losses['ls_bellman'][0]

    @property
    def loss_r(self):
        return self.losses['reward'][0]

    @property
    def loss_m(self):
        return self.losses['model'][0]

    @property
    def loss_fm(self):
        return self.losses['fullmodel'][0]

    @property
    def w(self):
        return self.params[-1]

    @property
    def w_t(self):
        return self.params_t[-1]

    @property
    def thetas(self):
        return self.params[:-1]

    @property
    def thetas_t(self):
        return self.params_t[:-1]

    @property
    def theano_vars(self):
        return self.params_t + [self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t]

    @property
    def flat_params(self):
        z = numpy.array([])
        for p in self.params:
            z = numpy.append(z, p.flatten())
        return z

    def decode_t(self, PHI0_t):
        z = PHI0_t
        for t in self.thetas_t[::-1][:-1]: # all but the first layer
            z = self.nonlin_t(TT.dot(z, t.T)[:, :-1]) # remove bias activation column
        return TT.dot(z, self.thetas_t[0].T)[:, :-1] # returns result w/o bias

    def decode(self, PHI0):
        z = PHI0
        for t in self.thetas[::-1][:-1]: # all but the first layer
            z = self.nonlin(numpy.dot(z, t.T)[:, :-1]) # remove bias activation column
        return numpy.dot(z, self.thetas[0].T)[:, :-1] # returns result w/o bias

    def encode_t(self, S_t):
        # assumes input S_t is sparse and already includes a bias column
        z = self.nonlin_t(TS.structured_dot(self.S_t, self.thetas_t[0])) 
        for t in self.thetas_t[1:]:
            z = self.nonlin_t(TT.dot(self.stack_bias(z), t))
        return z

    def encode(self, S):
        if sp.issparse(S):
            z = self.nonlin(sp.dot(S, self.thetas[0]))
        else:
            z = self.nonlin(numpy.dot(S, self.thetas[0]))
        for t in self.thetas[1:]:
            z = self.stack_bias(z)
            z = self.nonlin(numpy.dot(z, t))
        return z

    def estimated_value(self, state, add_bias = True):
        '''Compute the estimated value of a given world state or matrix of states.'''
        def bias(z):
            if len(z.shape) == 1:
                z = z.reshape((1, len(z)))
            return numpy.hstack([z, numpy.ones((len(z), 1))]) # what if S is sparse? xxx
        return numpy.dot(bias(self.encode(state, add_bias)), self.w)


    def compile_loss(self, loss):
        loss_t = getattr(self, '%s_calc' % loss)()

        logger.info('compiling %s loss' % loss)
        loss = theano.function(self.theano_vars, loss_t, on_unused_input='ignore')

        return loss, loss_t

    def compile_grad(self, grad):
        kw = dict(on_unused_input='ignore')

        loss_tuple = self.losses.get(grad)
        if loss_tuple is None: # if the fn hasn't been compiled yet (which shouldn't happen?)
            loss_tuple = self.compile_loss(grad)
        loss_t = loss_tuple[1]

        grad_list = []
        for var in self.params_t:
            try: # theano doesn't like it when you take the gradient wrt an irrelevant var
                grad_list.append(theano.function(self.theano_vars, theano.grad(loss_t, var), **kw))
            except ValueError: # function not differentiable wrt var, just return zeros
                grad_list.append(theano.function(self.theano_vars, TT.zeros_like(var), **kw))

        return grad_list
        
    def set_regularizer(self, reg_tuple):
        self.reg_type = reg_tuple
        if reg_tuple is not None:
            self.reg_type, self.reg_param = reg_tuple

    def set_loss(self, loss_type, wrt = 'all'):
        self.loss_type = loss_type
        self.wrt = wrt
        self.loss_func, _ = self.losses[loss_type]
        self.loss_grads = self.gradients[loss_type]

    def l2code_calc(self):
        return TT.sqrt(TT.sum(TT.sqr(self.PHI_full_t)))

    def l1code_calc(self):
        '''regularize the size of the feature values after encoding.'''
        return TT.sum(TT.abs_(self.PHI_full_t))

    def l1theta_calc(self):
        '''regularize the size of the feature weights.'''
        return sum(TT.sum(TT.abs_(t)) for t in self.params_t)

    def ls_bellman_calc(self):
        ''' uses matrix inverse (least squares/normal equations) to solve for w'''
        # lstd weights for bellman error using normal eqns
        return TT.sqrt(TT.sum(TT.sqr(self.Rlam_t - TT.dot(self.A_t, self.w_lstd_t))))

    def bellman_calc(self):    
        ''' uses self.w_t when measuring loss'''
        return TT.sqrt(TT.sum(TT.sqr(self.Rlam_t - TT.dot(self.A_t, self.w_t))))

    def ls_reward_calc(self):
        # reward loss: ||(PHI0 (PHI0.T * PHI0))^-1 PHI0.T * Rlam - Rlam||
        w_r = TT.dot(self.covc_inv_t, self.b_t)
        return TT.sqrt(TT.sum(TT.sqr(self.Rlam_t - TT.dot(self.PHI0c_t, w_r)))) # frobenius norm

    def ls_fullmodel_calc(self):
        # model loss: ||PHI0 (PHI0.T * PHI0)^-1 PHI0.T * PHIlam - PHIlam||
        B = TT.dot(self.PHI0c_t.T, self.PHIlam_t) 
        W_m = TT.dot(self.covc_inv_t, B) # least squares weight matrix
        return TT.sqrt(TT.sum(TT.sqr(self.PHIlam_t - TT.dot(self.PHI0c_t, W_m)))) # frobenius norm

    def ls_model_calc(self):
        v = TT.dot(self.PHIlamc_t, self.w_lstd_t)
        b = TT.dot(self.PHI0c_t.T, v)
        w_m = TT.dot(self.covc_inv_t, b) # least squares weight matrix
        return TT.sqrt(TT.sum(TT.sqr(v - TT.dot(self.PHI0c_t, w_m)))) # frobenius norm
        

    def loss(self, vec, S, R, Mphi, Mrew):
        args = self._unpack_params(vec) + [S, R, Mphi, Mrew]

        loss =  self.loss_func(*args)

        #print 'loss pre reg: ', loss
        if self.reg_type: 
            reg_loss, _ = self.losses[self.reg_type]
            loss += self.reg_param * reg_loss(*args)

        return loss / S.shape[0]

    def grad(self, vec, S, R, Mphi, Mrew):
        args = self._unpack_params(vec) + [S, R, Mphi, Mrew]

        grad = numpy.zeros_like(vec)
        o = 0
        for i, (var, (a, b)) in enumerate(zip(self.param_names, self.shapes)):
            if (var == self.wrt) or (self.wrt == 'all'):
                sl = slice(o, o + (a + 1) * b)
                grad[sl] += self.loss_grads[i](*args).flatten()
                if self.reg_type: 
                    reg_grad = self.gradients[self.reg_type]
                    grad[sl] += self.reg_param * reg_grad[i](*args).flatten()
            o += (a + 1) * b

        return grad / S.shape[0]

    def set_params(self, vec = None, params = None):
        if vec is not None:
            self.params = self._unpack_params(vec)
        if params is not None:
            logger.info('setting params %s' % ', '.join(str(p.shape) for p in params))
            self.params = [p.reshape((a + 1, b)) for (a, b), p in zip(self.shapes, params)]

    def _unpack_params(self, vec):
        i = 0
        params = []
        for a, b in self.shapes:
            j = i + (a + 1) * b
            params.append(vec[i:j].reshape((a + 1, b)))
            i = j
        return params
    
    @staticmethod
    def _calc_n_steps(lam, gam, eps):
        # calculate number of steps to perform lambda-averaging over
        return 1 if lam == 0 else int(numpy.ceil(numpy.log(eps) / numpy.log(min(lam, gam))))

    @staticmethod
    def get_mixing_matrices(m, lam, gam, sampled = True, eps = 1e-5):
        ''' returns a matrix for combining feature vectors from different time
        steps for use in TD(lambda) algorithms. m is the number of final
        rows/samples, and n_steps is the number of extra time steps. Here we use
        only all m+n_step updates; slightly different than recursive TD
        algorithm, which uses all updates from all samples '''

        n_steps = BellmanBasis._calc_n_steps(lam, gam, eps = eps)
        vec = map(lambda i: (lam*gam)**i, xrange(n_steps)) # decaying weight vector
        if sampled:
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
    
    @staticmethod
    def lstd_weights(self, PHI_full, R, Mphi, Mrew):
        PHIlam = numpy.dot(Mphi, PHI_full)
        PHI0 = PHI_full[0:m,:]
        Rlam = numpy.dot(Mrew, R)
        A = numpy.dot(PHI0.T, (PHI0 - PHIlam)) 
        A += self.shift * numpy.eye(A.shape[0])
        b = numpy.dot(PHI0.T, Rlam)
        if A.ndim > 0:
            return numpy.linalg.solve(A,b) 
        return numpy.array(b/A)


class KrylovBasis(BellmanBasis):

    def __init__(self, n, ks, thetas = None, w = None, 
                 reg_tuple = None, nonlin = None, 
                 losses = None, gradients = None, 
                 rew_wt = 1.):
            
        self.rew_wt = rew_wt # relative weighting of reconstructing reward vs. predicting features
        wvec = numpy.insert(numpy.ones(self.ks[-1]), 0, self.rew_wt)
        self.ReWt_t = TS.square_diagonal(theano.shared(wvec))
        self.Z_t = TT.concatenate([self.Rlam_t, self.PHIlam_t], axis=1)

        super(KrylovBasis, self).__init__(self, n, ks, thetas, w, 
                                            reg_tuple, nonlin, 
                                            losses, gradients)
    
    def krylov_calc(self, norm_cols = True):
        # next-step feature loss: || PHI0 PHI0.T Z - Z || where Z = [ R | PHIlam ]
        
        if norm_cols: 
            Z = TT.true_div(self.Z_t,  TT.sqrt(TT.sum(TT.sqr(self.Z_t), axis=0)))
        else:
            Z = self.Z_t
        
        A = TT.dot(self.PHI0_t, TT.dot(self.PHI0_t.T, Z)) - Z
        B = TS.structured_dot(self.ReWt_t, A.T).T
        return TT.sqrt(TT.sum(TT.sqr(B))) # frobenius norm

    def value_krylov_calc(self, norm_cols = False):
        
        q = TT.dot(self.PHIlamc_t, self.w_t)
        r = self.Rlam_t
        if norm_cols:
            q = TT.true_div(q,  TT.sqrt(TT.sum(TT.sqr(q))))
            r = TT.true_div(self.Rlam_t,  TT.sqrt(TT.sum(TT.sqr(self.Rlam_t))))
            
        q_err = TT.dot(self.PHI0_t, TT.dot(self.PHI0_t.T, q)) - q
        r_err = TT.dot(self.PHI0_t, TT.dot(self.PHI0_t.T, r)) - r
        return TT.sum(TT.abs_(q_err)) + TT.sum(TT.abs_(r_err)) # currently no relative weighting

    
    def reward_krylov_calc(self, norm_cols = True):
        if norm_cols:
            r = TT.true_div(self.Rlam_t,  TT.sqrt(TT.sum(TT.sqr(self.Rlam_t), axis=0)))
        else:
            r = self.Rlam_t
        
        A = TT.dot(self.PHI0_t, TT.dot(self.PHI0_t.T, r)) - r
        return TT.sqrt(TT.sum(TT.sqr(A))) # frobenius norm


class LaplacianBasis(BellmanBasis):
    ''' basis for training on the Bellman error with Laplacian loss/regularization''' 

    def __init__(self, n, ks, thetas = None, w = None, 
                 reg_tuple = None, nonlin = None, 
                 losses = None, gradients = None, 
                 lap_reg = 1., gam_ratio = 1.):

        self.lap_reg = lap_reg
        self.gam_ratio = gam_ratio

        super(LaplacianBasis, self).__init__(self, n, ks, thetas, w, 
                                            reg_tuple, nonlin, 
                                            losses, gradients)

    def covariance_calc(self): # TODO why is sum needed?
        return TT.sum(TT.dot(TT.dot(self.w_t.T, 
            (self.covc_t - self.gam_ratio * self.lam_covc_t)), self.w_t)) 

    def full_covariance_calc(self):
        # | PHI0.T (I - P) PHI0 | 
        # TODO weight by stationary distribution if unsampled?
        # normalize columns?
        return TT.sum(TT.abs_(self.cov_t - self.gam_ratio * self.lam_cov_t)) # l1 matrix norm

    def full_laplacian_calc(self):
        # | PHI0.T (I - P) PHI0 | + eta * BE
        return self.full_covariance_calc() + self.lap_reg * self.bellman_calc() 

    def laplacian_calc(self):
        # v_hat.T (I - P) v_hat + eta * BE
        return self.covariance_calc() + self.lap_reg * self.bellman_calc()

