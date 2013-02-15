import numpy
import scipy.sparse
import theano
import theano.tensor as TT
import theano.sparse as TS
import theano.sandbox.linalg as TL
import matplotlib.pyplot as plt
import sirf

theano.gof.compilelock.set_lock_status(False)
theano.config.warn.sum_div_dimshuffle_bug = False
theano.config.on_unused_input = 'ignore'

logger = sirf.get_logger(__name__)

class BellmanBasis:

    LOSSES = 'bellman layered model reward covariance prediction nonzero l1code l1theta'.split()

    def __init__(self, n, ks, beta, thetas = None, w = None, reg_tuple = None,
                 nonlin = None, nonzero = None, shift = 1e-6):

        logger.info('building bellman basis')

        self.n = n # dim of data
        self.ks = ks # num of hidden layer features
        self.nonzero = nonzero  # set this to some positive float to penalize zero theta vectors.
        self.shift = shift # prevent singular matrix inversion with this pseudoinverse scalar.

        # params is a list of all learnable parameters in the model -- first the thetas, then w.
        params = []
        if thetas is None:
            thetas = [numpy.random.randn(a, b) for a, b in self.shapes]
            for theta in thetas:
                theta /= numpy.sqrt((theta * theta).sum(axis=0))
        else:
            # TODO: this no longer works for multiple layers.
            assert theta.shape == (k * n, ) or theta.shape == (self.n, self.k)
        params.extend(thetas)

        if w is None:
            w = numpy.random.randn(self.k + 1, 1)
            w = w / numpy.linalg.norm(w)
        else:
            assert w.shape == (self.k + 1, ) or w.shape == (self.k + 1, 1)
        params.append(w)

        self.set_params(params = params)

        # primitive theano vars
        self.param_names = ['theta-%d' % i for i in range(len(thetas))] + ['w']
        self.params_t = [TT.dmatrix(n) for n in self.param_names]

        self.S_t = TS.csr_matrix('S')
        self.Rfull_t = TS.csr_matrix('R')
        self.Mphi_t = TT.dmatrix('Mphi') # mixing matrix for PHI_lam
        self.Mrew_t = TT.dmatrix('Mrew') # mixing matrix for reward_lambda
        self.beta_t = theano.shared(beta) # multiplier on gamma set by env xxx

        # pick out a numpy and corresponding theano nonlinearity
        relu_t = lambda z: TT.maximum(0, z)
        relu = lambda z: numpy.clip(z, 0, numpy.inf)
        sigmoid_t = lambda z: 1. / (1 + TT.exp(-z))
        sigmoid = lambda z: 1. / (1 + numpy.exp(-z))
        ident = lambda z: z
        g, self.nonlin = dict(
            sigmoid=(sigmoid_t, sigmoid),
            relu=(relu_t, relu)).get(nonlin, (ident, ident))

        # encode s and mix lambda components
        z = self.S_t
        for i, t in enumerate(self.thetas_t):
            g(TS.structured_dot(z, t)) if i == 0 else g(TT.dot(z, t))
        self.PHI_full_t = TT.dot(z, self.w_t)
        self.PHIlam_t = TT.dot(self.Mphi_t, self.PHI_full_t)
        self.PHI0_t = self.PHI_full_t[:self.PHIlam_t.shape[0], :]
        self.Rlam_t = TS.structured_dot(self.Rfull_t.T, self.Mrew_t.T).T

        self.cov = TT.dot(self.PHI0_t.T, self.PHI0_t) + TS.square_diagonal(TT.ones((self.k, )) * self.shift)
        self.cov_inv = TL.matrix_inverse(self.cov) # l2 reg to avoid sing matrix

        # precompile theano functions and gradients.
        self.losses = dict(zip(self.LOSSES, [self.compile_loss(lo) for lo in self.LOSSES]))

        self.set_loss('bellman', ['theta-all'])
        self.set_regularizer(reg_tuple)

    @property
    def k(self):
        return self.ks[-1]

    @property
    def shapes(self):
        return zip([self.n] + self.ks[:-1], self.ks)

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

    def compile_loss(self, loss):
        kw = dict(on_unused_input='ignore')
        loss_t = getattr(self, '%s_funcs' % loss)()

        logger.info('compiling %s loss', loss)
        loss = theano.function(self.theano_vars, loss_t, **kw)

        grad_list = []
        for var in self.params_t:
            try: # theano doesn't like it when you take the gradient of wrt an irrelevant var
                grad_list.append(theano.function(self.theano_vars, theano.grad(loss_t, var), **kw))
            except ValueError: # function not differentiable wrt var, just return zeros
                grad_list.append(theano.function(self.theano_vars, TT.zeros_like(var), **kw))

        return loss, grad_list

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
        return sum(TT.sum(TT.abs_(t)) for t in self.params_t)

    def nonzero_funcs(self):
        '''Try to make sure feature weights are nonzero.'''
        l = lambda t: (1. / ((t * t).sum(axis=0) + 1e-10)).sum()
        return sum(l(t) for t in self.params_t)

    def bellman_funcs(self):
        ''' uses matrix inverse to solve for w'''
        # lstd weights for bellman error using normal eqns
        b = TT.dot(self.PHI0_t.T, self.Rlam_t)
        a = TT.dot(self.PHI0_t.T, self.PHI0_t - self.PHIlam_t) + TS.square_diagonal(TT.ones((self.k, )) * self.shift)
        w_lstd = TT.dot(TL.matrix_inverse(a), b)
        return TT.sum(TT.sqr(self.Rlam_t - TT.dot(self.PHI0_t - self.PHIlam_t, w_lstd)))

    def layered_funcs(self):
        ''' uses self.w_t when measuring loss'''
        # append const feature
        A = TT.horizontal_stack(self.PHI0_t - self.PHIlam_t, TT.ones((self.PHI0_t.shape[0], 1)))
        return TT.sum(TT.sqr(self.Rlam_t - TT.dot(A, self.w_t)))

    def reward_funcs(self):
        # reward loss: ||(PHI0 (PHI0.T * PHI0))^-1 PHI0.T * Rlam - Rlam||
        d = TT.dot(self.PHI0_t.T, self.Rlam_t)
        w_r = TT.dot(self.cov_inv, d)
        return TT.sum(TT.sqr(self.Rlam_t - TT.dot(self.PHI0_t, w_r))) # frobenius norm

    def model_funcs(self):
        # model loss: ||PHI0 (PHI0.T * PHI0)^-1 PHI0.T * PHIlam - PHIlam||
        bb = TT.dot(self.PHI0_t.T, self.PHIlam_t)
        w_m = TT.dot(self.cov_inv, bb) # least squares weight matrix
        return TT.sum(TT.sqr(self.PHIlam_t - TT.dot(self.PHI0_t, w_m))) # frobenius norm

    def covariance_funcs(self):
        # todo weight by stationary distribution if unsampled?
        return TT.sum(TT.abs_(self.cov - self.beta_t * TT.dot(self.PHI0_t.T,self.PHIlam_t)))

    def prediction_funcs(self):
        # next-step feature loss: || PHI0 PHI0.T Z - Z || where Z = [ R | PHIlam ]
        Z = TT.horizontal_stack(self.Rlam_t, self.PHIlam_t)
        A = TT.dot(self.PHI0_t, TT.dot(self.PHI0_t.T, Z)) - Z
        return TT.sum(TT.sqr(A)) # frobenius norm

    def loss(self, vec, S, R, Mphi, Mrew):
        params = self._unpack_params(vec)
        args = params + [S, R, Mphi, Mrew]
        loss =  self.loss_func(*args)
        #print 'loss pre reg: ', loss
        if self.reg_type is 'l1code':
            l1_loss, _ = self.losses['l1code']
            loss += self.reg_param * l1_loss(*args)
        if self.reg_type is 'l1theta':
            l1_loss, _ = self.losses['l1theta']
            loss += self.reg_param * l1_loss(*args)
        if self.nonzero:
            nz_loss, _ = self.losses['nonzero']
            loss += self.nonzero * nz_loss(*args)
        return loss / S.shape[0]

    def grad(self, vec, S, R, Mphi, Mrew):
        args = self._unpack_params(vec) + [S, R, Mphi, Mrew]
        gradient = numpy.zeros_like(vec)
        def add(grad, i):
            grad += self.loss_grads[i](*args)
            if self.reg_type is 'l1code':
                _, l1_grads = self.losses['l1code']
                grad += self.reg_param * l1_grads[i](*args)
            if self.reg_type is 'l1theta':
                _, l1_grads = self.losses['l1theta']
                grad += self.reg_param * l1_grads[i](*args)
            if self.nonzero:
                _, nz_grads = self.losses['nonzero']
                grad += self.nonzero * nz_grads[i](*args)
        for v in self.wrt:
            if 'all' in v:
                n = len(self.param_names)
                for i in range(n)[:n if v is 'all' else n - 1]:
                    add(gradient, i)
            else:
                try:
                    add(gradient, self.param_names.index(v))
                except ValueError:
                    logger.error('unknown variable name %r', v)
        return gradient

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

    def encode(self, S):
        z = S
        for t in self.thetas:
            z = self.nonlin(numpy.dot(z, t))
        return numpy.dot(z, self.w)

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

        a = numpy.dot(PHI0.T, PHI0 - gam * PHIlam)
        if scipy.sparse.issparse(Rlam):
            b = Rlam.T.dot(PHI0).T
        else:
            b = numpy.dot(PHI0.T, Rlam)

        return numpy.linalg.solve(a, b) if a.ndim > 0 else b / a

    def set_params(self, vec = None, params = None):
        if vec is not None:
            self.params = self._unpack_params(vec)
        if params is not None:
            self.params = [p.reshape(s) for s, p in zip(self.shapes, params)]

    def _unpack_params(self, vec):
        i = 0
        params = []
        for a, b in self.shapes:
            j = i + a * b
            params.append(vec[i:j].reshape((a, b)))
            i = j
        return params

    @property
    def flat_params(self):
        z = numpy.array([])
        for p in self.params:
            z = numpy.append(z, p.flatten())
        return z


def plot_features(phi, r = None, c = None):
    logger.info('plotting features')

    plt.clf()
    j,k = phi.shape
    if r is None:
        r = c = numpy.round(numpy.sqrt(j))
        assert r*c == j

    m = int(numpy.floor(numpy.sqrt(k)))
    n = int(numpy.ceil(k/float(m)))
    assert m * n >= k

    logger.info('plotting features, n: %d', n)
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
