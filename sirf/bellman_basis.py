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
from dbel import Model

class BellmanBasis:

    def __init__(self, n, k, beta, loss_type = 'bellman', theta = None,
            reg_tuple = None, partition = None, wrt = 'all'):
        
        self.n = n # dim of data
        self.k = k # num of features/columns
        self.loss_type = loss_type

        if theta is None: 
            theta = 1e-6 * numpy.random.standard_normal((self.n, self.k))
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

        # encode s and mix lambda components
        self.PHI_full_t = TS.structured_dot(self.S_t, self.theta_t)
        self.PHIlam_t = TT.dot(self.Mphi_t, self.PHI_full_t)
        self.PHI0_t = self.PHI_full_t[0:self.PHIlam_t.shape[0],:]
        self.Rlam_t = TS.structured_dot(self.Rfull_t.T, self.Mrew_t.T).T #xxx

        self.cov = TT.dot(self.PHI0_t.T, self.PHI0_t) 
        self.cov_inv = theano.sandbox.linalg.matrix_inverse(self.cov)

        # dictionary mapping loss funcs/grads to names
        self.d_losses = {
            'bellman': self.bellman_funcs, 'model': self.model_funcs, 
            'reward': self.reward_funcs, 'covariance':self.covariance_funcs}

        self.set_loss(loss_type, wrt)
        self.set_regularizer(reg_tuple)

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
            
            Lreg_grad = theano.grad(Lreg, [self.grad_var])
            self.reg_func = theano.function([self.theta_t, self.S_t, self.Mphi_t], Lreg)
            self.reg_grad = theano.function([self.theta_t, self.S_t, self.Mphi_t], Lreg_grad)
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

    def bellman_funcs(self):

        # lstd weights for bellman error using normal eqns
        b = TT.dot(self.PHI0_t.T, self.Rlam_t)
        a = TT.dot(self.PHI0_t.T, (self.PHI0_t - self.PHIlam_t))
        #w_lstd = theano.sandbox.linalg.solve(a,b) # solve currently has no gradient implemented
        w_lstd = TT.dot(theano.sandbox.linalg.matrix_inverse(a), b)
        e_be = self.Rlam_t - TT.dot((self.PHI0_t - self.PHIlam_t), w_lstd) # error vector
        Lbe = TT.sqrt(TT.sum(TT.sqr(e_be))) # need sqrt?
        Lbe_grad = theano.grad(Lbe, [self.grad_var])[0]

        loss_be = theano.function([self.theta_t, self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t], Lbe, on_unused_input='ignore')
        grad_be = theano.function([self.theta_t, self.S_t, self.Rfull_t, self.Mphi_t, self.Mrew_t], Lbe_grad, on_unused_input='ignore')

        return loss_be, grad_be

    def reward_funcs(self):
        
        # reward loss: ||PHI0 (PHI0.T * PHI0)^-1 PHI0.T * Rlam - Rlam||
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
        if self.reg_type is not None:
            loss += self.reg_param * self.reg_func(theta, S, R, Mphi, Mrew)
        return loss / (S.shape[0] * self.n) # norm loss by num samples and dim of data
        
    def grad(self, theta, S, R, Mphi, Mrew):
        
        theta = self._reshape_theta(theta)
        grad = self.loss_grad(theta, S, R, Mphi, Mrew)
        if self.reg_type is not None:
            grad += self.reg_param * self.reg_grad(theta, S, R, Mphi, Mrew)
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

    def get_mixing_matrices(self, m, lam, gam, sampled = True, eps = 1e-5):
        ''' returns a matrix for combining feature vectors from different time
        steps for use in TD(lambda) algorithms. m is the number of final
        samples/states, and n_steps is the number of extra time steps. Here we
        use only all m+n_step updates; slightly different than recursive TD
        algorithm, which uses all updates from all samples '''
        n_steps = self._calc_n_steps(lam, gam, eps = eps)
        vec = map(lambda i: (lam*gam)**i, xrange(n_steps)) # decaying weight vector
        if sampled:
            M = numpy.zeros((m, m + n_steps))
            for i in xrange(m): # for each row
                M[i, i:i+n_steps] = vec
            m_phi = numpy.hstack((numpy.zeros((m,1)), M))

        else:
            M = numpy.zeros((m, m * n_steps))
            for i in xrange(m): # xxx optimize
                for j in xrange(n_steps):
                    M[i, i + j*m] = vec[j]

            m_phi = numpy.hstack((numpy.zeros((m,m)), M))

        m_phi = (1-lam) * gam * m_phi
        m_rew = M
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

# TODO add bellman basis tests




def test_theano_basis(n = 81, k = 9, k_r = 1, lam = 0.95, mb_size = 5000, reconstructive = False,
        reg_type = 'l1-theta', reg_param = 0., max_iter = 2, weighting = 'policy', decay_eps = 1e-3, patience = 5):
    
    theano.config.warn.subtensor_merge_bug = False

    mdp = grid_world.MDP(walls_on = True)    
    #mdp.policy = OptimalPolicy(mdp.env, m)
    m = Model(mdp.env.R, mdp.env.P) 

    theta0 = 1e-8 * numpy.random.standard_normal(n*k)
    if reconstructive:
        theta0 = numpy.apply_along_axis(lambda v: v/numpy.linalg.norm(v), 0, theta0)

    b = Basis(n, k, theta = theta0, batch_size = mb_size, loss_type = 'bellman', 
       reconstructive = reconstructive , reg_type = reg_type, reg_param = reg_param, 
         wrt = 'all', partition = (['reward','model'],[k_r, k-k_r]))
    
    # sample the hold out test set
    #X = mdp.sample_grid_world(mb_size + n_time_steps + 1, distribution = weighting) 
    #numpy.random.shuffle(X)
    #X = scipy.sparse.csr_matrix(X[:,:-2], dtype = numpy.float64) # throw out the actions
    s_test, sp_test, r_test, _, = mdp.sample_grid_world(mb_size + n_time_steps, distribution = weighting)
    s_test = numpy.vstack((s_test, sp_test[-1,:]))
    #n_vars = (X.shape[1]-1)/2.
    #s_test = X[:,:n_vars]
    ##s_p_test = X[:,n_vars:-1]
    #r_test = X[:-1,-1] 

    print 'building mixing matrix'
    m_phi, m_rew = b.get_mixing_matrix(mb_size, n_time_steps, lam, m.gam)

    sample_be = numpy.zeros(1, dtype = numpy.float64)
    true_be = numpy.zeros(1, dtype = numpy.float64)
    true_lsq = numpy.zeros(1, dtype = numpy.float64)
    
    sample_be[0] = b.loss_be(b.theta, s_test, r_test, m_phi, m_rew)
    true_be[0] = m.bellman_error(b.theta, weighting = weighting)
    true_lsq[0] = m.value_error(b.theta, weighting = weighting)

    loss_types = ['reward', 'model', 'bellman']
    grad_parts = ['reward', 'model', 'all']
    

 
    cnt = 1
    for i,lt in enumerate(loss_types):
        
        print '\n \nusing %s loss, wrt %s' % (lt,grad_parts[i])
        print 'reconstructive: ', reconstructive
        b.set_loss(lt)
        b.set_grad(grad_parts[i])
        #running_avg = 1
        best_test_loss = numpy.inf
        n_test_inc = 0
    
        #while running_avg > train_eps:
        while n_test_inc < patience:
            
            print 'minibatch: ', cnt
            
            s, sp, r, _, = mdp.sample_grid_world(mb_size + n_time_steps + 1, distribution = weighting)
            s = numpy.vstack((s, sp[-1,:]))
            #X = mdp.sample_grid_world(mb_size + n_time_steps + 1, distribution = weighting) 
            #X = scipy.sparse.csr_matrix(X[:,:-2], dtype = numpy.float64) 
            #s = X[:,:n_vars]
            #r = X[:-1,-1]
            

            if cnt == 1:
                print 'initial sample loss: ', b.loss_be(b.theta, s, r, m_phi, m_rew)
                print 'initial true bellman error: ', m.bellman_error(b.theta)
            
            print 'performing minibatch optimization'
            theta_old = copy.deepcopy(b.theta)
            b.set_theta( fmin_cg(b.loss, b.theta.flatten(), b.grad,
                                    args = (s, r, m_phi, m_rew),
                                    full_output = False,
                                    maxiter = max_iter, 
                                    gtol = 1e-8) )

            delta = numpy.linalg.norm(b.theta - theta_old)
            print 'change in theta: ', delta
            #running_avg = running_avg * (1-alpha) + delta * alpha
            
            #print 'running average of delta', running_avg
            test_loss = b.loss(b.theta, s, r, m_phi, m_rew)#b.loss_be(b.theta, s_test, r_test, m_phi, m_rew)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_theta = b.theta
                n_test_inc = 0
                print 'new best'
            else:
                n_test_inc += 1
                print 'iters without better loss: ', n_test_inc

            sample_be = numpy.hstack((sample_be, test_loss)) # TODO add true bellman error for reconstructive feats
            true_be = numpy.hstack((true_be, m.bellman_error(b.theta, weighting = weighting)))
            true_lsq = numpy.hstack((true_lsq,  m.value_error(b.theta, weighting = weighting)))

            print 'sample BE loss: ', sample_be[cnt]
            print 'true BE : ', true_be[cnt], '\n'

            cnt += 1

        b.theta = best_theta

        #plot_features(b.theta)
        #plt.show()
    
    # plot basis functions
    plot_features(b.theta)
    plt.savefig('basis.k=%i.%s.pdf' % (k, weighting))
    
    # plot learning curve
    #plt.figure()
    plt.clf()
    ax = plt.axes()
    x = range(len(sample_be))
    ax.plot(x, sample_be/max(sample_be), 'r-', x, true_be/max(true_be), 'g-', x, true_lsq / max(true_lsq), 'b-')
    ax.legend(['Test BE','True BE','True RMSE'])
    plt.savefig('loss.k=%i.%s.pdf' % (k, weighting))

    # plot value functions, true and approx
    plt.clf()
    f = plt.figure()
    side = numpy.sqrt(n)
    
    # true model value fn
    f.add_subplot(311)
    plt.imshow(numpy.reshape(m.V, (side, side)), cmap = 'gray', interpolation = 'nearest')

    # bellman error estimate (using true model)
    f.add_subplot(312)
    basis = b.theta
    w_be = m.get_lstd_weights(basis)
    v_be = numpy.dot(basis, w_be)
    plt.imshow(numpy.reshape(v_be, (side, side)), cmap = 'gray', interpolation = 'nearest')
    
    # least squares solution with true value function
    f.add_subplot(313)
    w_lsq = numpy.linalg.lstsq(basis, m.V)[0]
    v_lsq = numpy.dot(basis, w_lsq)
    plt.imshow(numpy.reshape(v_lsq, (side, side)), cmap = 'gray', interpolation = 'nearest')
    plt.savefig('value.k=%i.%s.pdf' % (k, weighting))

def plot_features(phi, r = None, c = None):
 
    j,k = phi.shape
    if r is None:
        r = c = numpy.round(numpy.sqrt(j))
        assert r*c == j
        
    m = numpy.floor(numpy.sqrt(k))
    n = numpy.ceil(k/float(m))
    assert m*n >= k 

    f = plt.figure()
    for i in xrange(k):
        
        u = numpy.floor(i / m) 
        v = i % n
        
        im = numpy.reshape(phi[:,i], (r,c))
        ax = f.add_axes([float(u)/m, float(v)/n, 1./m, 1./n])
        ax.imshow(im, cmap = 'gray', interpolation = 'nearest')

    #plt.colorbar()
            

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
