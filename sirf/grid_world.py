import copy
import numpy
import scipy.sparse
import scipy.optimize
from random import choice
import matplotlib.pyplot as plt

class GridWorld:
    ''' Grid world environment. State is represented as an (x,y) array and 
    state transitions are allowed to any (4-dir) adjacent state, excluding 
    walls. When a goal state is reached, a reward of 1 is given and the state
    is reinitialized; otherwise, all transition rewards are 0.

    '''

    _vecs = numpy.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]]) # 4 movement dirs

    def __init__(self, wall_matrix, goal_matrix, init_state = None, uniform = False):
        
        self.walls = wall_matrix
        self.goals = goal_matrix
        goal_list = zip(*self.goals.nonzero())
        # TODO assert walls and goals are correct format

        self.n_rows, self.n_cols = self.walls.shape
        self.n_states = self.n_rows * self.n_cols

        self._adjacent = {}
        self._actions = {}

        if init_state is None:
            self.state = self.random_state()
        else:
            assert self._check_valid_state(init_state)
            self.state = init_state

        # precompute adjacent state and available actions given wall layout
        for i in xrange(self.n_rows):
            for j in xrange(self.n_cols):
                if self._check_valid_state((i,j)):
                    
                    # bc this is a valid state, add it to the possible states goals can transition to
                    for g in goal_list:
                        
                        adj = self._adjacent.get(g)
                        if adj is None:
                            self._adjacent[g] = set()

                        self._adjacent[g].add((i,j)) 
                    
                    # check all possible actions and adjacent states
                    for v in self._vecs:
                        
                        pos = numpy.array([i,j]) + v
                        if self._check_valid_state(pos):

                                act_list = self._actions.get((i,j))
                                adj_list = self._adjacent.get((i,j))
                                if adj_list is None:
                                    self._adjacent[(i,j)] = set()
                                if act_list is None:
                                    self._actions[(i,j)] = set()
                                
                                pos = tuple(pos)
                                #if self._adjacent.get(pos) is None:
                                    #self._adjacent[pos] = set()
                                #self._adjacent[pos].add((i,j))

                                self._adjacent[(i,j)].add(pos)
                                self._actions[(i,j)].add(tuple(v))

        # form transition matrix P 
        P = numpy.zeros((self.n_states, self.n_states))

        for state, adj_set in self._adjacent.items():

            idx = self.state_to_index(state)
            adj_ids = map(self.state_to_index, adj_set)
            P[adj_ids,idx] = 1
            #P[idx, adj_ids] = 1

        # normalize columns to have unit sum
        self.P = numpy.dot(P, numpy.diag(1./(numpy.sum(P, axis=0)+1e-14)))
        
        # build reward function R
        self.R = numpy.zeros(self.n_states)
        nz = zip(*self.goals.nonzero())
        gol_ids = map(self.state_to_index, nz)
        self.R[gol_ids] = 1
        
        if uniform:
            self.D = scipy.sparse.dia_matrix(([1]*self.n_states, 0), \
                (self.n_states, self.n_states))
            assert self.D == scipy.sparse.csc_matrix(numpy.eye(self.n_states))
        else:
            # find limiting distribution
            v = numpy.zeros((self.P.shape[0],1))
            v[1,0] = 1
            delta = 1
            while  delta > 1e-12:
                v_old = copy.deepcopy(v)
                v = numpy.dot(self.P,v)
                v = v / numpy.linalg.norm(v)
                delta = numpy.linalg.norm(v-v_old)
             
            self.D = scipy.sparse.dia_matrix((v[:,0],0),(self.n_states, self.n_states))
            
            #db
            #plot_im(numpy.reshape(v[:,0], (9,9)))
            #plt.show()

            

    def _check_valid_state(self, pos):
        ''' Check if position is in bounds and not in a wall. '''
        if pos is not None:
            if (pos[0] >= 0) & (pos[0] < self.n_rows) \
                    & (pos[1] >= 0) & (pos[1] < self.n_cols):
                if (self.walls[pos[0], pos[1]] != 1):
                    return True

        return False


    def get_actions(self, state):
        ''' return available actions as a list of length-2 arrays '''
        if type(state) == tuple:
            return self._actions[state]
        elif type(state) == numpy.ndarray:
            return self._actions[tuple(state.tolist())]
        else:
            assert False

    def next_state(self, action):
        ''' return (sampled / deterministic) next state given the current state 
        without changing the current state '''

        # if at goal position, reinitialize to rand state
        if self.goals[tuple(self.state)] == 1:
            return self.random_state()

        pos = self.state + action
        if self._check_valid_state(pos):
            return pos
        else:
            return self.state
        
    def take_action(self, action):
        '''take the given action, if valid, changing the state of the 
        environment. Return resulting state and reward. '''
        rew = self.get_reward(self.state)
        self.state = self.next_state(action)
        return self.state, rew


    def get_reward(self, state):
        ''' sample reward function for a given afterstate. Here we assume that 
        reward is a function of the subsequent state only, not the previous 
        state and action '''
        if self.goals[tuple(self.state)] == 1:
            return 1
        else:
            return 0

    def state_to_index(self, state):
        return state[0] * self.n_cols + state[1]

    def random_state(self):

        r_state = None
        while not self._check_valid_state(r_state):
            r_state = numpy.round(numpy.random.random(2) * self.walls.shape)\
                        .astype(numpy.int)
        
        return r_state

class RandomPolicy:

    def choose_action(self, actions):
        return choice(list(actions))

class MDP:
    
    def __init__(self, environment, policy):
        self.env = environment
        self.policy = policy
        
    def sample(self, n_samples):
        ''' sample the interaction of policy and environment for n_samples, 
        returning arrays of state positions and rewards '''
        
        states = numpy.zeros((n_samples+1,2), dtype = numpy.int)
        states[0] = self.env.state
        rewards = numpy.zeros(n_samples, dtype = numpy.int)
        
        for i in xrange(n_samples):
            actions = self.env.get_actions(self.env.state)
            action = self.policy.choose_action(actions)
            next_state, reward = self.env.take_action(action)
            assert self.env.walls[next_state[0], next_state[1]] == 0
            states[i+1] = next_state
            rewards[i] = reward
         
        return states, rewards

class SIRFObjectiveFn:
    
    def __init__(self, W, rewards, states, states_p, env, rew_wt, mod_wt, reg_wt, \
                rew_avg = None):
        
        self.W = scipy.sparse.csc_matrix(W)
        self.rew_wt = rew_wt
        self.mod_wt = mod_wt
        self.reg_wt = reg_wt

        
        self.R = scipy.sparse.csc_matrix(rewards).T

        if rew_avg is None:
            avg = numpy.mean(rewards)
        else:
            avg = rew_avg
        rewards = rewards - avg # worth loss is sparsity to center?

        if len(rewards.nonzero()[0]) is not 0: 
            rewards = rewards / numpy.linalg.norm(rewards)
        self.R_norm = scipy.sparse.csc_matrix(rewards).T # sparse row vector

        self.n_samples = max(rewards.shape)
        self.n_states, self.n_features = W.shape
        
         #create tabular feature data matrices given the sampled states
        #X_all = scipy.sparse.lil_matrix((self.n_samples+1, self.n_states), dtype = numpy.int)
        #for j in xrange(self.n_samples+1):
            #X_all[j, env.state_to_index(states[j])] = 1
        #X_all[:,-1] = 1 # append constant vector
        #X_all = scipy.sparse.csc_matrix(X_all)
        
        # create tabular feature data matrices given the sampled states - new
        X = scipy.sparse.lil_matrix((self.n_samples, self.n_states), dtype = numpy.int)
        X_p = scipy.sparse.lil_matrix((self.n_samples, self.n_states), dtype = numpy.int)
        for j in xrange(self.n_samples):
            X[j, env.state_to_index(states[j])] = 1
            X_p[j, env.state_to_index(states_p[j])] = 1
        
        #print 'sample heat map'
        #plot_im(numpy.sum(X.toarray(), axis = 0).reshape((9,9)))
        #plt.show() # db

        #X[:,-1] = 1 # append constant vector
        self.X = scipy.sparse.csc_matrix(X)
        self.X_p = scipy.sparse.csc_matrix(X_p)

        self.C = self.X.T * self.X # diagonal matrix of counts
        assert (self.C.todense() == numpy.diag(numpy.sum(self.X.todense(), axis = 0).flat)).all()
        assert (self.C.nonzero()[0] == self.C.nonzero()[1]).all() # diagonal
        self.C_inv = copy.deepcopy(self.C)
        self.C_inv.data = 1./self.C_inv.data

        #self.D = scipy.csc_matrix(scipy.sparse.dia_matrix((1./numpy.sum(self.X.toarray(), axis=0),0), \
                        #(self.n_samples, self.n_samples)) )# for normalizing sample distribution
        self.D = scipy.sparse.csc_matrix(numpy.eye(self.n_samples)) # not used currently

        self._precompute()
        
    def _precompute(self):

        self.PHI = self.X * self.W
        self.PHI_p = self.X_p * self.W
        self.E = self.PHI.T * self.PHI

        # for model update
        #means = scipy.sparse.csc_matrix.mean(self.PHI, axis=0)
        #M = scipy.sparse.csc_matrix(numpy.repeat(means, self.PHI.shape[0], \
                                                 #axis=0))
        #self.PHI_p_norm = self.PHI_p - M # subtract sample mean
        # TODO need to subtract sample mean?

        # normalize each column of PHI_p to one:
        self.diag = scipy.sparse.dia_matrix( ((1./ numpy.apply_along_axis( \
            numpy.linalg.norm, 0, self.PHI_p.toarray())), 0), \
            (self.n_features, self.n_features)) \
            
        self.PHI_p_norm = self.PHI_p * self.diag

    def _reward_gradient(self):
        
        B = self.R_norm.T * self.PHI

        dW_rew = self.X.T * (self.R_norm * (B * self.E) + (self.PHI * B.T) * B \
                                                    - 2 * self.R_norm * B)

        return dW_rew

    def _model_gradient(self):
        
        E_p = self.PHI_p.T * self.PHI

        #update using chain rule
        A = E_p * self.E - 2 * E_p

        return (self.X_p.T * self.PHI * A.T + self.X.T * self.PHI_p * A \
                + self.X.T * self.PHI * E_p.T * E_p + self.X_p.T * self.PHI_p) \
                * self.diag * self.diag * (1./self.n_features)

        #C = self.diag * (E_p * self.E - 2*E_p)
        #return (self.X.T * (self.PHI_p_norm * C + self.PHI * E_p.T * E_p) \
               #+ self.X_p.T * (self.PHI * C.T + self.PHI_p_norm * self.diag)) \
               #* (1./self.n_features)

        #alternate update, treating next step as static images
        #B = self.PHI_p_norm.T * self.PHI
        #return (self.X.T * (self.PHI_p_norm * (B * self.E) +  \
               #(self.PHI * B.T) * B - 2 * self.PHI_p_norm * B)) \
               #* (1./self.n_features)

    def _sparsity_gradient(self):
        
        #print 'active feature means: ', numpy.mean(U.data)
        #print 'active feature std: ', numpy.std(U.data)
        # weight sparsity by probability state is sampled
        # TODO add constant here for tanh scaling?
        return self.C * numpy.tanh(self.W.toarray())

    def get_gradient(self, W):
        
        #print 'gradient eval'
        #print 'W: ', W.shape
        W = scipy.sparse.csc_matrix(numpy.reshape(W, (self.n_states, self.n_features)))
        if not (W.data == self.W.data).all(): # too slow for inner loop?
            self.W = W
            self._precompute()
        
        dW_rew = (self._reward_gradient()).toarray()
        dW_mod = (self._model_gradient()).toarray()
        dW_reg = self._sparsity_gradient()

        dW = (self.rew_wt * dW_rew + self.mod_wt * dW_mod + self.reg_wt * dW_reg).flatten()

        #print 'gradient norm: ', scipy.linalg.norm(dW)
        
        return dW

    def _reward_loss(self):
        r_err = (self.PHI * self.PHI.T * self.D * self.R_norm - self.R_norm)
        #return 0.5 * scipy.sparse.csc_matrix.sum(r_err.multiply(r_err))
        return 0.5 * float((r_err.T * self.D * r_err).todense())
    
    # TODO adding D for uniform distribution learning
    def _model_loss(self):
        m_err = (self.PHI * self.PHI.T * self.D * self.PHI_p_norm - self.PHI_p_norm)
        #return 0.5 * scipy.sparse.csc_matrix.sum(m_err.multiply(m_err))
        return 0.5 * numpy.trace((m_err.T * self.D * m_err).todense())
        

    def _sparsity_loss(self):
        U = abs((self.X.T * self.PHI).toarray().flatten())
        L = numpy.zeros(U.shape)
        L[U > 1e2] = U[U > 1e2]
        L[U <= 1e2] = numpy.log(numpy.cosh(U[U <= 1e2]))
        return numpy.sum(L)
        
    
    # TODO add caching for loss, gradient
    def get_loss(self, W):
        #print 'loss eval'
        #print 'W: ', W.shape

        W = scipy.sparse.csc_matrix(numpy.reshape(W, (self.n_states, self.n_features)))
        if not (W.data == self.W.data).all():
            self.W = W
            self._precompute()
        
        # plot reward, reconstruction and gradient
        #plt.subplot(311)
        #plot_im(numpy.reshape((self.X.T * self.R_norm).toarray(), (9,9)))
        #plt.subplot(312)
        #plot_im(numpy.reshape((self.X.T * self.PHI * self.PHI.T * self.R_nrom).toarray(), (9,9)))
        #plt.subplot(313)
        #plot_im(numpy.reshape((self.get_gradient(W.toarray())), (9,9)))
        #plt.show()
        
        # reward loss:
        r_loss = self._reward_loss()
        #print 'reward loss: ', 1./self.n_samples * r_loss

        # model loss:
        m_loss = self._model_loss()
        #print 'model loss: ', m_loss

        # regularization / sparsity loss
        s_loss = self._sparsity_loss()
        #print 'sparsity loss: ', s_loss

        return (self.rew_wt * r_loss + self.mod_wt * m_loss + self.reg_wt * s_loss)

def init_mdp(goals = None, walls_on = False, size = 9):

    if goals is None:

        buff = size/9
        pos = size/3-1
        goals = numpy.zeros((size,size))
        goals[pos-buff:pos+buff, pos-buff:pos+buff] = 1
        #goals[pos-buff:pos+buff, size-pos-buff:size-pos+buff] = 1
        #goals[size-pos-buff:size-pos+buff, pos-buff:pos+buff] = 1
        #goals[size-pos-buff:size-pos+buff, size-pos-buff:size-pos+buff] = 1

 
    walls = numpy.zeros((size,size))
    if walls_on:
        #walls[:, size/2] = 1
        walls[size/2, :] = 1
        walls[size/2, size/2] = 0 

    grid_world = GridWorld(walls, goals)

    rand_policy = RandomPolicy()

    mdp = MDP(grid_world, rand_policy)

    return mdp


def plot_weights(W, im_shape):
        
    n_rows, n_cols = im_shape
    n_states, n_features = W.shape
    if n_features == 1:
        plt.imshow(numpy.reshape(W, \
            (n_rows, n_cols)) \
            ,interpolation = 'nearest', cmap = 'gray')
        plt.colorbar()
    else:
        for i in xrange(n_features):
            plt.subplot(n_features/10 , 10 , i + 1)
            plot_im(numpy.reshape(W[:,i], (n_rows, n_cols)))

    plt.show()

def plot_im(W):
    plt.imshow(W, interpolation = 'nearest', cmap = 'gray')
    plt.colorbar()

def reconstruction_loss(obj, mod_wt, reg_wt):
    
    L_rew = 1./obj.n_samples * obj._reward_loss()
    L_mod = 1./obj.n_samples * obj._model_loss()
    L_reg = 1./obj.n_samples * obj._sparsity_loss()
    L_obj = L_rew + mod_wt * L_mod + reg_wt * L_reg

    return L_obj, L_rew, L_mod, L_reg

# TODO weight optimal bellman error by policy distribution
# add goal/restart transitions to P
def _bell_err(PHI, PHI_p, R, gam, D = None):
    
    n_samples, n_features = PHI.shape

    # append constant features?
    n_features += 1
    PHI = scipy.sparse.hstack((PHI, \
                                    numpy.ones((n_samples,1))))
    PHI_p = scipy.sparse.hstack((PHI_p, \
                                    numpy.ones((n_samples,1))))
    

    # normalize columns to 1?
    PHI = PHI * scipy.sparse.dia_matrix( \
        (1./numpy.apply_along_axis(numpy.linalg.norm, 0, PHI.toarray()), 0)\
        , (n_features, n_features))

    PHI_p = PHI_p * scipy.sparse.dia_matrix( \
        (1./numpy.apply_along_axis(numpy.linalg.norm, 0, PHI_p.toarray()), 0)\
        , (n_features, n_features))
        
    if D is None:
        D = scipy.sparse.csc_matrix(scipy.sparse.dia_matrix(([1]*PHI.shape[0], 0), \
                (n_samples, n_samples)))

    # test bellman error (using same dataset used to set w)
    A = PHI.T * D * (PHI - gam * PHI_p) + scipy.sparse.csc_matrix(1e-8 * numpy.eye(n_features))
    b = PHI.T * D * R
    w = scipy.sparse.csc_matrix(numpy.linalg.solve(A.toarray(), b.toarray()))
    print 'solving'
    # TODO (1./n_samples) here? Dsqrt * to weight BE?
    be = R + (gam * PHI_p - PHI) * w
    BE = float((be.T * D * be).todense()) # weighted squared norm
    # TODO add weighting to solver
    # reward error
    A = PHI.T * D * PHI + scipy.sparse.csc_matrix(1e-8*numpy.eye(n_features))
    b = PHI.T * D * R
    w_rew = scipy.sparse.csc_matrix(numpy.linalg.solve(A.toarray(), b.toarray()))
    re = PHI * w_rew - R
    RE = float((re.T * D * re).todense())

    # model error (general and component that contributes to BE)
    B = PHI.T * D * PHI_p
    w_phi = scipy.sparse.csc_matrix(numpy.linalg.solve(A.toarray(), B.toarray()))
    M = (PHI * w_phi - PHI_p) # model error matrix

    ME_gen = numpy.trace((M.T * D * M).todense())
    me = (M * w)
    ME_bel = float((me.T * D * me).todense())
    
    
    # visualize bellman error, db
    #print 'bellman error'
    #plt.imshow(be.toarray().reshape(9,9), interpolation = 'nearest', cmap = 'gray')
    #plt.colorbar()
    #plt.show()
    
    
    ## visualize (approx) value function , db
    print 'value function'
    v = (PHI * w)
    V = v.toarray().reshape(9,9)
    #plt.subplot(211)
    plt.imshow(V, interpolation = 'nearest', cmap = 'gray')
    plt.colorbar()
    #plt.subplot(212)
    #d = scipy.sparse.csc_matrix(1./(D.toarray() + 1e-12)) 
    #V = (d * v).toarray().reshape((9,9))
    #plt.imshow(V, interpolation = 'nearest', cmap = 'gray')
    #plt.colorbar()
    plt.show()
    #visualize one-step transitions
    #plot_weights(PHI_p.todense(), (9,9))

    return BE, RE, ME_gen, ME_bel

def zero_array(num_zeros, n):
    obj_list = [None] * num_zeros
    for i in xrange(num_zeros):
        obj_list[i] = numpy.zeros(n)

    return obj_list
        
def cg_main(mb_size = 600, n_iters = 100, size = 9, goals = None, \
        n_features = 10, rew_wt = 1., mod_wt = 1., reg_wt = 2., \
        lam = 0.9, gam = 0.999):
    
    mdp = init_mdp(size=size, walls_on = True)
    env = mdp.env
    n_states = size**2 # assumes square room
    W = 1e-1 * numpy.random.standard_normal((n_states, n_features))
    W_flat = W.flatten()
    
    # sample all data ahead of time
    states, rewards = mdp.sample(mb_size * n_iters)
    SR = numpy.vstack((states[:-1,:].T, states[1:,:].T, rewards))
    numpy.random.shuffle(SR) # randomize the data
    STA = SR[0:2,:].T
    STA_p = SR[2:4,:].T
    REW = SR[4,:].T

     #view heat map
    #H = numpy.zeros((size,size))
    #for s in states:
        #H[s[0], s[1]] += 1
    
    #print 'heat map'
    #plot_im(H.T)
    #plt.show()
 
    print 'SR shape: ', SR.shape
    print 'STA shape: ', STA.shape
    print 'REW shape: ', REW.shape
    
    # initialize lists for performance tracking
    L_rew, L_mod, L_reg, L_obj, \
    L_rew_test, L_mod_test, L_reg_test, L_obj_test, \
    BE, RE, ME_gen, ME_bel = zero_array(12, n_iters + 1)

    
    n_test_samples = mb_size
    test_states, test_rewards = mdp.sample(n_test_samples)
    
    I = scipy.sparse.csc_matrix(numpy.eye(n_states)) # tabular representation
    Y = scipy.sparse.dia_matrix(([1]*n_states, 0), (n_states, n_states)) 
    assert numpy.sum((Y - I).todense()) == 0 # db


    P = scipy.sparse.csc_matrix(env.P)
    R = scipy.sparse.csc_matrix(env.R).T
    r_avg = 0
    
     #db
    #plot_im(R.toarray().reshape((9,9)))
    #plt.show()

    print 'ideal bellman error: ', _bell_err(I, P*I, R, gam)

    for i in xrange(n_iters):
        print 'iter: ', i+1

        # TODO make sampling object?
        print 'creating objective function'
        s = STA[i*mb_size:(i+1)*mb_size, :]
        s_p = STA_p[i*mb_size:(i+1)*mb_size, :]
        r = REW[i*mb_size:(i+1)*mb_size]

        r_avg = (i*r_avg + numpy.mean(r)) / float(i+1)
        obj = SIRFObjectiveFn(W, r, s, s_p, env, rew_wt, mod_wt, reg_wt, r_avg)
        
        print 'recording error/loss'
        
        #print 'ideal bellman error on data: ', _bell_err(I, P*I, R, gam, obj.C) # db for plotting
        
        # training error
        L_obj[i], L_rew[i], L_mod[i], L_reg[i] = reconstruction_loss(obj, mod_wt, reg_wt)
        # test error
        #test_obj = SIRFObjectiveFn(W, test_rewards, test_states, env, rew_wt, mod_wt, reg_wt)
        #L_obj_test[i], L_rew_test[i], L_mod_test[i], L_reg_test[i] = reconstruction_loss(test_obj, mod_wt, reg_wt)
        # bellman error
        w = scipy.sparse.csc_matrix(W)
        BE[i], RE[i], ME_gen[i], ME_bel[i] = _bell_err(w, P*w, R, gam, env.D)
        print 'Bellman error: ', BE[i]
        
        print 'updating weights using cg'
        W_flat = scipy.optimize.fmin_cg(obj.get_loss, W_flat, \
                                        fprime = obj.get_gradient, \
                                        maxiter = 1,
                                        gtol = 1e-8)
        
        print 'dW norm: ', numpy.linalg.norm(W_flat - W.flatten()), W.flatten().shape, W_flat.shape
        W = numpy.reshape(W_flat, (n_states, n_features))
        print 'W column norms: ', numpy.apply_along_axis(numpy.linalg.norm, 0, W)

        # TODO:
        # get reward, model, and bellman errors
        # get training, test error for model, reward, and objective
        # precompute, store, and mix samples
        # add lambda param
        # use better environment for displaying features
        # autosave final weights, image
        # compute optimal value function, compare optimal, reconstructed, BE FP
        # average mean and normalizing constants

        # video of learning?
        

        #plot_weights(W, (size,size))
    
    print 'final bellman error: ', _bell_err(w, P*w, R, gam, env.D)

    #states, rewards = mdp.sample(n_samples)
    #obj = SIRFObjectiveFn(W, rewards, states, env, rew_wt, mod_wt, reg_wt)

    ## record final error and loss
    ## training error
    #L_obj[n_iters], L_rew[n_iters], L_mod[n_iters], L_reg[n_iters] = reconstruction_loss(obj, mod_wt, reg_wt)
    ## test error
    #test_obj = SIRFObjectiveFn(W, test_rewards, test_states, env, rew_wt, mod_wt, reg_wt)
    #L_obj_test[n_iters], L_rew_test[n_iters], L_mod_test[n_iters], L_reg_test[n_iters] = reconstruction_loss(test_obj, mod_wt, reg_wt)
    #w = scipy.sparse.csc_matrix(W)
    #BE[n_iters], RE[n_iters], ME_gen[n_iters], ME_bel[n_iters] = _bell_err(w, P*w, R, gam, env.D)
    
    ##print 'Bellman Error: ', BE[n_iters]
    
    fig = plt.figure(1)
    x = range(n_iters + 1)    
    ax = fig.add_subplot(2,1,1)
    ax.plot(x, L_obj, 'k-', x, L_rew, 'g-', x, L_mod, 'b-', x, L_reg, 'r-')
    #ax.legend(('objective','reward loss', 'model loss', 'regularization loss'), loc=5)

    #ax = fig.add_subplot(4,1,2)
    #ax.plot(x, L_obj_test, 'k-', x, L_rew_test, 'g-', x, L_mod_test, 'b-', x, L_reg_test, 'r-')
    ##ax.legend(('objective','reward loss', 'model loss', 'regularization loss'), loc=5)

    #ax = fig.add_subplot(4,1,3)
    #ax.plot(x, L_obj_test / max(L_obj_test), 'k-', x, L_rew_test / max(L_rew_test), 'g-', \
             #x, L_mod_test / max(L_mod_test), 'b-', x, L_reg_test / max(L_reg_test), 'r-')
    ##ax.legend(('objective','reward loss', 'model loss', 'regularization loss'), loc=5)

    ax = fig.add_subplot(2,1,2)       
    ax.plot(x, BE/max(BE), 'k-', x, RE/max(RE), 'g-', \
        x, ME_gen/max(ME_gen), 'b-', x, ME_bel/max(ME_bel), 'r-')
    #ax.legend(('Bellman err','reward err', 'gen model err', 'bellman mod err'), loc=5)

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, BE, 'k-')
    #plt.ylim([0,4])

    plt.show()

    plot_weights(W, (size,size))
# TODO effects of centering features
def sgd_main(n_samples = 250, size = 9, goals = None, \
        n_features = 10, rew_wt = 1, mod_wt = 2, reg_wt = 4e-1, \
        lam = 0.9, gam = 0.995, lr = 1e-3):
    
    mdp = init_mdp(size=size, walls_on = True)
    env = mdp.env
    n_states = size**2 # assumes square room
    W = 1e-6 * numpy.random.standard_normal((n_states, n_features))
    W_flat = W.flatten()
    
    
    # initialize lists for performance tracking
    n_iters = 100
    L_rew, L_mod, L_reg, L_obj, \
    L_rew_test, L_mod_test, L_reg_test, L_obj_test, \
    BE, RE, ME_gen, ME_bel = zero_array(12, n_iters + 1)

    
    n_test_samples = 4*n_samples
    test_states, test_rewards = mdp.sample(n_test_samples)
    
    I = scipy.sparse.csc_matrix(numpy.eye(n_states)) # tabular representation
    P = scipy.sparse.csc_matrix(env.P)
    R = scipy.sparse.csc_matrix(env.R).T
    r_avg = 0
    print 'ideal bellman error: ', _bell_err(I, P*I, R, gam, env.D)

    for i in xrange(n_iters):
        print 'iter: ', i+1

        if i % 30 == 0:
            lr = lr * (1-1e-4)
            print 'learning rate: ', lr

        print 'sampling mdp and creating objective function'
        states, rewards = mdp.sample(n_samples)
        r_avg = (i*r_avg + numpy.mean(rewards)) / float(i+1)
        obj = SIRFObjectiveFn(W, rewards, states, env, rew_wt, mod_wt, reg_wt, r_avg)
        
        print 'recording error/loss'
        # training error
        L_obj[i], L_rew[i], L_mod[i], L_reg[i] = reconstruction_loss(obj, mod_wt, reg_wt)
        # test error
        test_obj = SIRFObjectiveFn(W, test_rewards, test_states, env, rew_wt, mod_wt, reg_wt)
        L_obj_test[i], L_rew_test[i], L_mod_test[i], L_reg_test[i] = reconstruction_loss(test_obj, mod_wt, reg_wt)
        # bellman error
        w = scipy.sparse.csc_matrix(W)
        BE[i], RE[i], ME_gen[i], ME_bel[i] = _bell_err(w, P*w, R, gam, env.D)
        print 'Bellman error: ', BE[i]
        
        
        
        print 'updating weights using sgd'
        W_flat += lr * obj.get_gradient(W_flat)
         
        print type(W_flat)
        print type(W.flatten())
        print 'dW norm: ', numpy.linalg.norm(W_flat - W.flatten())
        W = numpy.reshape(copy.deepcopy(W_flat), (n_states, n_features))
        print 'W column norms: ', numpy.apply_along_axis(numpy.linalg.norm, 0, W)

        # TODO:
        # get reward, model, and bellman errors
        # get training, test error for model, reward, and objective
        # precompute, store, and mix samples
        # add lambda param
        # use better environment for displaying features
        # autosave final weights, image
        # compute optimal value function, compare optimal, reconstructed, BE FP
        # average mean and normalizing constants

        # video of learning?
        

        #plot_weights(W, (size,size))
    
    print 'final bellman error: ', _bell_err(w, P*w, R, gam, env.D)

    states, rewards = mdp.sample(n_samples)
    obj = SIRFObjectiveFn(W, rewards, states, env, rew_wt, mod_wt, reg_wt)

    # record final error and loss
    # training error
    L_obj[n_iters], L_rew[n_iters], L_mod[n_iters], L_reg[n_iters] = reconstruction_loss(obj, mod_wt, reg_wt)
    # test error
    test_obj = SIRFObjectiveFn(W, test_rewards, test_states, env, rew_wt, mod_wt, reg_wt)
    L_obj_test[n_iters], L_rew_test[n_iters], L_mod_test[n_iters], L_reg_test[n_iters] = reconstruction_loss(test_obj, mod_wt, reg_wt)
    w = scipy.sparse.csc_matrix(W)
    BE[n_iters], RE[n_iters], ME_gen[n_iters], ME_bel[n_iters] = _bell_err(w, P*w, R, gam, env.D)
    
    #print 'Bellman Error: ', BE[n_iters]
    
    fig = plt.figure(1)
    x = range(n_iters + 1)    
    ax = fig.add_subplot(4,1,1)
    ax.plot(x, L_obj, 'k-', x, L_rew, 'g-', x, L_mod, 'b-', x, L_reg, 'r-')
    #ax.legend(('objective','reward loss', 'model loss', 'regularization loss'), loc=5)

    ax = fig.add_subplot(4,1,2)
    ax.plot(x, L_obj_test, 'k-', x, L_rew_test, 'g-', x, L_mod_test, 'b-', x, L_reg_test, 'r-')
    #ax.legend(('objective','reward loss', 'model loss', 'regularization loss'), loc=5)

    ax = fig.add_subplot(4,1,3)
    ax.plot(x, L_obj_test / max(L_obj_test), 'k-', x, L_rew_test / max(L_rew_test), 'g-', \
             x, L_mod_test / max(L_mod_test), 'b-', x, L_reg_test / max(L_reg_test), 'r-')
    #ax.legend(('objective','reward loss', 'model loss', 'regularization loss'), loc=5)

    ax = fig.add_subplot(4,1,4)       
    ax.plot(x, BE/max(BE), 'k-', x, RE/max(RE), 'g-', \
        x, ME_gen/max(ME_gen), 'b-', x, ME_bel/max(ME_bel), 'r-')
    #ax.legend(('Bellman err','reward err', 'gen model err', 'bellman mod err'), loc=5)

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, BE, 'k-')
    #plt.ylim([0,4])

    plt.show()

    plot_weights(W, (size,size))


def limit_dist():
    mdp = init_mdp(size=9, walls_on = True)
    env = mdp.env

    P = env.P #+ numpy.eye(env.P.shape[0]) * 1e-6
    #w, v = numpy.linalg.eig(P)
    #v = v[numpy.argsort(w)]

    #pi = v[:,0]    
    #print pi
    #plot_im(numpy.reshape(pi, (9,9)))
    #plt.show()
    
    pi = numpy.zeros(P.shape[0])
    pi[0] = 1
    delta = 1e10
    i=0
    while delta > 1e-10:
        
        print i
        i += 1

        pi_old = copy.deepcopy(pi)
        pi = numpy.dot(P,pi)
        pi = pi / numpy.linalg.norm(pi)

        delta = numpy.linalg.norm(pi-pi_old)
    
    plot_im(numpy.reshape(pi, (9,9)))
    plt.show()



def test_grid_world():

    rand_policy = RandomPolicy()

    walls = numpy.zeros((9,9))
    grid_world = GridWorld(walls)

    mdp = MDP(grid_world, rand_policy)
    
    states, rewards = mdp.sample(100)
    assert len(states) == len(rewards) + 1

    # assert states are all in bounds
    assert len(states[states < 0]) == 0
    x_pos = states[:,0]
    y_pos = states[:,1]
    assert len(x_pos[ x_pos >= grid_world.n_rows ]) == 0
    assert len(y_pos[ y_pos >= grid_world.n_cols ]) == 0
            

#TOD 
# add tests for reward function/goals
# video of learning - ggplot or mpl
# conjugate gradient
# regularization - l1 and parsity
# centering and normalization

# grid search / cv for learning rates and hyperparams
# plot sample error during learning: BE/value error, and reward and model error
# compare update rules (how far to chain rule?)
# plot optimal value function and projected value function
#   -solve for optimal using value iteration
# get rid of constant feature?

# mix samples for minibatches
# average reward over time - use mean offset reward and no centering for phi?
# use uniform distribution for learning
# look at heat map
# sgd vs cg


if __name__ == "__main__":
    cg_main()
    #limit_dist()


            
    
