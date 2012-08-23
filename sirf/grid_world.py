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

    def __init__(self, wall_matrix, goal_matrix, init_state = None):
        
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
                                
                                if act_list is None:
                                    self._adjacent[(i,j)] = set()
                                    self._actions[(i,j)] = set()

                                self._adjacent[(i,j)].add(tuple(pos))
                                self._actions[(i,j)].add(tuple(v))

        # form transition matrix P and reward function R
        P = numpy.zeros((self.n_states, self.n_states))

        for state, adj_set in self._adjacent.items():

            idx = self.state_to_index(state)
            adj_ids = map(self.state_to_index, adj_set)
            P[adj_ids,idx] = 1

        # normalize columns to have unit sum
        self.P = numpy.dot(P, numpy.diag(1./(numpy.sum(P, axis=0)+1e-8)))
        
        # build reward function
        self.R = numpy.zeros(self.n_states)
        nz = zip(*self.goals.nonzero())
        gol_ids = map(self.state_to_index, nz)
        self.R[gol_ids] = 1

        # find limiting distribution
        v = numpy.zeros((self.P.shape[0],1))
        v[-1,0] = 1
        delta = 1
        while  delta > 1e-8:
            v_old = copy.deepcopy(v)
            v = numpy.dot(self.P,v)
            v = v / numpy.linalg.norm(v)
            delta = numpy.linalg.norm(v-v_old)
        
        #plot_im(numpy.reshape(v[:,0], (9,9)))
        ##plot_im(self.P)
        #plt.show()
        
        self.D = scipy.sparse.dia_matrix((v[:,0],0),(self.n_states, self.n_states))

            

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
        
        #randomly ignore actions and self-transition
        #if numpy.random.random() < 2e-1:
            #return self.state

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
        return state[0] + state[1]*self.n_rows

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

            states[i+1] = next_state
            rewards[i] = reward
         
        return states, rewards

class SIRFObjectiveFn:
    
    def __init__(self, W, rewards, states, env, rew_wt, mod_wt, reg_wt, \
                rew_avg = None):
        
        self.W = scipy.sparse.csc_matrix(W)
        self.rew_wt = rew_wt
        self.mod_wt = mod_wt
        self.reg_wt = reg_wt

        #TODO  average reward mean over episodes
        self.R = scipy.sparse.csc_matrix(rewards).T

        if rew_avg is None:
            avg = numpy.mean(rewards)
        else:
            avg = rew_avg
        rewards = rewards - avg # worth loss is sparsity to center?
        rewards = rewards / numpy.sqrt(numpy.mean(rewards**2)) # normalize std to one
        assert (rewards.std() - 1) < 1e-8
        if len(rewards.nonzero()[0]) is not 0: 
            rewards = rewards / numpy.linalg.norm(rewards)
        self.R_norm = scipy.sparse.csc_matrix(rewards).T # sparse row vector

        self.n_samples = max(rewards.shape)
        self.n_states, self.n_features = W.shape
        
        # create tabular feature data matrices given the sampled states
        X_all = scipy.sparse.lil_matrix((self.n_samples+1, self.n_states), dtype = numpy.int)
        for j in xrange(self.n_samples+1):
            X_all[j, env.state_to_index(states[j])] = 1
        #X_all[:,-1] = 1 # append constant vector
        X_all = scipy.sparse.csc_matrix(X_all)

        self.X = X_all[:-1]
        self.X_p = X_all[1:]
        self.C = self.X.T * self.X # diagonal matrix of counts
        assert (self.C.nonzero()[0] == self.C.nonzero()[1]).all()
        self.C_inv = copy.deepcopy(self.C)
        self.C_inv.data = 1./self.C_inv.data

        self.D = scipy.sparse.dia_matrix(1./(numpy.sum(self.X.toarray(), axis=0),0), \
                        (self.n_samples, self.n_samples)) # for normalizing sample distribution

        self._precompute()
        
    def _precompute(self):

        self.PHI = self.X * self.W
        self.PHI_p = self.X_p * self.W
        self.E = self.PHI.T * self.PHI

        # for model update
        means = scipy.sparse.csc_matrix.mean(self.PHI, axis=0)
        M = scipy.sparse.csc_matrix(numpy.repeat(means, self.PHI.shape[0], \
                                                 axis=0))
        self.PHI_p_norm = self.PHI_p - M # subtract sample mean

        # create diag matrix used to normalize each feature's variance to 1:
        #STD = scipy.sparse.csc_matrix.mean(self.PHI.multiply(self.PHI), axis = 0)
        #STD = numpy.sqrt(STD) + 1e-12
        #D = scipy.sparse.dia_matrix((1./STD, 0), (self.n_features, self.n_features))
        #self.PHI_p_norm = self.PHI_p * D

        # or instead normalize each column of PHI_p to one:
        self.diag = scipy.sparse.dia_matrix( ((1./ numpy.apply_along_axis( \
            numpy.linalg.norm, 0, self.PHI_p.toarray())), 0), \
            (self.n_features, self.n_features)) \
            
        self.PHI_p_norm = self.PHI_p_norm * self.diag

    def _reward_gradient(self):
        
        B = self.R_norm.T * self.PHI

        dW_rew = self.X.T * (self.R_norm * (B * self.E) + (self.PHI * B.T) * B \
                                                    - 2 * self.R_norm * B)

        return dW_rew

    def _model_gradient(self):
        
        # old update using chain rule
        #E_p = self.PHI_p_norm.T * self.PHI
        #C = self.diag * (E_p * self.E - 2*E_p)
        #return (self.X.T * (self.PHI_p_norm * C + self.PHI * E_p.T * E_p) \
               #+ self.X_p.T * (self.PHI * C.T + self.PHI_p_norm * self.diag)) \
               #* (1./self.n_features)

        # alternate update, treating next step as static images
        B = self.PHI_p_norm.T * self.PHI
        return (self.X.T * (self.PHI_p_norm * (B * self.E) +  \
               (self.PHI * B.T) * B - 2 * self.PHI_p_norm * B)) \
               * (1./self.n_features)

    def _sparsity_gradient(self):
        
        #print 'active feature means: ', numpy.mean(U.data)
        #print 'active feature std: ', numpy.std(U.data)
        # weight sparsity by probability state is sampled
        # TODO add constant here for tanh scaling?
        return (1./self.n_samples) * self.C * numpy.tanh(self.W.toarray())

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
        return 0.5 * scipy.sparse.csc_matrix.sum(r_err.multiply(r_err))
    
    # TODO adding D for uniform distribution learning
    def _model_loss(self):
        m_err = (self.PHI * self.PHI.T * self.D * self.PHI_p_norm - self.PHI_p_norm)
        return 0.5 * scipy.sparse.csc_matrix.sum(m_err.multiply(m_err))

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
        goals[size-pos-buff:size-pos+buff, size-pos-buff:size-pos+buff] = 1

 
    walls = numpy.zeros((size,size))
    if walls_on:
        #walls[:, size/2] = 1
        walls[size/2, :] = 1
        walls[size/2, size/2] = 0 

    grid_world = GridWorld(walls, goals)

    rand_policy = RandomPolicy()

    mdp = MDP(grid_world, rand_policy)

    return mdp

def init_weights(shape):

    W = 1e-3 * numpy.random.standard_normal(shape)
    #inv_norms = numpy.diag(1./numpy.sqrt(numpy.sum(W**2, axis=0)))
    #W = numpy.dot(W, inv_norms)

    return W

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

def bellman_error(obj, gam, uniform_weighting = False):

    if not uniform_weighting:
        obj

    PHI = obj.PHI
    PHI_p = obj.PHI_p

    n_samples, n_features = PHI.shape

    # normalize columns to 1
    PHI = obj.PHI * scipy.sparse.dia_matrix( \
        (1./numpy.apply_along_axis(numpy.linalg.norm, 0, obj.PHI.toarray()), 0)\
        , (n_features, n_features))

    PHI_p = obj.PHI_p * scipy.sparse.dia_matrix( \
        (1./numpy.apply_along_axis(numpy.linalg.norm, 0, obj.PHI_p.toarray()), 0)\
        , (n_features, n_features))

    R = obj.R_norm

    # append constant features
    #PHI = scipy.sparse.hstack((obj.PHI, \
                                    #numpy.ones((n_samples,1))))
    #PHI_p = scipy.sparse.hstack((obj.PHI_p, \
                                    #numpy.ones((n_samples,1))))

    return _bell_err(PHI, PHI_p, R, gam, n_features)

# TODO weight optimal bellman error by policy distribution
# add goal/restart transitions to P
def _bell_err(PHI, PHI_p, R, gam, D = None):
        
    n_samples, n_features = PHI.shape

    if D is None:
        D = scipy.sparse.dia_matrix(([1]*PHI.shape[0], 0), \
                (n_samples, n_samples))
    
    #Dsqrt = scipy.sparse.csc_matrix(numpy.sqrt(D.todense()))

    # test bellman error (using same dataset used to set w)
    A = PHI.T * D * (PHI - gam * PHI_p) + scipy.sparse.csc_matrix(1e-8 * numpy.eye(n_features))
    b = PHI.T * D * R
    w = scipy.sparse.csc_matrix(numpy.linalg.solve(A.toarray(), b.toarray()))
    # TODO (1./n_samples) here? Dsqrt * to weight BE?
    #BE = numpy.linalg.norm(((R + (gam * PHI_p - PHI) * w)).data)
    be = R + (gam * PHI_p - PHI) * w
    BE = float((be.T * D * be).todense()) # weighted squared norm

    ## visualize (approx) value function 
    #V = (PHI * w).toarray().reshape(9,9)
    #plt.imshow(V, interpolation = 'nearest', cmap = 'gray')
    #plt.show()
    ## visualize one-step transitions
    #plot_weights(PHI_p.todense()[:,25:50], (9,9))


    # reward error
    A = PHI.T * D * PHI + scipy.sparse.csc_matrix(1e-8*numpy.eye(n_features))
    b = PHI.T * D * R
    w_rew = scipy.sparse.csc_matrix(numpy.linalg.solve(A.toarray(), b.toarray()))
    re = PHI * w_rew - R
    RE = float((re.T * D * re).todense())
    #RE = numpy.linalg.norm(((PHI * w_rew - R)).data)

    # model error (general and component that contributes to BE)
    B = PHI.T * D * PHI_p
    w_phi = scipy.sparse.csc_matrix(numpy.linalg.solve(A.toarray(), B.toarray()))
    M = (PHI * w_phi - PHI_p) # model error matrix

    
    ME_gen = numpy.trace((M.T * D * M).todense())
    me = (M * w_phi)
    ME_bel = float((me.T * D * me).todense())
    #ME_bel = numpy.linalg.norm((M * w_phi).data)

    return BE, RE, ME_gen, ME_bel
        
def cg_main(n_samples = 250, size = 9, goals = None, \
        n_features = 10, rew_wt = 1, mod_wt = 8, reg_wt = 4e-1, \
        lam = 0.9, gam = 0.99):
    
    mdp = init_mdp(size=size, walls_on = True)
    env = mdp.env
    n_states = size**2 # assumes square room
    W = init_weights((n_states, n_features))
    W_init = W.flatten()
    
    #plot_weights(W, (size,size))
    
    n_iters = 300
    L_rew = numpy.zeros(n_iters+1)
    L_mod = numpy.zeros(n_iters+1)
    L_reg = numpy.zeros(n_iters+1)
    L_obj = numpy.zeros(n_iters+1)
    
    # TODO combine into array /obj
    L_rew_test = numpy.zeros(n_iters+1)
    L_mod_test = numpy.zeros(n_iters+1)
    L_reg_test = numpy.zeros(n_iters+1)
    L_obj_test = numpy.zeros(n_iters+1)

    BE = numpy.zeros(n_iters+1)
    RE = numpy.zeros(n_iters+1)
    ME_gen = numpy.zeros(n_iters+1)
    ME_bel = numpy.zeros(n_iters+1)

    
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
            n_samples = numpy.round(n_samples * 1.1).astype(numpy.int)
            print 'minibatch size: ', n_samples
        
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

        
        print 'updating weights using cg'
        W_init = scipy.optimize.fmin_cg(obj.get_loss, W_init, \
                                        fprime = obj.get_gradient, \
                                        maxiter = 3,
                                        gtol = 1e-8)
        
        W_init = numpy.reshape(W_init, (n_states, n_features))
        print 'dW norm: ', numpy.linalg.norm(W_init - W)
        W = W_init
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
    
    print 'Bellman Error: ', BE
    
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
    plt.ylim([0,4])

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
    while delta > 1e-6:
        
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


            
    
