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

    _vecs = numpy.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) # 4 movement dirs

    def __init__(self, wall_matrix, goal_positions, init_state = None):
        
        self.walls = wall_matrix
        self.goals = goal_positions
        # TODO assert walls and goals are correct format

        self.n_rows, self.n_cols = self.walls.shape
        self.n_states = self.n_rows * self.n_cols

        self._adjacent = {}
        self._actions = {}

        if init_state is None:
            self.state = self.random_state()
        else:
            assert _check_valid_state(init_state)
            self.state = init_state

        # precompute adjacent state and available actions given wall layout
        for i in xrange(self.n_rows):
            for j in xrange(self.n_cols):
                for v in self._vecs:
                    
                    pos = numpy.array([i,j]) + v
                    if self._check_valid_state(pos):

                            act_list = self._actions.get((i,j))
                            
                            if act_list is None:
                                self._adjacent[(i,j)] = []
                                self._actions[(i,j)] = []

                            self._adjacent[(i,j)].append(pos)
                            self._actions[(i,j)].append(v)

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
        if self.state.tolist() in self.goals.tolist():
            return self.random_state()
        
        #randomly ignore actions and self-transition
        if numpy.random.random() < 2e-1:
            return self.state

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
        if self.state.tolist() in self.goals.tolist():
            return 1
        else:
            return 0

    def state_to_index(self, state):
        return state[0]*self.n_cols + state[1]

    def random_state(self):

        r_state = None
        while not self._check_valid_state(r_state):
            r_state = numpy.round(numpy.random.random(2) * self.walls.shape)\
                        .astype(numpy.int)
        
        return r_state

class RandomPolicy:

    def choose_action(self, actions):
        return choice(actions)

class MDP:
    
    def __init__(self, environment, policy):
        self.environment = environment
        self.policy = policy
        
    def sample(self, n_samples):
        ''' sample the interaction of policy and environment for n_samples, 
        returning arrays of state positions and rewards '''
        
        states = numpy.zeros((n_samples+1,2), dtype = numpy.int)
        states[0] = self.environment.state
        rewards = numpy.zeros(n_samples, dtype = numpy.int)
        
        for i in xrange(n_samples):
            actions = self.environment.get_actions(self.environment.state)
            action = self.policy.choose_action(actions)
            next_state, reward = self.environment.take_action(action)

            states[i+1] = next_state
            rewards[i] = reward
         
        return states, rewards

class SIRFObjectiveFn:
    
    def __init__(self, W, rewards, states, env, mod_wt):
        
        self.W = scipy.sparse.csc_matrix(W)
        self.mod_wt = mod_wt

        rewards = rewards - numpy.mean(rewards) # worth loss is sparsity to center?
        rewards = rewards / numpy.sqrt(numpy.mean(rewards**2)) # normalize std to one
        assert (rewards.std() - 1) < 1e-8
        self.R = scipy.sparse.csc_matrix(rewards).T # sparse row vector

        self.n_samples = max(rewards.shape)
        self.n_states, self.n_features = W.shape
        
        # create tabular feature data matrices given the sampled states
        X_all = scipy.sparse.lil_matrix((self.n_samples+1, self.n_states), dtype = numpy.int)
        for j in xrange(self.n_samples+1):
            X_all[j, env.state_to_index(states[j])] = 1
        X_all[:,-1] = 1 # append constant vector
        X_all = scipy.sparse.csc_matrix(X_all)

        self.X = X_all[:-1]
        self.X_p = X_all[1:]

        self._precompute()
        
    def _precompute(self):

        self.PHI = self.X * self.W
        self.PHI_p = self.X_p * self.W
        self.E = self.PHI.T * self.PHI

        # for model update
        means = scipy.sparse.csc_matrix.mean(self.PHI, axis=0)
        M = scipy.sparse.csc_matrix(numpy.repeat(means, self.PHI.shape[0], axis=0))
        self.PHI_p = self.PHI_p - M # subtract sample mean

        # create diag matrix used to normalize each feature's variance to 1: TODO use diag?
        STD = scipy.sparse.csc_matrix.mean(self.PHI.multiply(self.PHI), axis = 0)
        STD = numpy.sqrt(STD) + 1e-12
        D = scipy.sparse.dia_matrix((1./STD, 0), (self.n_features, self.n_features))
        
        print self.PHI_p.shape
        print D.shape
        self.PHI_p_norm = self.PHI_p * D

    def _reward_gradient(self):
        
        B = self.R.T * self.PHI
        dW_rew = self.X.T * (self.R * (B * self.E) + (self.PHI * B.T) * B \
                                                    - 2 * self.R * B)
        #db
        #A = self.X.T * R
        #dW_rew_old = A * (B * self.PHI.T) * self.PHI + X.T * (self.PHI * B.T) * B - 2 * A * B  could factor X.T out
        #assert numpy.linalg.norm((dW_rew - dW_rew_old).todense()) < 1e-4
        return dW_rew

    def _model_gradient(self):
        
        # old update using chain rule
        #E_p = PHI_p_norm.T * PHI
        #C = D * (E_p * E - 2*E_p)
        #return (X.T * (PHI_p * C + PHI * E_p.T * E_p) \
               #+ X_p.T * (PHI * C.T + PHI_p_norm * D)) * (1./n_features)

        # alternate update, treating next step as static images
        B = self.PHI_p_norm.T * self.PHI
        return (self.X.T * (self.PHI_p_norm * (B * self.E) +  \
               (self.PHI * B.T) * B - 2 * self.PHI_p_norm * B)) \
               * (1./self.n_features)

    def get_gradient(self, W):
        
        print 'gradient'

        W = scipy.sparse.csc_matrix(numpy.reshape(W, (self.n_states, self.n_features)))
        if not (W.data == self.W.data).all(): # too slow for inner loop?
            self.W = W
            self._precompute()
            
        dW_rew = (1./self.n_samples) * self._reward_gradient()
        dW_mod = (1./self.n_samples) * self._model_gradient()
        
        return (dW_rew + self.mod_wt * dW_mod).todense().flatten() 

    def get_loss(self, W):
        
        print 'loss'

        W = scipy.sparse.csc_matrix(numpy.reshape(W, (self.n_states, self.n_features)))
        if not (W.data == self.W.data).all():
            self.W = W
            self._precompute()
        
        # reward loss:
        r_err = (self.PHI * self.PHI.T * self.R - self.R)
        r_loss = 0.5 * scipy.sparse.csc_matrix.sum(r_err.multiply(r_err))

        # model loss:
        m_err = (self.PHI * self.PHI.T * self.PHI_p_norm - self.PHI_p_norm)
        m_loss = 0.5 * scipy.sparse.csc_matrix.sum(m_err.multiply(m_err))

        return (1./self.n_samples) * (r_loss + self.mod_wt * m_loss)

def init_mdp(goals = None, walls_on = False, size = 9):

    if goals is None:
        #goals = numpy.array([walls.shape]) / 2 # single goal ixn center
        goals = numpy.array([[2,2], [2,6], [6,2], [6,6]]) # four goals in the corners 
    
    walls = numpy.zeros((size,size))
    if walls_on:
        walls[:, size/2 + 1] = 1
        walls[size/2 + 1, :] = 1

    grid_world = GridWorld(walls, goals)

    rand_policy = RandomPolicy()

    mdp = MDP(grid_world, rand_policy)

    return mdp

def init_weights(shape):

    W = numpy.random.standard_normal(shape)
    inv_norms = numpy.diag(1./numpy.sqrt(numpy.sum(W**2, axis=0)))
    W = numpy.dot(W, inv_norms )
    W = scipy.sparse.csc_matrix(W)

    return W

def cg_main(n_samples = 250, size = 9, goals = None, \
        n_features = 8, mod_wt = 8):
    
    mdp = init_mdp()
    env = mdp.environment
    W = init_weights((size**2, n_features)).todense()

    n_iters = 10
    for i in xrange(n_iters):
        
        states, rewards = mdp.sample(n_samples)
        obj = SIRFObjectiveFn(W, rewards, states, env, mod_wt)
        print scipy.optimize.fmin_cg(obj.get_loss, W, fprime = obj.get_gradient)

    
def main(n_samples = 250, walls = numpy.zeros((9,9)), goals = None, \
        n_features = 8, mod_wt = 8): #, lr = 1e-2, total_decay = 1e-1):

    if goals is None:
        #goals = numpy.array([walls.shape]) / 2 # single goal ixn center
        goals = numpy.array([[2,2], [2,6], [6,2], [6,6]]) # four goals in the corners 
    
    walls[:,4] = 1
    walls[4,:] = 1
    grid_world = GridWorld(walls, goals)

    rand_policy = RandomPolicy()

    mdp = MDP(grid_world, rand_policy)
    
    n_states = numpy.prod(walls.shape) 

    W = numpy.random.standard_normal((n_states, n_features))
    inv_norms = numpy.diag(1./numpy.sqrt(numpy.sum(W**2, axis=0)))
    W = numpy.dot(W, inv_norms )
    #W[-1, :] = 0 # set bias to zero initialy
    W = scipy.sparse.csc_matrix(W)

    n_iters = 10000
    decay_rate = numpy.e**(numpy.log(total_decay) / n_iters)
    for i in xrange(n_iters):
        
        states, rewards = mdp.sample(n_samples)

        
        dW_rew, dW_mod = get_weight_update(W, states, rewards, grid_world)

        #for i in xrange(n_features):
            #plt.subplot(2,2,i)
            #plt.imshow(numpy.reshape(dW_mod.todense()[:,i], \
                #(grid_world.n_rows, grid_world.n_cols)) \
                #,interpolation = 'nearest', cmap = 'gray')
            #plt.colorbar()
        
        #plt.show()
        ##assert False

        if numpy.isnan(dW_rew.data).any():
            print 'W: ', W
            assert False
        if numpy.isnan(dW_mod.data).any():
            print 'W: ', W
            assert False
        if numpy.isnan(W.data).any():
            print 'W: ', W
        

        reward_update = lr_rew * decay_rate**i * (1./n_samples) * dW_rew
        model_update =  lr_mod * decay_rate**i * (1./n_samples) * dW_mod
        dW = reward_update + model_update
        #dW = lr_rew * decay_rate**i * (1./n_samples) * dW_rew \
           #+ lr_mod * decay_rate**i * (1./n_samples) * dW_mod
        W = W - dW    
        
        if i % 500 == 0:

            print 'iter: ', i
            print 'reward update norm: ', numpy.linalg.norm(dW_rew.data)
            print 'model update norm: ', numpy.linalg.norm(dW_mod.data)

            for i in xrange(n_features):
                plt.subplot(4,n_features,i+1)
                plt.imshow(numpy.reshape(W.todense()[:,i], \
                    (grid_world.n_rows, grid_world.n_cols)) \
                    ,interpolation = 'nearest', cmap = 'gray')
                plt.colorbar()
                plt.title('feature' + str(i))
            
            for i in xrange(n_features):
                plt.subplot(4,n_features,i+n_features+1)
                plt.imshow(numpy.reshape(dW.todense()[:,i], \
                    (grid_world.n_rows, grid_world.n_cols)) \
                    ,interpolation = 'nearest', cmap = 'gray')
                plt.colorbar()
                plt.title('update' + str(i))

            for i in xrange(n_features):
                plt.subplot(4,n_features,i+2*n_features+1)
                plt.imshow(numpy.reshape(reward_update.todense()[:,i], \
                    (grid_world.n_rows, grid_world.n_cols)) \
                    ,interpolation = 'nearest', cmap = 'gray')
                plt.colorbar()
                plt.title('reward update' + str(i))

            for i in xrange(n_features):
                plt.subplot(4,n_features,i+3*n_features+1)
                plt.imshow(numpy.reshape(model_update.todense()[:,i], \
                    (grid_world.n_rows, grid_world.n_cols)) \
                    ,interpolation = 'nearest', cmap = 'gray')
                plt.colorbar()
                plt.title('model update' + str(i))
            
            plt.show()


    for i in xrange(n_features):
        plt.subplot(2,2,i)
        plt.imshow(numpy.reshape(W[:,i].todense(), \
            (grid_world.n_rows, grid_world.n_cols)) \
            ,interpolation = 'nearest', cmap = 'gray')
        plt.colorbar()

    print W
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
            

#TODO 
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



if __name__ == "__main__":
    cg_main()


            
    
