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

    _vecs = numpy.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) # 4 movement dirs
    action_to_code = dict(zip(map(tuple,_vecs),[(0, 0),(0, 1), (1, 0), (1, 1)]))
    code_to_action = dict(zip(action_to_code.values(), action_to_code.keys()))

    def __init__(self, wall_matrix, goal_matrix, init_state = None, uniform = False):
        
        self.walls = wall_matrix
        self.goals = goal_matrix
        goal_list = zip(*self.goals.nonzero())

        self.n_rows, self.n_cols = self.walls.shape
        self.n_states = self.n_rows * self.n_cols
        self.n_act_var = 2

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
            #P[adj_ids,idx] = 1.
            P[idx, adj_ids] = 1.

        # normalize rows? to have unit sum
        self.P = numpy.dot(numpy.diag(1./(numpy.sum(P, axis=1)+1e-14)), P)
        
        # build reward function R
        self.R = numpy.zeros(self.n_states)
        nz = zip(*self.goals.nonzero())
        gol_ids = map(self.state_to_index, nz)
        self.R[gol_ids] = 1
        
        if uniform: # use uniform diagonal weight matrix D
            self.D = scipy.sparse.dia_matrix(([1]*self.n_states, 0), \
                (self.n_states, self.n_states))
            assert self.D == scipy.sparse.csc_matrix(numpy.eye(self.n_states))
        else:
            # find limiting distribution (for measuring Bellman error, etc)
            v = numpy.zeros((self.P.shape[0],1))
            v[1,0] = 1
            delta = 1
            while  delta > 1e-12:
                v_old = copy.deepcopy(v)
                v = numpy.dot(self.P,v)
                v = v / numpy.linalg.norm(v)
                delta = numpy.linalg.norm(v-v_old)

            self.D = scipy.sparse.csc_matrix(numpy.diag(v[:,0]))

            

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

        # if at goal position, 
        if self.goals[tuple(self.state)] == 1:
            return self.random_state() # reinitialize to rand state

        pos = self.state + action
        if self._check_valid_state(pos):
            return pos
        else:
            return self.state

    def is_goal_state(self):
        if self.goals[tuple(self.state)] == 1:
            return True
        return False
        
    def take_action(self, action):
        '''take the given action, if valid, changing the state of the 
        environment. Return resulting state and reward. '''
        rew = self.get_reward(self.state)
        self.state = self.next_state(action)
        return self.state, rew


    def get_reward(self, state):
        ''' sample reward function for a given afterstate. Here we assume that 
        reward is a function of the state only, not the state and action '''
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

class OptimalPolicy:
    ''' acts according to the value function of a random agent - should be 
    sufficient in grid world'''

    def __init__(self, env, gam = 0.99):
        self.env = env
        self.v = value_iteration(env.P, env.R, gam, eps = 1e-4)
    

    def choose_action(self, actions):
        
        max_val = -numpy.inf
        act = None
        for a in actions:
            next_s = self.env.next_state(a)
            next_s = self.env.state_to_index(next_s)
            val = self.v[next_s]
            if val > max_val:
                act = a
                max_val = val
        assert act is not None

        return act

class MDP:
    
    def __init__(self, environment=None, policy=None, walls_on = True):
        self.env = environment
        self.policy = policy

        if environment is None:
            size = 9
            goals = numpy.zeros((size,size))
            goals[1,1] = 1
            print 'goal position: ', goals.nonzero()

            walls = numpy.zeros((size,size))
            if walls_on:
                walls[size/2, :] = 1
                walls[size/2, size/2] = 0 

            self.env = GridWorld(walls, goals)

        if policy is None:
            self.policy = RandomPolicy()


    def _sample(self, n_samples, distribution = 'policy'):

        ''' sample the interaction of policy and environment according to the 
        given distribution for n_samples, returning arrays of state positions 
        and rewards '''
        
        if distribution is 'uniform':
        
            print 'sampling with a uniform distribution'
    
            
            states   = numpy.zeros((n_samples,2), dtype = numpy.int)
            states_p = numpy.zeros((n_samples,2), dtype = numpy.int)
            actions = numpy.zeros((n_samples,2), dtype = numpy.int)
            actions_p = numpy.zeros((n_samples,2), dtype = numpy.int)
            rewards = numpy.zeros(n_samples, dtype = numpy.int)
            
            for i in xrange(n_samples):

                self.env.state = self.env.random_state()
                s = copy.deepcopy(self.env.state)
        
                choices = self.env.get_actions(self.env.state)
                a = self.policy.choose_action(choices)
                s_p, r = self.env.take_action(a)
            
                choices = self.env.get_actions(self.env.state)
                a_p = self.policy.choose_action(choices)
                
                states[i] = s
                states_p[i] = s_p
                actions[i] = a
                actions_p[i] = a_p
                rewards[i] = r
            
            s = states
            s_p = states_p
            a = actions
            a_p = actions_p


        elif distribution is 'policy':

            print 'sampling with a policy distribution'

            states = numpy.zeros((n_samples+1,2), dtype = numpy.int)
            states[0] = self.env.state
            actions = numpy.zeros((n_samples+1,2), dtype = numpy.int)
            rewards = numpy.zeros(n_samples, dtype = numpy.int)

            for i in xrange(n_samples):
                    
                choices = self.env.get_actions(self.env.state)
                action = self.policy.choose_action(choices)
                next_state, reward = self.env.take_action(action)

                states[i+1] = next_state
                actions[i] = action
                rewards[i] = reward
        
            choices = self.env.get_actions(self.env.state)
            action = self.policy.choose_action(choices)
            actions[i+1] = action

            s   = states[:-1,:]
            s_p = states[1:, :]
            a = actions[:-1,:]
            a_p = actions[1:,:]
            
        else:
            print 'bad distribution string'
            assert False

        return s, s_p, a, a_p, rewards


    def sample_grid_world(self, n_samples, state_rep = 'tabular', 
                                                distribution = 'policy'):
        ''' returns a matrix X whos rows are samples from the grid world mdp
        using state_rep for the features and sampling from the given 
        distribution type, either on-policy or uniform (one-step). Rows
        of X are in the form [s,s',r,a]'''

        states, states_p, actions, actions_p, rewards = self._sample(n_samples, distribution)

        if state_rep == 'tabular':
            n_state_var = self.env.n_states
        elif state_rep == 'factored':
            n_state_var = self.env.n_rows + self.env.n_cols

        X = numpy.zeros((n_samples, 2 * n_state_var + self.env.n_act_var + 1)) 

        for i in xrange(n_samples):

            if state_rep == 'tabular':
                X[i,self.env.state_to_index(states[i,:])] = 1
                X[i,n_state_var + self.env.state_to_index(states_p)] = 1
                X[i,2*n_state_var] = rewards[i]
                X[i,2*n_state_var + 1:] = self.env.action_to_code[tuple(actions[i,:])]

            elif state_rep == 'factored':
                state = states[i,:]
                state_p = states[i,:]
                
                # todo add standard encoding function
                # encode row and col position
                X[i,state[0]] = 1 
                X[i,self.env.n_rows + state[1]] = 1
                X[i,n_state_var + state_p[0]] = 1
                X[i,n_state_var + self.env.n_rows + state_p[1]] = 1
                X[i,2*n_state_var] = rewards[i]     
                X[i,2*n_state_var + 1:] = self.env.action_to_code[tuple(actions[i,:])]

        return X

def value_iteration(P, R, gam = (1-2*1e-2), eps = 1e-4):
    '''solve for true value function using value iteration '''
    V = numpy.ones((P.shape[0], 1), dtype = numpy.float64) / P.shape[0]
    if R.ndim == 1:
        R = R[:,None]
    
    delta = 1e4
    while numpy.linalg.norm(delta) > eps:
        delta = R + gam * numpy.dot(P, V) - V
        V = V + delta

    #plt.imshow(numpy.reshape(V, (9,9)), interpolation = 'nearest', cmap = 'jet')
    #plt.colorbar()
    #plt.show()
    return V

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
            plt.subplot(n_features/5, 5 , i + 1)
            plot_im(numpy.reshape(W[:,i], (n_rows, n_cols)))

    plt.show()

def plot_im(W):
    plt.imshow(W, interpolation = 'nearest', cmap = 'gray')
    plt.colorbar()


def main():
    mdp = MDP()

    X = mdp.sample_grid_world(100)

    R = mdp.env.R
    P = mdp.env.P
    
    v = value_iteration(P,R)


if __name__ == '__main__':
    main()






            
   
        
#def test_grid_world():

    #mdp = init_mdp()

    #grid_world = mdp.env
    
    #states, states_p, actions, actions_p, rewards = mdp.sample(100)

    #assert len(states) == len(rewards) == len(states_p)
    #assert len(states) == len(actions) == len(actions_p)

    ## assert states are all in bounds
    #assert len(states[states < 0]) == 0
    #x_pos = states[:,0]
    #y_pos = states[:,1]
    #assert len(x_pos[ x_pos >= grid_world.n_rows ]) == 0
    #assert len(y_pos[ y_pos >= grid_world.n_cols ]) == 0
    
    #X = sample_grid_world(100) 
