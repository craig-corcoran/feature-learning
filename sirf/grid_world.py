import copy
import numpy
import scipy.sparse
import scipy.optimize
from random import choice
from itertools import izip
import matplotlib.pyplot as plt

class GridWorld:
    ''' Grid world environment. State is represented as an (x,y) array and 
    state transitions are allowed to any (4-dir) adjacent state, excluding 
    walls. When a goal state is reached, a reward of 1 is given and the state
    is reinitialized; otherwise, all transition rewards are 0.

    '''

    _vecs = numpy.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) # 5 movement dirs
    n_actions = len(_vecs)
    action_to_code = dict(zip(map(tuple,_vecs),[(0, 0),(0, 1), (1, 0), (1, 1), (0, 2)]))
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
        
        # TODO should R, P be sparse vectors? - corresponding changes in dbel.model

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

    def __init__(self, env, m):
        self.env = env
        self.m = m
        self.v = m.V
    

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
    
    def __init__(self, environment=None, policy=None, walls_on = True, size = 9):
        self.env = environment
        self.policy = policy

        if environment is None:
            goals = numpy.zeros((size,size))
            goals[2,2] = 1
            print 'goal position: ', goals.nonzero()

            walls = numpy.zeros((size,size))
            if walls_on:
                walls[size/2, :] = 1
                walls[size/2, size/2] = 0 

            self.env = GridWorld(walls, goals)

        if policy is None:
            self.policy = RandomPolicy()

        self.tiles = TileFeatures(self.env.n_rows, numpy.eye(self.n_states)) # assumes square

    
    @property
    def n_states(self):
        return self.env.n_states

    @property
    def n_actions(self):
        return self.env.n_actions

    def sample_uniform_onestep(self, n_samples):
        ''' sample the interaction of policy and environment for n_samples, 
        restarting each sample in a uniformly random state in the environment.
        returns arrays of rewards, state positions, next state, action directions,
        and next action.'''

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
        
        return rewards, states, states_p, actions, actions_p

    def sample_policy(self, n_samples):
        ''' sample the interaction of policy and environment for n_samples, 
        returning arrays of rewards, state positions, and action directions '''

        states = numpy.zeros((n_samples+1,2), dtype = numpy.float)
        states[0] = self.env.state
        actions = numpy.zeros((n_samples+1,2), dtype = numpy.float)
        rewards = numpy.zeros(n_samples, dtype = numpy.float)

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
        
        return rewards, states, actions

    def sample_encoding(self, n_samples, encoding = None, append_const = True): # require reward?
        ''' returns samples from the grid world mdp
        using encoding for the features and sampling from the mdp's policy. 
        samples are returned in the order rew, state, act.'''

        rewards, states, actions = self.sample_policy(n_samples)

        if encoding == 'tabular':
            
            col_s = numpy.zeros(n_samples, dtype = numpy.int)
            col_a = numpy.zeros(n_samples, dtype = numpy.int)
            for i in xrange(n_samples + 1):
                col_s[i] = self.env.state_to_index(states[i,:])
                col_a[i] = self.env.action_to_code[tuple(actions[i,:])]
            row = numpy.arange(n_samples + 1)
            
            # use sparse matrices for tabular
            S = scipy.sparse.coo_matrix(numpy.ones(n_samples+1), numpy.hstack((row[None,:], col_s[None,:])), shape = (n_samples + 1 , self.n_states)) 
            A = scipy.sparse.coo_matrix(numpy.ones(n_samples+1), numpy.hstack((row[None,:], col_a[None,:])), shape = (n_samples + 1 , self.n_actions))

            S = scipy.sparse.csr_matrix(S)
            A = scipy.sparse.csr_matrix(A)
            

        elif encoding == 'factored':
            raise NotImplementedError 
        
        elif encoding == 'square-tile':
            
            S = numpy.zeros((n_samples+1, self.tiles.n_tiles))
            A = numpy.zeros((n_samples+1, self.tiles.n_tiles))
            print 'tiling not performed over actions, action matrix will be empty'
            for i in xrange(n_samples + 1):
                S[i,:] = self.tiles.encode_pos(states[i])
                #A[i,:] = tiles.encode_act(actions[i]) # not implemented, returns empty action matrix
            S = scipy.sparse.csr_matrix(S)
            A = scipy.sparse.csr_matrix(A)

        R = scipy.sparse.csr_matrix(rewards[:,None])

        if append_const:
            S = scipy.sparse.hstack([S, scipy.sparse.csr_matrix(numpy.ones(S.shape[0])[:,None])])
            A = scipy.sparse.hstack([A, scipy.sparse.csr_matrix(numpy.ones(S.shape[0])[:,None])])

        return R, S, A

class TileFeatures():
    
    def __init__(self, env_size, X):
        ''' builds square tile codes for a square grid world that is env_size x env_size
        and X is a matrix where each row is a low-level (tabular) representation of the state'''
        
        self.env_size = env_size
        self.tile_ind = {}
        # generate square tiles : dim, pos, ind
        ind = 0
        for si in xrange(2, env_size):
            for i in xrange(env_size - si):
                for j in xrange(env_size - si):
                    dim = (si, si)
                    pos = (i,j)
                    self.tile_ind[dim + pos] = ind
                    ind += 1

        self.ind_to_key = dict(izip(self.tile_ind.values(), self.tile_ind.keys()))
    
        # precompute encoded states
        self.d_code_vec = {}
        self.d_code_pos = {}
        for x in X:
            nz = zip(*numpy.nonzero(numpy.reshape(x, (self.env_size, self.env_size))))
            assert len(nz) == 1
            pos = nz[0]
            co = self._encode_pos(pos)
            self.d_code_vec[x.tostring()] = co
            self.d_code_pos[pos] = co
    
    # precompute encoding for each state? / cache?
    def _encode_pos(self, pos):
        ''' assumes pos is in an environment that is self.env_size square '''
        code = numpy.zeros(self.n_tiles)
        for i in xrange(self.env_size - 1):
            for j in xrange(self.env_size - 1): 
                min_size = min(pos[0] - i, pos[1] - j) + 1
                if min_size > 0:
                    max_size = min(self.env_size - i, self.env_size - j)
                    for si in xrange(max(2, min_size), max_size):            
                        if ((pos[0] - i) < si) and ((pos[1] - j) < si):
                            #_check_in_bounds(pos, (i,j), (si,si))
                            ind = self.tile_ind[(si, si, i, j)]
                            code[ind] = 1
        return code

    def encode_vec(self, vec):
        '''expects indicator vector that is len env_size^2 '''
        return self.d_code_vec[vec.tostring()]

    def encode_pos(self, pos):
        '''expects position row, col tuple '''
        return self.d_code_pos[tuple(pos)]
    
    @property
    def n_tiles(self):
        return len(self.tile_ind)
    
def test_tiles(n_samples = 100):
    
    mdp = MDP()
    r, s, a = mdp.sample_encoding(n_samples, 'square-tile')
    assert r.shape[0] == s.shape[0]-1 == a.shape[0] - 1
    
    print s.todense()
    
    env_size = mdp.env.n_rows
    print 'env size: ', env_size

    tiles = TileFeatures(mdp.env.n_rows, numpy.eye(env_size**2)) # assumes square env
    
    test_pos = numpy.array([(0,0), (env_size-1, env_size-1), (0, env_size-1), (env_size - 1, 0)])
    test_pos = numpy.vstack([test_pos, numpy.round(numpy.random.random((n_samples, 2)) * (env_size-1))])
    for pos in test_pos:
        code = tiles.encode_pos(pos)
        nz = numpy.nonzero(code)[0] # array of nonzero indices

        for ind in nz:
            key = tiles.ind_to_key[ind]
            size = key[:2] # size of tile
            corner = key[2:] # position of top left corner
            _check_in_bounds(pos, corner, size)


def _check_in_bounds(pos, corner, size):
    assert size[0] == size[1] # square
    assert corner[0] <= pos[0]
    assert corner[1] <= pos[1]

    #print pos, corner, size
    assert (pos[0] - corner[0]) <= size[0]
    assert (pos[1] - corner[1]) <= size[1]

if __name__ == '__main__':
    test_tiles()






            
   
        
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
