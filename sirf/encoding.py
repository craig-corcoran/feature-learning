import numpy
import scipy.sparse as sp
import matplotlib.pyplot as plt
from itertools import izip
from plotting import plot_features
import grid_world
import sirf

# assumes square env

logger = sirf.get_logger(__name__)

class TabularFeatures():
    
    tostring = 'tabular'

    def __init__(self, env_size, append_const = True):
        self.env_size = env_size
        self.B = sp.identity(self.env_size**2)
        
        if append_const:
            self.B = sp.hstack([self.B, sp.csr_matrix(numpy.ones((self.B.shape[0],1)))])

    @property
    def n_features(self):
        return self.B.shape[1]

    def encode(self, X):
        # ensure we use the sparsity of B, whether X is sparse or not
        return self.B.T.dot(X.T).T 

    def weights_to_basis(self, W):
        # multiply basis by weights
        assert W.shape[0] == self.n_features
        return self.B.dot(W)

    def __str__(self):
        return self.tostring

    def __unicode__(self):
        return u'' + self.tostring
    def __repr__(self):
        return self.tostring
    

class TileFeatures(TabularFeatures):
    
    tostring = 'tile'
    def __init__(self, env_size, append_const = True):
        ''' builds square tile codes for a square grid world that is env_size^2'''
        
        self.env_size = env_size
        self.tile_ind = {}
            
        # generate square tiles : dim, pos, ind
        ind = 0
        for si in xrange(2, env_size):
            for i in xrange(env_size - si + 1):
                for j in xrange(env_size - si + 1):
                    dim = (si, si)
                    pos = (i,j)
                    self.tile_ind[dim + pos] = ind

                    im = numpy.zeros((self.env_size, self.env_size))
                    im[pos[0]:pos[0]+dim[0], pos[1]:pos[1]+dim[1]] = 1.

                    flat_im = im.flatten()[:, None]
                    self.B = flat_im if ind is 0 else numpy.hstack([self.B, flat_im])
                    ind += 1
        
        logger.info('append constant : %s' % append_const)
        if append_const:
            self.B = numpy.hstack((self.B, numpy.ones((self.B.shape[0],1))))
        self.B = sp.csr_matrix(self.B)

        # inverse dictionary
        self.ind_to_key = dict(izip(self.tile_ind.values(), self.tile_ind.keys()))
        logger.info('total number of tiles: %i' % ind)

def test_tiles(n_samples = 100, env_size = 15):
    
    mdp = grid_world.MDP(size = env_size)
    r, s, a = mdp.sample_encoding(n_samples, 'square-tile')
    assert r.shape[0] == s.shape[0]-1 == a.shape[0] - 1
     
    env_size = mdp.env.n_rows
    n = env_size**2
    print 'env size: ', env_size
    print 'dim: ', n

    tiles = TileFeatures(env_size)
    print 'number of tiles: ', tiles.n_features
    
    test_pos = numpy.array([(0,0), (env_size-1, env_size-1), (0, env_size-1), (env_size - 1, 0)])
    test_pos = numpy.vstack([test_pos, numpy.round(numpy.random.random((n_samples, 2)) * (env_size-1))])
    X = numpy.array([indicator(mdp.state_to_index(pos), mdp.n_states ) for pos in test_pos])
    S = tiles.encode(X)
    for pos, st in izip(test_pos, S):
        nz = numpy.nonzero(st)[0] # array of nonzero indices
        for ind in nz:
            key = tiles.ind_to_key[ind]
            size = key[:2] # size of tile
            corner = key[2:] # position of top left corner
            _check_in_bounds(pos, corner, size)

    #bellman_basis.plot_features(tiles.weights_to_images(numpy.eye(tiles.n_features))[:,:121])
    plot_features(tiles.weights_to_basis(numpy.eye(tiles.n_features)))
    plt.show()

def indicator(ind, le):
    a = numpy.zeros(le)
    a[ind] = 1
    return a

def _check_in_bounds(pos, corner, size):
    assert size[0] == size[1] # square
    assert corner[0] <= pos[0]
    assert corner[1] <= pos[1]

    #print pos, corner, size
    assert (pos[0] - corner[0]) <= size[0]
    assert (pos[1] - corner[1]) <= size[1]

if __name__ == '__main__':
    test_tiles()

        
