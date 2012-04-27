# cython: profile=True
import numpy 
import random
import mmh3

cimport cython
cimport numpy

# TODO how to handle rotational/reflective symmetries?
def rect_template(py_grid, edge_max = 9, py_size = 9, num_bins = 2**18,
        pos_invariant = True, pos_dependent = True, return_count = False):
    
    ''' 

        calculates the sparse feature mapping for all rectangular templates up
        to edge_max by edge_max. Can evaluate both position dependent and position
        invariant templates. 

        params: 
    
        grid - numpy array of go board
        edge_max - max size of template rectangles
        size - size of go board
        num_bins - number to limit feature indices to
        pos_invariant - boolean switch for evaluating position invariant features
        pos_dependent - boolean switch for evaluating position dependent features
        count - boolean switch for outputting count dictionary of (index,count) 
            pairs. Returns list of the nonzero indices if False

        returns a dictionary of (feature index, count) or list of nonzero indices 
        depending on the return_count switch.
    '''

    cdef unsigned int size = py_size
    cdef unsigned long long int n_bins = num_bins

    py_grid = numpy.reshape(py_grid,(9,9))
    print py_grid.shape
    assert py_grid.shape == (size, size)

    cdef numpy.ndarray[numpy.uint8_t, ndim = 2] grid = py_grid.astype(numpy.uint8)

    cdef unsigned int p
    cdef unsigned int q
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int iz
    cdef unsigned int jz
    cdef unsigned int pi
    cdef unsigned int qi
    cdef unsigned int pq_max = edge_max + 1
    
    cdef numpy.ndarray[numpy.uint8_t, ndim = 2] window
    cdef numpy.ndarray[numpy.uint8_t, ndim = 2] dep_grid = numpy.zeros((size, size), dtype = numpy.uint8)
    cdef unsigned long long int index

    active_feats = {}

    print pq_max
    for p in xrange(pq_max):
        for q in xrange(pq_max):
    
            # only make a new array once per window size
            window = numpy.zeros((p, q), dtype = numpy.uint8)

            for i in xrange(size-p):
                for j in xrange(size-q):
                    
                    if pos_dependent:
                        # dep_grid = zeros # does this hurt?
                        for iz in xrange(size):
                            for jz in xrange(size):
                                dep_grid[i,j] = 0
                        
                    # window = grid[i:i+p, j:j+q]
                    for pi in xrange(p):
                        for qi in xrange(q):
                            window[pi, qi] = grid[i+pi, j+qi]
                            if pos_dependent:
                                dep_grid[i+pi, j+qi] = window[pi, qi] 


                    if pos_invariant:
                        # calculate invariant hash
                        index = inv_hash(window, p, q, n_bins) # % n_bins
                            
                        if not active_feats.has_key(index):
                            active_feats[index] = 1
                        else: 
                            active_feats[index] += 1


                    if pos_dependent:
                        
                        index = dep_hash(dep_grid, p, q, i, j, n_bins) # % n_bins

                        if not active_feats.has_key(index):
                            active_feats[index] = 1
                        else:
                            # there should be no collisions with within the 
                            # position dependent features, must be hash fn and/or col w/ invariant features
                            active_feats[index] += 1

    
    # returns either a dictionary of (indices, counts) or a list of indices for 
    # binary features
    if return_count:
        return active_feats
    else: 
        return active_feats.keys()


def inv_hash(numpy.ndarray[numpy.uint8_t, ndim = 2] grid, unsigned int p, 
        unsigned int q, unsigned long long num_bins):
    
    return mmh3.hash64(''.join([grid.tostring(), str(p), str(q)]))[0] % num_bins
    
def dep_hash(numpy.ndarray[numpy.uint8_t, ndim = 2] grid, unsigned int p,
        unsigned int q, unsigned int i, unsigned int j, unsigned long long num_bins):
        
    return mmh3.hash64(''.join([grid.tostring(), str(p), str(q), str(i), str(j)]))[0] % num_bins

