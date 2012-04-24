# cython: profile=True
import numpy
import hashlib
import random

cimport cython
cimport numpy

# TODO keep collision count?
# TODO how to handle rotational/reflective symmetries?
# TODO why are number of collisions sensitive to 1e7 vs 10**7 and adding 1
def rect_template(py_grid, edge_max = 9, py_size = 9, py_num_bins = 2**18,
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

    cdef numpy.ndarray[numpy.uint32_t, ndim = 1] primes = numpy.array([293, 43,
    349, 41, 233, 883, 379, 1039, 619, 823, 514229, 613, 97, 1597, 337, 37,
    19577194573, 241, 683, 409, 281, 211, 149, 907, 761, 463, 29, 383, 433,
    251, 167, 31, 113, 127, 5741, 780291637, 23, 863, 103, 163, 577, 709,
    3276509, 653, 94418953, 317, 2971215073, 563, 2897, 487, 179, 631, 991, 73,
    25209506681, 96557, 2922509, 28657, 79, 33461, 787, 673, 1405695061, 277,
    321534781, 67, 727, 2, 7, 151, 997, 7561, 1093, 109, 739, 617, 239, 3, 107,
    331, 157, 191, 263, 139, 193, 367, 937, 313, 13, 5, 426389, 89, 421, 643,
    19, 601, 43261, 17, 769, 1009, 397, 1686049, 283, 223, 311, 433494437, 541,
    1033, 881, 307], numpy.uint32)

    cdef unsigned int size = py_size
    cdef unsigned long long int num_bins = py_num_bins

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
                        index = inv_hash(window, p, q, num_bins, primes) # % num_bins
                            
                        if not active_feats.has_key(index):
                            active_feats[index] = 1
                        else: 
                            active_feats[index] += 1


                    if pos_dependent:
                        
                        index = dep_hash(dep_grid, p, q, i, j, num_bins, primes) # % num_bins

                        if not active_feats.has_key(index):
                            active_feats[index] = 1
                        else:
                            # there should be no collisions with within the 
                            # position dependent features, must be hash fn and/or col w/ invariant features
                            active_feats[index] += 1

    # TODO change to logger
    #print 'size of full active set: ', len(active_feats)   
    #print 'sum of counts: ', sum(active_feats.values())
    
    # returns either a dictionary of (indices, counts) or a list of indices for 
    # binary features
    if return_count:
        print 'returning dictionary'
        return active_feats
    else:
        print 'returning list' 
        return active_feats.keys()


def inv_hash(numpy.ndarray[numpy.uint8_t, ndim = 2] grid, unsigned int p, 
        unsigned int q, unsigned long long num_bins, 
        numpy.ndarray[numpy.uint32_t, ndim = 1] primes):

    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned long long hash_index = 0
    cdef unsigned int rows = grid.shape[0]
    cdef unsigned int cols = grid.shape[1]

    for i in xrange(rows):
        for j in xrange(cols):
           hash_index += grid[i,j]*primes[i*cols + j]

    hash_index += p*primes[-1]
    hash_index += q*primes[-2]

    return hash_index % num_bins
    
def dep_hash(numpy.ndarray[numpy.uint8_t, ndim = 2] grid, unsigned int p,
        unsigned int q, unsigned int i, unsigned int j, unsigned long long num_bins,
        numpy.ndarray[numpy.uint32_t, ndim = 1] primes):

    cdef unsigned int r
    cdef unsigned int c
    cdef unsigned long long hash_index = 0
    cdef unsigned int rows = grid.shape[0]
    cdef unsigned int cols = grid.shape[1]

    for r in xrange(rows):
        for c in xrange(cols):
           hash_index += grid[i,j]*primes[i*cols + j]

    hash_index += p*primes[-1]
    hash_index += q*primes[-2]
    hash_index += i*primes[-3]
    hash_index += j*primes[-4]

    return hash_index % num_bins
    


