# cython: profile=True
import numpy
import hashlib

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

    py_grid = numpy.reshape(py_grid,(9,9))
    assert py_grid.shape == (size, size)

    #cdef numpy.ndarray[numpy.uint8_t, ndim = py_size] grid = py_grid.astype(numpy.uint8)
    grid = py_grid.astype(numpy.uint8)
    
    cdef unsigned int size = py_size
    cdef unsigned long int num_bins = py_num_bins

    cdef unsigned int p
    cdef unsigned int q
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int pq_max = edge_max + 1
    
    #cdef numpy.ndarray[numpy.uint8] window
    #cdef numpy.ndarray[numpy.uint8] dep_grid
    cdef unsigned long int index

    active_feats = {}

    print pq_max
    for p in xrange(pq_max):
        for q in xrange(pq_max):
            
            print size-p 

            for i in xrange(size-p):
                for j in xrange(size-q):

                    window = grid[i:i+p, j:j+q]

                    if pos_invariant:
                        # calculate invariant hash
                        index = inv_hash(window.flatten(), p, q)

                        if not num_bins == None:
                            index = index % num_bins
                            
                        if not active_feats.has_key(index):
                            active_feats[index] = 1
                        else: 
                            active_feats[index] += 1


                    if pos_dependent:
                        dep_grid = numpy.zeros((size,size))
                        dep_grid[i:i+p, j:j+q] = window
                        index = dep_hash(dep_grid.flatten(), p, q, i, j)

                        if not num_bins == None:    
                            index = index % num_bins

                        if not active_feats.has_key(index):
                            active_feats[index] = 1
                        else:
                            # there should be no collisions with within the 
                            # position dependent features, must be hash fn and/or col w/ invariant features
                            active_feats[index] += 1

    # TODO change to logger
    print 'size of full active (possibly hashed) set: ', len(active_feats)   
    print 'sum of counts: ', sum(active_feats.values())
    
    # returns either a dictionary of (indices, counts) or a list of indices for 
    # binary features
    if return_count:
        print 'returning dictionary'
        return active_feats
    else:
        print 'returning list' 
        return active_feats.keys()

def inv_hash(grid, p, q):
    # hashlib helps with collisions
    return int(hashlib.md5(str(grid) + str.format('{p}{q}',p=p,q=q)).hexdigest(),16) 
    #return hash(str(grid) + str.format('{p}{q}',p=p,q=q))
    

def dep_hash(grid, p, q, i, j):
    return int(hashlib.md5(str(grid) + str.format('{p}{q}{i}{j}',p=p,q=q,i=i,j=j)).hexdigest(),16)
    #return hash(str(grid) + str.format('{p}{q}{i}{j}',p=p,q=q,i=i,j=j))


