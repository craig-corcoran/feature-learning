import numpy
import hashlib

# TODO keep collision count?
# TODO how to handle rotational/reflective symmetries?
# TODO why are number of collisions sensitive to 1e7 vs 10**7 and adding 1
def rect_template(grid, edge_max = 9, size = 9, num_bins = 2**18,
        pos_invariant = True, pos_dependent = True, count = False):
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
        depending on count switch.
    '''

    grid = numpy.asarray(numpy.reshape(grid,(9,9)), numpy.int8)
    assert grid.shape == (size, size)

    active_feats = dict()
    full_list = []
    collision_count = 0 # db

    for p in xrange(edge_max+1):
        for q in xrange(edge_max+1):

            inv_list = [] # db

            for i in xrange(size-p):
                for j in xrange(size-q):
                    
                    window = grid[i:i+p, j:j+q]

                    if pos_invariant:
                        # calculate invariant hash
                        inv_index = inv_hash(window.flatten(), p, q)

                        full_list.append(inv_index) # db
                        if not num_bins == None:
                            inv_index = inv_index % num_bins
                            
                        if not active_feats.has_key(inv_index):
                            active_feats[inv_index] = 1
                        else:
                            if inv_index not in inv_list: # db
                                collision_count += 1    
                            
                            active_feats[inv_index] += 1

                        inv_list.append(inv_index) # db

                    if pos_dependent:
                        dep_grid = numpy.zeros((size,size))
                        dep_grid[i:i+p, j:j+q] = window
                        dep_grid = dep_grid.flatten()
                        dep_index = dep_hash(dep_grid, p, q, i, j)

                        full_list.append(dep_index) # db
                        if not num_bins == None:    
                            dep_index = dep_index % num_bins

                        if not active_feats.has_key(dep_index):
                            active_feats[dep_index] = 1
                        else:
                            # there should be no collisions with within the 
                            # position dependent features, must be hash fn and/or col w/ invariant features
                            active_feats[dep_index] += 1
                            collision_count += 1
    
    # TODO change to logger
    print 'size of full active (possibly hashed) set: ', len(active_feats)   
    print 'length of the full list: ', len(full_list)
    print 'sum of counts: ', sum(active_feats.values())
    print 'number of collisions: ', collision_count
    
    # returns either a dictionary of (indices, counts) or a list of indices for 
    # binary features
    if count:
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
    

