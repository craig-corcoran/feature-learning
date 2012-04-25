'''
Created on Apr 18, 2012

@author: craig
'''
import nose.tools
import numpy
import pyfeat
import vowpal

def test_go_player():
    pass

def test_template_maps():
    grid = numpy.round(2*numpy.random.random((9,9))-1)
    
    # check that the dictionary counts of each class separately sums to count
    # for both classes
    count_dict = pyfeat.feature_maps.rect_template(grid, count = True)
    total_counts = sum(count_dict.values())
    
    count_dict = pyfeat.feature_maps.rect_template(grid, pos_dependent = False, count = True)
    inv_cnt = sum(count_dict.values())
    count_dict = pyfeat.feature_maps.rect_template(grid, pos_invariant = False, count = True)
    dep_cnt = sum(count_dict.values())

    assert total_counts == (inv_cnt + dep_cnt)
    
    # check that there are no collisions between the classes of features (dep,inv)
    # by doing the same as above but with only the binary features (no count info)
    active = pyfeat.feature_maps.rect_template(grid)
    total_active = len(active)
    print total_active
    
    active = pyfeat.feature_maps.rect_template(grid, pos_dependent = False)
    inv_act = len(active)
    active = pyfeat.feature_maps.rect_template(grid, pos_invariant = False)
    dep_act = len(active)

    assert total_active == (dep_act + inv_act)

    # see that we do get collisions when num_bins is set below ~10**7
    active = pyfeat.feature_maps.rect_template(grid, num_bins = 10**4)
    hashed_active = len(active)

    assert hashed_active < total_active

    # but we can avoid collisions by making the array large enough
    active = pyfeat.feature_maps.rect_template(grid, num_bins = 10**10)
    hashed_active = len(active)
    
    assert hashed_active == total_active

    # check that the invariant features are invariant to translated boards
    window = numpy.round(2*numpy.random.random((3,3))-1)
    board1 = numpy.zeros((9,9)); board2 = numpy.zeros((9,9))
    board1[0:3,0:3] = window; board1 = numpy.asarray(board1, numpy.int8)
    board2[-3:,-3:] = window; board2 = numpy.asarray(board2, numpy.int8)

    active1 = pyfeat.feature_maps.rect_template(board1, pos_dependent = False, edge_max = 3)    
    active2 = pyfeat.feature_maps.rect_template(board2, pos_dependent = False, edge_max = 3)
    active3 = pyfeat.feature_maps.rect_template(numpy.zeros((9,9)), pos_dependent = False, edge_max = 3)
    
    assert len(set(active1).intersection(set(active2))) > len(set(active1).intersection(set(active3)))
    assert len(set(active1).intersection(set(active2))) > len(set(active2).intersection(set(active3)))

    # check that the position dependent features are not invariant to translated
    # boards, but do have a constant number of active features
    active1 = pyfeat.feature_maps.rect_template(board1, pos_invariant = False)    
    active2 = pyfeat.feature_maps.rect_template(board2, pos_invariant = False)

    assert active1 != active2
    assert len(active1) == len(active2)

    # assert that the total feature counts for all boards are the same
    total_count1 = sum(pyfeat.feature_maps.rect_template(board1, count = True).values())
    total_count2 = sum(pyfeat.feature_maps.rect_template(board1, count = True).values())

    assert total_counts == total_count1 == total_count2

def test_vw_stream(num_examples = 20, path_vw = '/usr/local/bin/vw' ):
    """
        Predicting from an ExampleStream.  An ExampleStream basically
        writes an input file for you from VowpalExample objects.
        All training examples (value != None) must appear before test examples.
    """
    stream = vowpal.ExampleStream('vw.stream.txt')
    examples = []
    for i in xrange(num_examples):

        grid = numpy.round(2*numpy.random.random((9,9))) # 0,1,2

        #active_set = pyfeat.go.hash_maps.rect_template(grid, py_num_bins = 10**5)
        active_set = pyfeat.go.py_hash_maps.rect_template(grid)
        
        if i >= num_examples*(3/4.):
            value = None
        else:
            value = numpy.random.random()

        # {'temp_feats' : {32 : None, 12: None, ..., idx: None} } for count = False (default)
        # {'temp_feats' : {32 : 2, 12 : 1 , ..., idx: count} } for count=True
        all_sections = dict([ ( 'temp_feats', dict( 
                        zip(active_set, [None]*len(active_set))) )
                        ])
        #all_sections = dict(zip(active_set, [None]*len(active_set)))
                        

        ex = vowpal.VowpalExample(i, value)

        for (namespace, section) in all_sections.items():
            ex.add_section(namespace, section)
        stream.add_example(ex)
    
    print 'running through examples in vw'
    vw = vowpal.Vowpal(path_vw, './vw.%s', {'--passes' : '200' })
    preds = vw.predict_from_example_stream(stream)
    
    print 'print values from predictions: ', preds
    for (id, value) in preds:
        print 'prediction for %s is %s' % (id, value)

def test_vw_file( path_vw = '/usr/local/bin/vw'):
 
    vw = vowpal.Vowpal(path_vw, './vw.%s', {'--passes' : '200' })
    preds = vw.predict_from_file('vw.file.txt')
    
    print 'print values from predictions: ', preds
    for (id, value) in preds:
        print 'prediction for %s is %s' % (id, value)
    
if __name__ == '__main__':
    test_vw_stream()
