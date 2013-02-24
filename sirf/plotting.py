import numpy
import matplotlib.pyplot as plt

def plot_features(phi, r = None, c = None, vmin = None, vmax = None):
    plt.clf()
    j,k = phi.shape
    if r is None:
        r = c = numpy.round(numpy.sqrt(j))
        assert r*c == j
        
    m = numpy.floor(numpy.sqrt(k))
    n = numpy.ceil(k/float(m))
    assert m*n >= k 
    
    f = plt.figure()
    for i in xrange(k):
            
        ax = f.add_subplot(m,n,i+1)
        im = numpy.reshape(phi[:,i], (r,c))
        ax.imshow(im, cmap = 'RdBu', interpolation = 'nearest', vmin = vmin, vmax = vmax)
        ax.set_xticks([])
        ax.set_yticks([])
