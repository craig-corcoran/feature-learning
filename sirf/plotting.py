import numpy
import matplotlib.pyplot as plt
import sirf

logger = sirf.get_logger(__name__)

def plot_stacked_features(phi, r = None, c = None, vmin = None, vmax = None):
    logger.info('plotting features')

    plt.clf()
    j,k = phi.shape
    if r is None:
        r = c = numpy.round(numpy.sqrt(j))
        assert r*c == j

    m = int(numpy.floor(numpy.sqrt(k)))
    n = int(numpy.ceil(k/float(m)))
    assert m * n >= k

    F = None
    for i in xrange(m):
        slic = phi[:,n*i:n*(i+1)]
        if i == 0:
            F = _stack_feature_row(slic, r, c)
        else:
            F = numpy.vstack((F, _stack_feature_row(slic, r, c)))
    F = F.astype(numpy.float64)
    plt.imshow(F, cmap='RdBu', interpolation = 'nearest', vmin = vmin, vmax = vmax)
    plt.axis('off')
    plt.colorbar()


def _stack_feature_row(phi_slice, r, c):
    for i in xrange(phi_slice.shape[1]):
        im = numpy.reshape(phi_slice[:,i], (r,c))
        I = numpy.zeros((r+2,c+2)) # pad with zeros
        I[1:-1,1:-1] = im
        if i == 0:
            F = I
        else:
            F = numpy.hstack((F,I))
    return F


def plot_features(phi, r = None, c = None, vmin = None, vmax = None):
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

def gen_spectrum(Inp, P, R):
    plt.clf()
    Inp /= numpy.sqrt((Inp * Inp).sum(axis=0))
    w,v = numpy.linalg.eig( P )#- beta * numpy.eye(m.P.shape[0]))
    v = v[:, numpy.argsort(abs(w))[::-1]]
    v = numpy.real(v)
    mag = abs(numpy.dot(Inp.T, v))
    r_mag = abs(numpy.dot(R.T, v))
    
    order = numpy.argsort(r_mag)[::-1]
    mag = mag[:, order]
    r_mag = r_mag[:, order]

    #r_mag = r_mag[r_mag > 1e-10] 
    
    # plot feature spectrums
    x = range(len(w))
    for sp in mag:
        plt.semilogy(x, sp, '.')
    plt.semilogy(range(len(r_mag)), r_mag, 'ko')
    plt.ylim(1e-6, 1)

#def make_learning_movie(movie_path, out_string):
    
    ##logger.info('Making movie animation.mpg - this make take a while')
    #mod_out_str = out_string.replace('(','').replace(')','').replace('[','').replace(']','')
    #cmd = "ffmpeg -qscale 2 -r 8 -b 10M  -i %simg_%s.png  %slearning_mov.%s.mp4" % (movie_path, '%03d', movie_path, mod_out_str)
    #os.system(cmd)
    ##os.system("ffmpeg -qscale 2 -r 4 -b 10M  -i %simg_%03d.png  %slearning_mov.%s.mp4" % (movie_path, movie_path, out_string))
    ##os.system("mencoder %s -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation.mpg" % path)

def plot_learning_curves(d_loss, switch, filt = '', ylim = None, mean_norm = False):
    plt.clf()
    ax = plt.axes()
    plotted = False
    for name, curve in d_loss.items():
        if filt in name:
            plotted = True
            x = range(len(curve))
            ax.plot(x, curve, label = name)
    
    if len(switch) > 1:
        for i in xrange(len(switch)-1):
            ax.plot([switch[i], switch[i]], [0, plt.gca().get_ylim()[1]], 'k--', label = 'training switch')
    
    if plotted:
        plt.title('Losses per CG Minibatch')
        if ylim is not None:
            plt.ylim(ylim)
        ax.legend() #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return plotted

def _plot_features(phi, r = None, c = None, vmin = None, vmax = None):
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

def plot_value_functions(size, m, IM):
    # TODO append bias when solving for value function
    # plot value functions, true and approx
    plt.clf()
    f = plt.figure()
    
    # true model value fn
    ax = f.add_subplot(311)
    ax.imshow(numpy.reshape(m.V, (size, size)), cmap = 'gray', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    # bellman error estimate (using true model)
    ax = f.add_subplot(312)
    w_be = m.get_lstd_weights(IM) # todo add lambda parameter here
    v_be = numpy.dot(IM, w_be)
    ax.imshow(numpy.reshape(v_be, (size, size)), cmap = 'gray', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # least squares solution with true value function
    ax = f.add_subplot(313)
    w_lsq = numpy.linalg.lstsq(IM, m.V)[0]
    v_lsq = numpy.dot(IM, w_lsq)
    ax.imshow(numpy.reshape(v_lsq, (size, size)), cmap = 'gray', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    logger.info('bellman error norm from v: %.2f' % numpy.linalg.norm(m.V - v_be))
    logger.info('lsq error norm from v: %.2f' % numpy.linalg.norm(m.V - v_lsq))


