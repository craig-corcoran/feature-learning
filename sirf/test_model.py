import copy
import grid_world
import numpy
from dbel import Model
from bellman_basis import plot_features
import matplotlib.pyplot as plt
import scipy.sparse

def main():
    mdp = grid_world.MDP(walls_on = True)    
    #mdp.policy = OptimalPolicy(mdp.env, m)
    m = Model(mdp.env.R, mdp.env.P) 

    w,v = numpy.linalg.eig(m.P)
    v = v[:, numpy.argsort(w)]
    
    plot_features(numpy.real(v))
    plt.show()
    plot_features(numpy.imag(v))
    plt.show()

def simultaneous_iteration(k = 16, eps = 1e-8, lr = 1e-3):
    mdp = grid_world.MDP(walls_on = True)    
    m = Model(mdp.env.R, mdp.env.P)

    P = m.P
    phi = m.R[:,None]
    
    # initialize features as P^i * R
    for i in xrange(k-1):
        phi = numpy.hstack((phi, numpy.dot(P, m.R)[:,None]))
        P = numpy.dot(P, m.P)

    #plot_features(phi)
    #plt.show()
    
    #phi = numpy.random.standard_normal((81,k))

    a = numpy.dot(phi.T, (phi - m.gam*numpy.dot(m.P,phi)))
    b = numpy.dot(phi.T, m.R)
    w_lstd = numpy.linalg.solve(a,b) 
    err = numpy.linalg.norm(m.R - numpy.dot((phi - m.gam*numpy.dot(m.P,phi)), w_lstd))
    print 'initial bellman error: ', err

    #plt.imshow(numpy.reshape(numpy.dot(phi, w_lstd), (9,9)), interpolation = 'nearest')
    #plt.show()

    delta = numpy.inf
    be = numpy.array([])
    while delta > eps:
        
        
        phi_old = copy.deepcopy(phi)
        phi_p = numpy.dot(m.P, phi)
        q,r = numpy.linalg.qr(phi_p)
    
        phi = numpy.hstack(( m.R[:,None], q[:,:-1]))
        delta = numpy.linalg.norm(phi - phi_old)
        print 'delta: ', delta
            
        a = numpy.dot(phi.T, (phi - m.gam*numpy.dot(m.P,phi)))
        b = numpy.dot(phi.T, m.R)
        w_lstd = numpy.linalg.solve(a,b)
        err = numpy.linalg.norm(m.R - numpy.dot((phi - m.gam*numpy.dot(m.P,phi)), w_lstd))
        
        be = numpy.append(be, err)
        
        #plot_features(phi)
        #plt.show()

    #print w_lstd

    print 'final bellman error: ', err
    plt.imshow(numpy.reshape(numpy.dot(phi, w_lstd), (9,9)), interpolation = 'nearest')
    plt.show()
    
    plt.plot(range(len(be)),be)
    #plt.ylim((0,1))
    plt.show()

    plot_features(phi)
    plt.show()
        


    



if __name__ == '__main__':
    simultaneous_iteration()

    
