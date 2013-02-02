import copy
import grid_world
import numpy
from scipy.optimize import fmin_cg
from dbel import Model
from bellman_basis import plot_features, SIBasis
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

def test_si_basis(n = 81, k = 16, patience = 1, gam = 1-1e-4, 
            max_iter = None, mb_size = 500, env_size = 9, weighting = 'policy'):
    #4.3322832102e-05
    assert n == env_size**2

    mdp = grid_world.MDP(walls_on = True, size = env_size)    
    m = Model(mdp.env.R, mdp.env.P)
    m.gam = gam

    b = SIBasis(n,k, m.gam)
    b.loss_type = 'model'

    R = m.R[:,None]
    
    # build initialization targets as P^i * R
    # TODO sample version
    C = m.R[:,None]
    P = m.P
    for i in xrange(k-1):
        C = numpy.hstack((C, numpy.dot(P, m.R)[:,None]))
        P = numpy.dot(P, m.P)

    # sample the hold out test set
    S_test, Sp_test, R_test, _, = mdp.sample_grid_world(2*mb_size, distribution = weighting)
    
    print S_test.shape
    print Sp_test.shape
    print R_test.shape

    sample_be = numpy.append(numpy.array([]), b.loss_be(b.theta, S_test, Sp_test, R_test))
    true_be = numpy.append(numpy.array([]), m.bellman_error(b.theta, weighting = weighting))
    true_lsq = numpy.append(numpy.array([]), m.value_error(b.theta, weighting = weighting))

    print 'initial test be, true be, and true lsq losses: ', sample_be[0], true_be[0], true_lsq[0]

    switch = numpy.array([])
    loss_types = ['model', 'bellman']
    
    # sample once at the beginning
    S, Sp, R, _, = mdp.sample_grid_world(mb_size, distribution = weighting)

    best_test_loss = numpy.inf
    for lt in loss_types:
        try:
            b.loss_type = lt
            n_test_inc = 0
            delta = numpy.inf
            while (n_test_inc < patience) & (delta > 1e-8):
         
                theta_old = copy.deepcopy(b.theta)
                
                # sample for each minibatch
                #S, Sp, R, _, = mdp.sample_grid_world(mb_size, distribution = weighting)
                #S = numpy.eye(n)
                #Sp = m.P

                b.set_theta( fmin_cg(b.loss, b.theta.flatten(), b.grad,
                                    args = (S, Sp, R, C),
                                    full_output = False,
                                    maxiter = max_iter, 
                                    gtol = 1e-8) )
            
                delta = numpy.linalg.norm(b.theta - theta_old)
                print 'delta: ', delta
                
                err = b.loss_be(b.theta, S_test, Sp_test, R_test) 
                sample_be = numpy.append(sample_be, err)
                true_be = numpy.append(true_be, m.bellman_error(b.theta, weighting = weighting))
                true_lsq = numpy.append(true_lsq, m.value_error(b.theta, weighting = weighting))
                
                if err < best_test_loss:
                    best_test_loss = err
                    best_theta = b.theta
                    n_best = len(sample_be)-1
                    n_test_inc = 0
                    print 'new best: ', best_test_loss
                else:
                    n_test_inc += 1
                    print 'iters without better loss: ', n_test_inc
                
                #plot_features(b.theta)
                #plt.show()

        except KeyboardInterrupt:
            print '\n user stopped current training loop'

        switch = numpy.append(switch, len(sample_be)-1)

    b.theta = best_theta
    err_test = b.loss_be(b.theta, S_test, Sp_test, R_test) 
    err_true = m.bellman_error(b.theta, weighting = weighting)
    
    print 'final bellman error (test, true): ', err_test,', ', err_true
    
    # TODO move plotting to separate script
    # plot basis functions
    plot_features(b.theta)
    plt.savefig('basis.k=%i.lam=%s.%s.pdf' % (k, str(0.), weighting))
    
    # plot learning curves
    plt.clf()
    ax = plt.axes()
    x = range(len(sample_be))
    ax.plot(x, sample_be/max(sample_be), 'r-', x, true_be/max(true_be), 'g-', x, true_lsq / max(true_lsq), 'b-')

    ypos = numpy.array([numpy.zeros_like(switch), numpy.ones_like(switch)])
    ax.plot(numpy.array([switch,switch]), ypos, 'k-')
    
    ax.plot(n_best, 0, 'go')
    ax.legend(['Test BE','True BE','True RMSE','switches', 'best theta'])
    plt.savefig('loss.k=%i.lam=%s.%s.pdf' % (k, str(0.), weighting))
    
    
    # plot value functions, true and approx
    plt.clf()
    f = plt.figure()
    
    # true model value fn
    f.add_subplot(311)
    plt.imshow(numpy.reshape(m.V, (env_size, env_size)), cmap = 'gray', interpolation = 'nearest')

    # bellman error estimate (using true model)
    f.add_subplot(312)
    w_be = m.get_lstd_weights(b.theta) # TODO add lambda parameter here
    v_be = numpy.dot(b.theta, w_be)
    plt.imshow(numpy.reshape(v_be, (env_size, env_size)), cmap = 'gray', interpolation = 'nearest')
    
    # least squares solution with true value function
    f.add_subplot(313)
    w_lsq = numpy.linalg.lstsq(b.theta, m.V)[0]
    v_lsq = numpy.dot(b.theta, w_lsq)
    plt.imshow(numpy.reshape(v_lsq, (env_size, env_size)), cmap = 'gray', interpolation = 'nearest')
    plt.savefig('value.k=%i.lam=%s.%s.pdf' % (k, str(0.), weighting))



if __name__ == '__main__':
    #simultaneous_iteration()
    test_si_basis()

    
