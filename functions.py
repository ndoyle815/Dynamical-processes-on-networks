# NB: required libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.linalg import issymmetric
from matplotlib import cm

##############################################################
# function to generate a synthetic directed Watts-Strogatz
# network using the Newman-Watts algorithm
##############################################################

def Newman_Watts_generator(N, K, p):
    #N: number of nodes
    #K: number of nearest neighbours for initial configuration
    #p: probability of rewiring edge
    
    # initial configuration is K-regular ring lattice (on one side only)
    A = np.zeros((N,N))
    start_edges = []

    # for each node n add directed edge going to node n+k for k = 1, ..., K
    for i in range(N):
        for k in range(1,K+1):
            A[i,int((i+k)%N)] = 1
            start_edges.append((i,int((i+k)%10))) # collect list of edges in initial configuration
    
    # re-wiring
    for (i,j) in start_edges:
        r = np.random.rand()
        
        # re-wire edge at probability p
        if r < p:
            
            # possible new destinations for rewired directed edge
            sample = [j1 for j1 in range(N) if j1!=i and (i,j1) not in start_edges]
            newdest = np.random.choice(sample,size=1) # choose at uniform probability
            
            # rewiring
            A[i,j] = 0
            A[i,newdest] = 1
    
    # return resulting adjacency matrix
    return A


##############################################################
# function to simulate desired reaction-diffusion system on 
# a given network
##############################################################

def simulate_rdm(A, func, D, init, tspan, K):
    # A: Adjacency matrix of underlying network (NxN array)
    # func: nonlinear functions representing reaction terms (function)
    # D: diffusion coefficients (list of length 2)
    # init: initial conditions (list of size 2xN)
    # tspan: length of time to simulate (array of times to output)
    # K: mean degree of initial configuration (for demonstrative purposes only)
    
    # Generate graph Laplacian from adjacency matrix A
    N = len(A)
    Delta = (A - np.diag(np.sum(A,axis=1)))
    
    # setting up system
    sol = init
    phiold = sol[:N]
    psiold = sol[N:]
    dt = tspan[1]

    # main Forwards Euler iteration
    for t in tspan[:-1]:
        f = func(phiold,psiold)
        phinew = phiold + dt*(f[0] + D[0]*np.matmul(Delta,phiold))
        psinew = psiold + dt*(f[1] + D[1]*np.matmul(Delta,psiold))
        sol = np.vstack(( sol, np.hstack(( phinew, psinew )) ))
        phiold = phinew
        psiold = psinew
   
    # meshgrid of nodes and timescale for plotting
    nodes = np.arange(1,N+1)
    ngrid, tgrid = np.meshgrid(nodes, tspan)

    # plotting phi concentrations
    fig = plt.figure(figsize=(10,6))
    c = plt.pcolormesh(ngrid, tgrid, sol[:,:N], cmap=cm.coolwarm)
    cbar = fig.colorbar(c, shrink=0.7, aspect=5)
    #plt.clim(0,2)
    plt.xlabel('Node $i$')
    plt.ylabel('time (s)')
    if K == 0:
        plt.title('$u_i$ concentration')
    else:
        plt.title('$u_i$ concentration, $K = $ %d' % K)
    plt.show()
    
    # plotting psi concentrations
    fig = plt.figure(figsize=(10,6))
    c = plt.pcolormesh(ngrid, tgrid, sol[:,N:], cmap=cm.coolwarm)
    cbar = fig.colorbar(c, shrink=0.7, aspect=5)
    #plt.clim(0,2)
    plt.xlabel('Node $i$')
    plt.ylabel('time (s)')
    if K == 0:
        plt.title('$v_i$ concentration')
    else:
        plt.title('$v_i$ concentration, $K = $ %d' % K)
    plt.show()

    return sol


##############################################################
# Reaction functions
# NB: output is tuple of f1, f2 evaluations
# using default parameters
##############################################################

# Brusselator model
def Brusselator(u,v):
    b = 9
    c = 30
    return (1 - (b + 1)*u + c*u*u*v, b*u - c*u*u*v)

# Mimura-Murray model
def MimuraMurray(u,v):
    a = 35
    b = 16
    c = 9
    d = 0.4
    return (((a + b*u - u*u)/c - v)*u, (u - (1 + d*v))*v)

# Fitzhugh-Nagumo model
def FitzhughNagumo(u,v):
    a = 0.5
    b = 0.04
    c = 26
    return (u - u*u*u - v, c*(u - a*v - b))


##############################################################
# function to compute the dispersion relation of dynamical
# process embedded on network
##############################################################

def dispersion_relation(A, model, plot_output=True):
    # A: Adjacency matrix of underlying network (NxN array)
    # model: 'Brusselator', 'Fitzhugh-Nagumo' or 'Mimura-Murray' (string)
    # plot_output: option which allows dispersion relation to be plotted (boolean)
    
    # Fixed points and Jacobian of reaction-diffusion model
    if model == 'Brusselator':
        b = 9
        c = 30
        D = [1, 7]
        u = 1
        v = b/c
        J = np.array([[-(b+1)+2*c*u*v, c*u*u],[b-2*c*u*v, -c*u*u]])
        
    elif model == 'Fitzhugh-Nagumo':
        a = 0.5
        b = 0.04
        c = 26
        D = [0.2, 15]
        u = 0.0794976
        v = 0.0789952
        J = np.array([[1-3*u*u, -1],[c, -a*c]])
    
    elif model == 'Mimura-Murray':
        a = 35
        b = 16
        c = 9
        d = 0.4
        D = [1, 0.0125]
        u = 5
        v = 10
        J = np.array([[(a+2*b*u-3*u*u)/c-v, -u],[v, u-(1+2*d)*v]])
    
    else:
        print('Model input must be string: either Brusselator, Fitzhugh-Nagumo or Mimura-Murray')
        
    # Generate graph Laplacian from adjacency matrix A
    N = len(A)
    Delta = (A - np.diag(np.sum(A,axis=1)))
    Lambda = la.eig(Delta)[0] # eigenvalues (real if undirected, complex if directed)
    
    # DIRECTED CASE:
    if issymmetric(A):
        # trace and determinant of modified Jacobian for diffusion (real and imaginary parts)
        trRE = np.trace(J) + np.sum(D)*np.real(Lambda)
        trIM = np.sum(D)*np.imag(Lambda)
        detRE = la.det(J) + (J[0,0]*D[1] + J[1,1]*D[0])*np.real(Lambda) + D[0]*D[1]*(np.real(Lambda)**2 - np.imag(Lambda)**2)
        detIM = (J[0,0]*D[1] + J[1,1]*D[0])*np.imag(Lambda) + 2*D[0]*D[1]*(np.real(Lambda)*np.imag(Lambda))
    
        # coefficients
        A = trRE**2 - trIM**2 - 4*detRE
        B = 2*trRE*trIM - 4*detIM
    
        gamma = np.sqrt(0.5*(A + np.sqrt(A**2 + B**2)))
        delta = np.sign(B)*np.sqrt(0.5*(-A + np.sqrt(A**2 + B**2)))
    
        # output is tuple of real and imaginary parts
        lambs = (0.5*(trRE + gamma), 0.5*(trIM + delta))
    
    # UNDIRECTED CASE:
    else:
        # trace and determinant of modified Jacobian for diffusion (real and imaginary parts)
        tr = np.trace(J) + np.sum(D)*Lambda
        det = la.det(J) + (J[0,0]*D[1] + J[1,1]*D[0])*Lambda + D[0]*D[1]*Lambda**2
        
        gamma = 0.5*(tr + np.sqrt(tr**2 - 4*det))
        
        # output is tuple of real and imaginary parts
        lambs = (np.real(gamma), np.imag(gamma))
        
    if plot_output:
        fig = plt.figure(figsize=(8,6))
        plt.plot(lambs[0], lambs[1], 'ro')
        l0min = np.minimum(lambs[0].min(),-0.1)
        l0max = np.maximum(lambs[0].max(),0.1)
        l1min = np.minimum(lambs[1].min(),-0.1)
        l1max = np.maximum(lambs[1].max(),0.1)
        plt.hlines(0,2*l0min,2*l0max,color='k',linestyle='dashed')
        plt.vlines(0,2*l1min,2*l1max,color='k',linestyle='dashed')
        plt.xlim(1.2*l0min,1.2*l0max)
        plt.ylim(1.2*l1min,1.2*l1max)
        plt.xlabel('Re($\\lambda_{\\alpha}$)')
        plt.ylabel('Im($\\lambda_{\\alpha}$)')
        plt.title('Dispersion Relation: ' + model + ' model')
        plt.grid()
        
        fig = plt.figure(figsize=(8,6))
        plt.plot(-np.real(Lambda), lambs[0], 'bo')
        l0min = -1
        l0max = -1.2*np.real(Lambda).min()
        l1min = np.minimum(lambs[0].min(),-1)
        l1max = np.maximum(lambs[0].max(),1)
        plt.hlines(0,l0min,l0max,color='k',linestyle='dashed')
        plt.xlim(l0min,l0max)
        plt.ylim(1.2*l1min,1.2*l1max)
        plt.xlabel('Re($\\Lambda_{\\alpha}$)')
        plt.ylabel('Im($\\Lambda_{\\alpha}$)')
        plt.grid()
    
    return lambs