# NB: required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

##############################################################
# function to generate a synthetic directed Watts-Strogatz network using the Newman-Watts algorithm
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
# function to simulate desired reaction-diffusion system on a given network
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
    plt.ylabel('time $(s)$')
    plt.title('$\phi_i$ concentration, $K = $ %d' % K)
    plt.show()
    
    # plotting psi concentrations
    fig = plt.figure(figsize=(10,6))
    c = plt.pcolormesh(ngrid, tgrid, sol[:,N:], cmap=cm.coolwarm)
    cbar = fig.colorbar(c, shrink=0.7, aspect=5)
    #plt.clim(0,2)
    plt.xlabel('Node $i$')
    plt.ylabel('time $(s)$')
    plt.title('$\psi_i$ concentration, $K = $ %d' % K)
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
