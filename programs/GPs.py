# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 13:19:48 2025

@author: bjama
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time

SEED = 151735984
rd.seed(SEED)
np.random.seed(seed=SEED)

def kernel(x1,x2,l=1):
    # RBF kernel with variance 1
    temp = -(np.abs(x1-x2)**2)/(2*(l**2))
    return np.exp(temp)

def x(index,xmin,xmax,N):
    # i should just use linspace
    temp = xmin
    temp += index*((xmax-xmin)/(N-1))
    return temp

def gram(xmin,xmax,N,l=1):
    # make the kernel matrix
    # though K(.,.,.) makes this obsolete?
    temp = np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            temp[i,j] = kernel(x(i,xmin,xmax,N),x(j,xmin,xmax,N),l)
    return temp

def sample(xmin,xmax,N,l=1):
    # assuming zero-mean, take samples.
    covs = gram(xmin,xmax,N,l)
    sample = sp.stats.multivariate_normal.rvs(mean=None,cov=covs)
    return sample

def K(X1,X2,l):
    # X1 and X2 are tuples,
    # find K-matrix.
    N1 = len(X1) # rows
    N2 = len(X2) # columns
    temp = np.zeros((N1,N2))
    for i in range(0,N1):
        for j in range(0,N2):
            temp[i,j] = kernel(X1[i],X2[j],l)
    return temp

def GP(x_ob,y_ob,xs,fs,hyps):
    N = len(x_ob)
    # means:
    rmeans = np.dot((K(xs,x_ob,hyps[0]) @ np.linalg.inv(K(x_ob,x_ob,hyps[0])
                - (hyps[1]*np.identity(N)))),y_ob)
    # and the covariances:
    rcovs = K(xs,xs,hyps[0]) - ((K(xs,x_ob,hyps[0]) @ 
                np.linalg.inv(K(x_ob,x_ob,hyps[0]) 
                    + (hyps[1]*np.identity(N)))) @ K(x_ob,xs,hyps[0]))
    # plot latents.
    plt.plot(xs,fs,'b--',alpha=0.6,label='Latent Distribution')
    plt.scatter(x_ob,y_ob)
    # plot fit.
    plt.plot(xs,rmeans,'r--',label='Mean Fit')
    # errors?
    sigmas = np.sqrt(np.diag(rcovs))
    plt.plot(xs,rmeans+2*sigmas,'r',alpha=0.2)
    plt.plot(xs,rmeans-2*sigmas,'r',alpha=0.2)
    plt.fill_between(xs, rmeans+2*sigmas, rmeans-2*sigmas,color='r',alpha=0.2)
    plt.legend()
    # save it
    plt.title(str(N) + ' Samples, Guessed HPs')
    #plt.savefig('hp_fitted.pdf')
    plt.show()
    return 0

# draw from GP:
xmin = 0
xmax = 5
scale = 0.5
N = 100

# take the samples
sample1 = sample(xmin,xmax,N,scale)
sample2 = sample(xmin,xmax,N,scale)
sample3 = sample(xmin,xmax,N,scale)

# plot.
xs = np.linspace(xmin,xmax,N)
plt.scatter(xs,sample1,s=5,label='Sample 1')
plt.scatter(xs,sample2,s=5,label='Sample 2')
plt.scatter(xs,sample3,s=5,label='Sample 3')
plt.legend()
plt.show()
    
# now, let us take a subset of one of these samples...
# and perform GP regression!

# make a random mask, but distribute the points over the domain
rands = np.zeros(N,dtype=bool)
N_obs = 5
divs = int(np.floor(N/N_obs))
for i in range(0,N_obs):
    rands[rd.randint(i*divs,(i+1)*divs-1)] = True

# this is our observation
pure_y = sample1[rands]
obs_x = xs[rands]
# add some random noise, variance=0.01
var = 0.01
noise = sp.stats.norm.rvs(loc=0,scale=np.sqrt(var),size=N_obs)
obs_y = pure_y + noise
# plot it
plt.plot(xs,sample1,'b--',alpha=0.6,label='Latent Distribution')
plt.scatter(obs_x,obs_y)

# measure comp. time:
begin = time.time()

# now, we want to perform regression on these observations.
# using the method derived in the paper

# the mean for point x' will be given as:
# K(x',x)[K(x,x) + var*I]^-1 y

reg_means = np.dot((K(xs,obs_x,scale) @ np.linalg.inv(K(obs_x,obs_x,scale)
            - (var*np.identity(N_obs)))),obs_y)

# and the covariances:
# K(x',x') - K(x',x) [K(x,x) + var*I]^-1 K(x,x')
reg_covs = K(xs,xs,scale) - ((K(xs,obs_x,scale) @ 
            np.linalg.inv(K(obs_x,obs_x,scale) 
                + (var*np.identity(N_obs)))) @ K(obs_x,xs,scale))


# finish measurement
end = time.time()
elap = end-begin
print("Time elapsed:",elap)


# now, let us plot our prediction
# just use the means.
plt.plot(xs,reg_means,'r--',label='Mean Fit')
# errors?
sigmas = np.sqrt(np.diag(reg_covs))
plt.plot(xs,reg_means+2*sigmas,'r',alpha=0.2)
plt.plot(xs,reg_means-2*sigmas,'r',alpha=0.2)
plt.fill_between(xs, reg_means+2*sigmas, reg_means-2*sigmas,color='r',alpha=0.2)
plt.legend()
# save it
plt.title(str(N_obs) + ' Samples, True HPs')
# plt.savefig(str(N_obs)+'sample_fit.pdf')
#plt.savefig('hp_known.pdf')
plt.show()

# CHECK FOR PROGRESSION
check = input('Proceed to HP estimation?')
if check == 'y':
    # likelihood maximisation?
    def loglik(hyps):
        Ky = K(obs_x,obs_x,hyps[0]) + ((hyps[1]**2)*np.identity(N_obs))
        temp = (0.5*np.dot(obs_y,np.linalg.inv(Ky) @ obs_y)) + (0.5*np.log(np.linalg.det(Ky)))
        return temp
    # here, hyps ~ [scale,variance]
    # now, should we try to minimise this?
    # initial guess:
    h0 = np.array([1,0.1])
    res = sp.optimize.minimize(loglik,h0,method='nelder-mead')
    res.x[1] = res.x[1]**2
    print(res.x)
    GP(obs_x,obs_y,xs,sample1,res.x)
    
    # plotting hyperparameter contours
    # first, check.
    check2 = input('Create a contour plot? (CAREFUL!)')
    if check2 == 'y':
        varis = np.linspace(res.x[1]/2,res.x[1]*2,num=50)
        scales = np.linspace(res.x[0]/2,res.x[0]*2,num=50)
        
        X,Y = np.meshgrid(varis,scales)
        
        liks = np.zeros((50,50))
        for i in range(0,50):
            for j in range(0,50):
                liks[i,j] = loglik(np.array([scales[i],np.sqrt(varis[j])]))
        
        plt.contourf(X,Y,liks)
        plt.colorbar()
        plt.scatter(res.x[1],res.x[0],marker='x',c='r')
        plt.xlabel('Noise Variance')
        plt.ylabel('Length Scale')
        #plt.savefig('hp_contours.pdf')
        plt.show()
