# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:21:20 2025

@author: bjama
"""

from jax import config
import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx
import matplotlib.pyplot as plt

# enable 64-bit numbers (important!)
config.update("jax_enable_x64",True)

# is this a seed?
key = jr.key(123)

# let us take 100 measurements
N = 100
noise = 0.3 # standard deviation!

# generate random 'keys'
# subkey is different from key.
key, subkey = jr.split(key)

# generate x-data:
x = jr.uniform(key=key, minval=-3.,maxval=3.,shape=(N,)) # drawn from uniform distr.
x = x.reshape(-1,1) # turn it into an Nx1 array

# define the latent function:
fun = lambda x: jnp.sin(3*x) + jnp.cos(2*x)
# this is defined over the observation points as:
signal = fun(x)
# for noisy observations, we use a normal distr:
y = signal + jr.normal(subkey,shape=signal.shape) * noise

# we then define a GPX 'dataset' as:
D = gpx.Dataset(X=x,y=y)

# we can generate points to plot the latent function:
x_lat = jnp.linspace(-3.5,3.5,500).reshape(-1,1)
y_lat = fun(x_lat)

plt.plot(x_lat,y_lat,'r',label='Latent')
plt.scatter(x,y,marker='x',label='Observations')
plt.legend()
#plt.savefig('gpjax_introplot.pdf')
plt.show()

# we want to use a zero-mean prior for the function
# with an RBF kernel. We write:
kernel = gpx.kernels.RBF() # 1D input
meanf = gpx.mean_functions.Zero()
# to construct:
prior = gpx.gps.Prior(mean_function=meanf,kernel=kernel)

# say that we want to draw samples from this prior:
prior_dist = prior.predict(x_lat)
sample = prior_dist.sample(key=key,sample_shape=(20,)) # will autofill x_lat.


plt.plot(x_lat,sample.T) # from prev. formatting
# plt.savefig('gpjax_example_samples.pdf')
plt.show()


# next, we want to construct the posterior...
# first, we need a likelihood.
# we can use a gaussian with noise parameter alpha
likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n) 
# D.n : gives length of D
# noise parameter is contained within the model (pytree)

# then, the posterior is simply constructed as:
posterior = prior * likelihood

# the params are initialised to 1.
# we can find their values by minimising the MLL (as before)

opt_posterior, history = gpx.fit_scipy(
    model = posterior,
    objective = lambda p,d: -gpx.objectives.conjugate_mll(p,d), # negative, as we are minimising.
    train_data = D
    )

print(-gpx.objectives.conjugate_mll(opt_posterior,D)) # value of the -MLL at min

# the optimised params are now contained within opt_posterior
# we can obtain our predictive distribution as:
latent_dist = opt_posterior.predict(x_lat, train_data=D)
pred_dist = opt_posterior.likelihood(latent_dist)
# extract means and stds as:
pred_mean = pred_dist.mean
pred_std = jnp.sqrt(pred_dist.variance)

# plot it all
plt.plot(x_lat,y_lat,'b--',label='Latent',alpha=0.6)
plt.scatter(x,y,label='Observations')
plt.plot(x_lat,pred_mean,'r--',label='Mean fit',alpha=0.3)
plt.plot(x_lat,pred_mean+2*pred_std,'r',alpha=0.2)
plt.plot(x_lat,pred_mean-2*pred_std,'r',alpha=0.2)
plt.fill_between(x_lat.squeeze(), pred_mean + 2*pred_std, pred_mean - 2*pred_std,color='r',alpha=0.2)
plt.legend()
plt.savefig('gpjax_result.pdf')
plt.show()