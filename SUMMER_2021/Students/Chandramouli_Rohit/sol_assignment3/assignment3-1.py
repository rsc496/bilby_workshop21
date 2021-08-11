import bilby
from bilby.core import likelihood
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.special import gamma
import emcee
import shutil
import os

label='assignment3-1'
outdir = 'outdir/assignment3-1'

mu_global = 1
nu_global = 3
sigma_global = 1

class StudentTLikelihood(bilby.Likelihood):
    def __init__(self):
        super().__init__(parameters={'x':None})
               
    def log_likelihood(self):
        x = self.parameters['x']
        res = (x - mu_global)/sigma_global
        l = gamma((nu_global+1)/2)/(np.sqrt(nu_global*np.pi)*sigma_global*gamma(nu_global/2))*(1+1/nu_global*res**2)**(-(nu_global+1)/2)
        return np.log(l)

priors = dict(x=bilby.core.prior.Uniform(-10,10,name='x'))
x_init=1

t_likelihood = StudentTLikelihood()
steps=20000
walkers=20
result = bilby.run_sampler(
    likelihood=t_likelihood, priors=priors, sampler='emcee', nsteps=steps,nwalker = walkers, injection_parameters=dict(x=x_init),outdir=outdir, label=label,clean=True)
result.plot_corner()

data = np.genfromtxt('outdir/assignment3-1/emcee_assignment3-1/chain.dat',skip_header=1)

def Chains_AC(data):
    chains=np.zeros((walkers,steps))
    AC=np.zeros(walkers)
    for i in range(0,walkers):
        chains[i,:] = np.extract(data[:,0]==i,data[:,1])
        AC[i]=emcee.autocorr.integrated_time(chains[i,:],c=5,tol=0)
    return [chains,AC.astype(int)]

chains,AC = Chains_AC(data)

def indie_chains(chains,AC):
    new_chains=[]
    for i in range(walkers):
        new_chains.append(chains[i,:][::AC[i]])
    return np.array(new_chains,dtype='object')

new_chains = indie_chains(chains,AC)
for i in range(walkers):
    print(len(new_chains[i]))

fig1 = plt.figure(figsize=(10,8))
plt.plot(new_chains[0],color='green',alpha=0.5,label='sampled Markov chain 0')
plt.plot(new_chains[1],color='red',alpha=0.5,label='sampled Markov chain 1')
plt.plot(new_chains[7],color='blue',alpha=0.5,label='sampled Markov chain 7')
plt.show()

final_samples=np.concatenate(new_chains)

min_range=min(final_samples)
max_range=max(final_samples)
bins=np.arange(min_range,max_range,0.5)
fig2=plt.figure(figsize=(10,8))
plt.hist(final_samples,bins,alpha=0.5,color='green',edgecolor='black', linewidth=1.2,density=True)
plt.show()
np.savetxt('outdir/assignment3-1/assignment3-1_final_samples.dat',final_samples)