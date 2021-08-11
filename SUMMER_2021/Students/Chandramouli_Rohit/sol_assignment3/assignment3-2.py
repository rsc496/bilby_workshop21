import bilby
from bilby.core import likelihood
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.special import gamma
import emcee
import shutil
import os

label='assignment3-2_thinned'
outdir = 'outdir/assignment3-2_thinned'

def studentT(x,mu,nu,sigma):
    res= (x-mu)/sigma
    return gamma((nu+1)/2)/(np.sqrt(nu*np.pi)*sigma*gamma(nu/2))*(1+1/nu*res**2)**(-(nu+1)/2)

class TLikelihood(bilby.Likelihood):
    def __init__(self,data):
        super().__init__(parameters={'mu':None,'nu':None,'sigma':None})
        self.data=data
        self.N = len(data)
    def log_likelihood(self):
        mu = self.parameters['mu']
        nu = self.parameters['nu']
        sigma = self.parameters['sigma']
        data=self.data
        return np.sum(np.log(studentT(data,mu,nu,sigma)))

t_data = np.genfromtxt('outdir/assignment3-1/assignment3-1_final_samples.dat')
t_data_100 = t_data[:100]
print(len(t_data))
min_range=min(t_data)
max_range=max(t_data)
bins=np.arange(min_range,max_range,0.5)
fig2=plt.figure(figsize=(10,8))
#plt.hist(t_data,bins,alpha=0.5,color='green',edgecolor='black', linewidth=1.2,density=True)
#plt.show()

mu_inj = 1
sigma_inj=1
nu_inj=3
steps=10000
walkers=10

'''
likelihood = TLikelihood(t_data)
priors=dict(mu=bilby.core.prior.Uniform(-5, 5, 'mu'), sigma=bilby.core.prior.Uniform(0.1, 10, 'sigma'), nu=bilby.core.prior.Uniform(0.1, 10, 'nu'))
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='emcee',nsteps=steps,nwalker = walkers, nburn=300 ,outdir=outdir, injection_parameters=dict(mu=mu_inj,sigma=sigma_inj,nu=nu_inj) ,label=label,clean=True)
result.plot_corner()
'''
#thinning
sampled_data = np.genfromtxt('outdir/assignment3-2_thinned/emcee_assignment3-2_thinned/chain.dat',skip_header=1)
mu_samples=sampled_data[:,[0,1]]
sigma_samples=sampled_data[:,[0,2]]
nu_samples=sampled_data[:,[0,3]]

def Chains_AC(data):
    chains=np.zeros((walkers,steps))
    AC=np.zeros(walkers)
    for i in range(0,walkers):
        chains[i,:] = np.extract(data[:,0]==i,data[:,1])
        AC[i]=emcee.autocorr.integrated_time(chains[i,:],c=5,tol=0)
    return [chains,AC.astype(int)]

mu_chains,mu_AC = Chains_AC(mu_samples)
sigma_chains,sigma_AC = Chains_AC(sigma_samples)
nu_chains,nu_AC = Chains_AC(nu_samples)

def indie_chains(chains,AC):
    new_chains=[]
    for i in range(walkers):
        new_chains.append(chains[i,:][::AC[i]])
    return np.array(new_chains,dtype='object')

indie_mu_chains = indie_chains(mu_chains[:,300:],mu_AC)
indie_nu_chains = indie_chains(nu_chains[:,300:],nu_AC)
indie_sigma_chains = indie_chains(sigma_chains[:,300:],sigma_AC)

#plt.plot(mu_chains[0])
plt.plot(np.concatenate(indie_mu_chains),color='red')
plt.show()


plt.hist(np.concatenate(indie_mu_chains),alpha=0.5,color='green',edgecolor='black', linewidth=1.2,density=True)
plt.axvline(mu_inj)
plt.show()

plt.hist(np.concatenate(indie_nu_chains),alpha=0.5,color='green',edgecolor='black', linewidth=1.2,density=True)
plt.axvline(nu_inj)
plt.show()

plt.hist(np.concatenate(indie_sigma_chains),alpha=0.5,color='green',edgecolor='black', linewidth=1.2,density=True)
plt.axvline(sigma_inj)
plt.show()