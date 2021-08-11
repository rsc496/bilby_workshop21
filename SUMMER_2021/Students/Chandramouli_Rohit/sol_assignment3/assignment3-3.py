import bilby
from bilby.core import likelihood
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.special import gamma
import emcee
import shutil
import os
import json 
label='assignment3-3_ptemcee_old'
outdir = 'outdir/assignment3-3_ptemcee_old'

def Pie(x,y):
    return (16/(3*np.pi))*(np.exp(-x**2 - (9 + 4*x**2 + 8*y)**2) + (1/2)*np.exp(-8*x**2-8*(y-2)**2))

#Plot the function in the X,Y plane
'''
Xaxis=np.arange(-4,4,0.1)
Yaxis=np.arange(-4,4,0.1)
X,Y=np.meshgrid(Xaxis, Yaxis)
Z=Pie(X,Y)
plt.rcParams["figure.figsize"] = (13,8)
PieContour=plt.contourf(X,Y,Z)
plt.xlabel('x',fontsize=15)
plt.ylabel('y',fontsize=15)
plt.colorbar()
plt.savefig('outdir/assignment3-3/contour_pie_func.png')
'''

class PieLikelihood(bilby.Likelihood):
    def __init__(self):
        super().__init__(parameters={'x':None,'y':None})
    
    def log_likelihood(self):
        x=self.parameters['x']
        y=self.parameters['y']
        return np.log(Pie(x,y))

def Chains_AC(data):
    chains=np.zeros((walkers,steps))
    AC=np.zeros(walkers)
    for i in range(0,walkers):
        chains[i,:] = np.extract(data[:,0]==i,data[:,1])
        AC[i]=emcee.autocorr.integrated_time(chains[i,:],c=5,tol=0)
    return [chains,AC.astype(int)]


def indie_chains(chains,AC):
    new_chains=[]
    for i in range(walkers):
        new_chains.append(chains[i,:][::AC[i]])
    return np.array(new_chains,dtype='object')


pielikelihood=PieLikelihood()

uniformpriors=dict(x=bilby.core.prior.Uniform(-10.,10.,name='x'),y=bilby.core.prior.Uniform(-10.,10.,name='y'))

priors = dict(x=bilby.core.prior.Uniform(-5, 5, 'x'),
              y=bilby.core.prior.Uniform(-5, 5, 'y'))
steps=20000
walkers=20
'''
result_emcee = bilby.run_sampler(
    likelihood=pielikelihood, priors=uniformpriors, sampler='emcee', nsteps=steps,nwalker = walkers,outdir=outdir, label=label,clean=True)
result_emcee.plot_corner()
'''

#analyzing the chains from the emcee run
'''
emcee_data = np.genfromtxt('outdir/assignment3-3/emcee_assignment3-3/chain.dat',skip_header=1)
x_data = emcee_data[:,[0,1]]
y_data = emcee_data[:,[0,2]]

x_chains,x_AC=Chains_AC(x_data)
y_chains,y_AC=Chains_AC(y_data)
print("AutoCorrelation per length of chain from emcee run for x-chain is "+str(np.sum(x_AC)/len(x_AC)/steps)+" and for y-chain is "+str(np.sum(y_AC)/len(y_AC)/steps))
'''
'''
indie_x_chains = indie_chains(x_chains[:,50:],x_AC)
indie_y_chains = indie_chains(y_chains[:,50:],y_AC)
plt.plot(indie_x_chains[0],color='green',alpha=0.5,label='sampled x chain 0')
plt.show()
plt.plot(indie_y_chains[9],color='blue',alpha=0.5,label='sampled y chain 9')
plt.show()


plt.hist(np.concatenate(indie_x_chains),alpha=0.5,color='green',edgecolor='black', linewidth=1.2,density=True)
plt.savefig('outdir/assignment3-3/indie_x_chains_histogram.png')
plt.show()

plt.hist(np.concatenate(indie_y_chains),alpha=0.5,color='blue',edgecolor='black', linewidth=1.2,density=True)
plt.savefig('outdir/assignment3-3/indie_y_chains_histogram.png')
plt.show()
'''
'''
#running sampler with ptmcmc
result_ptmcmc = bilby.run_sampler(
    likelihood=pielikelihood, priors=uniformpriors, sampler='ptmcmcsampler',Niter=2*10**5+1,burn=8*10**3,outdir=outdir, label=label,clean=True)
result_ptmcmc.plot_corner()
'''
#analyzing output from ptmcmc
'''
f=open('outdir/assignment3-3_ptmcmcsampler/assignment3-3_ptmcmcsampler_result.json',)
data=json.load(f)
xdata=np.asarray(data['posterior']['content']['x'])
ydata=np.asarray(data['posterior']['content']['y'])
'''
#plt.plot(xdata)
#plt.hist(xdata,alpha=0.6,color='red',edgecolor='black',linewidth=1.2,density=True)
#plt.savefig('outdir/assignment3-3_ptmcmcsampler/xdata_chain.png')

#plt.plot(ydata)
#plt.hist(ydata,alpha=0.6,color='blue',edgecolor='black',linewidth=1.2,density=True)
#plt.savefig('outdir/assignment3-3_ptmcmcsampler/ydata_chain.png')
#AC_x = emcee.autocorr.integrated_time(xdata,c=5,tol=0).astype(int)[0]
#AC_y = emcee.autocorr.integrated_time(ydata,c=5,tol=0).astype(int)[0]
#print("AutoCorrelations per length of chain from ptmcmcsampler run for x-chain is "+str(AC_x/len(xdata))+" and for y-chain is "+str(AC_y/len(ydata)))
#running sampler with ptemcee

result_ptemcee = bilby.run_sampler(
    likelihood=pielikelihood,priors=uniformpriors,sampler='ptemcee',outdir=outdir, label=label,clean=True)
result_ptemcee.plot_corner()
