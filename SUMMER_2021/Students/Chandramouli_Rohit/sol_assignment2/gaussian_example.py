#!/usr/bin/env python
"""
An example of how to use bilby to perform paramater estimation for
non-gravitational wave data consisting of a Gaussian with a mean and variance
"""
import bilby
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# A few simple setup steps
label = 'gaussian_example_uniform_priors_new'
outdir = 'outdir/uniform_priors_new'

# Here is minimum requirement for a Likelihood class to run with bilby. In this
# case, we setup a GaussianLikelihood, which needs to have a log_likelihood
# method. Note, in this case we will NOT make use of the `bilby`
# waveform_generator to make the signal.

# Making simulated data: in this case, we consider just a Gaussian
# default injected values : mu_inj=3, sigma_inj=4
mu_inj = 3
sigma_inj=4
data_gaussian = np.random.normal(mu_inj, sigma_inj, 10000)
#plt.hist(data_gaussian,bins=100)
#plt.axvline(mu_inj,color='red')
#plt.show()
#exit()

#data from student-t distribution
# default injected values: df = 3, loc = 3, scale = 4
df= 2
loc=3
scale=4
data_student_t = t.rvs(df,loc,scale,10000)

class SimpleGaussianLikelihood(bilby.Likelihood):
    def __init__(self, data):
        """
        A very simple Gaussian likelihood

        Parameters
        ----------
        data: array_like
            The data to analyse
        """
        super().__init__(parameters={'mu': None, 'sigma': None})
        self.data = data
        self.N = len(data)

    def log_likelihood(self):
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        res = self.data - mu
        #return -0.5
        return -0.5 * (np.sum((res / sigma)**2) +
                       self.N * np.log(2 * np.pi * sigma**2))
                       
class Exponential(bilby.core.prior.Prior):
    """Define a new prior class where p(x) ~ exp(alpha * x)"""
    
    def __init__(self, alpha, minimum, maximum, name=None, latex_label=None):
        super(Exponential, self).__init__(name=name, latex_label=latex_label, minimum=minimum, maximum=maximum)
        self.alpha = alpha
        
    def rescale(self, val):
        bilby.prior.Prior.test_valid_for_rescaling(val)
        normalization=(2-np.exp(-self.alpha*abs(self.minimum))-np.exp(-self.alpha*abs(self.maximum)))
        return np.where(val < (1-np.exp(-self.alpha*abs(self.minimum)))/normalization,
                        1/self.alpha*np.log(normalization*val+np.exp(-self.alpha*abs(self.minimum))),-1/self.alpha*np.log(2-normalization*val-np.exp(-self.alpha*abs(self.minimum)))) 

    def prob(self, val):
        in_prior = (val >= self.minimum) & (val <= self.maximum) 
        
        return -self.alpha * np.exp(-self.alpha * abs(val)) / (-2+np.exp(-self.alpha * abs(self.maximum))
                                                        + np.exp(-self.alpha * abs(self.minimum))) * in_prior

likelihood = SimpleGaussianLikelihood(data_gaussian)
#uniform prior
priors = dict(mu=bilby.core.prior.Uniform(0, 5, 'mu'),
              sigma=bilby.core.prior.Uniform(0, 10, 'sigma'))
#non-uniform priors
#Cauchy prior that is built into Bilby
cauchy_priors=dict(mu=bilby.core.prior.analytical.Cauchy(mu_inj,2,name='mu'),
                    sigma=bilby.core.prior.analytical.Cauchy(2,2,name='sigma'))
#Normal prior
normal_priors=dict(mu=bilby.core.prior.analytical.Normal(sigma_inj,2,name='mu'),
                    sigma=bilby.core.prior.analytical.Normal(5,2,name='sigma'))
#Exponential prior
my_priors=dict(mu=Exponential(name='mu',alpha=1,minimum=0,maximum=5),sigma=Exponential(name='sigma',alpha=1,minimum=0,maximum=10))

# And run sampler
# default settings: 
#npoints = 500, walks = 10, data= data_gaussian
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='emcee', iterations=10000,
    walks=20, outdir=outdir, injection_parameters=dict(mu=mu_inj,sigma=sigma_inj) ,nburn=1000,label=label,clean=True)
result.plot_corner()

