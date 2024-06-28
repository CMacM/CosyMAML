import numpy as np
import pyccl as ccl

from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.special import erfinv
from scipy.stats.qmc import MultivariateNormalQMC, LatinHypercube
from scipy.stats import skew

# First we focus on generating the different redshift distributions we will use
# We will have 100 different distributions with 100 different realisations of each
# With a mean shift

# Code to generate distributions kindly provided by Markus Rau

class PZErrorRBFModel(object): 
    from sklearn.gaussian_process.kernels import RBF
    def __init__(self, nz, z_grid, std_mean_goal, num_samp=2**15): 
        """
        nz: numpy array, Sample Redshift Distribution histogram heights
        z_grid: numpy array, Midpoints corresponding to the Sample Redshift Distribuiton historam heights
        num_samp: integer, Number of samples to generate during method of moments fitting, should be power of 2
        std_mean_goal: double, Standard deviation on the mean of the sample redshift distribution log-gp process to be fitted
        
        """
        self.fid_pz = nz
        self.midpoints = z_grid 
        self.num_samp = num_samp
        self.std_mean_goal = std_mean_goal

    def fit_log_gp_model(self, par_ini=np.sqrt(np.array([0.02, 0.05]))):
    
        res = minimize(self._cost, par_ini, method='Nelder-Mead')
        return res
        
    def quasi_random_sample(self, par, num_samp, seed=None): 
        """ Using Quasi Random Sampler --> num_samp should be power of 2
        It's QUASI RANDOM sampling using Sobol squences, good for fitting method of moments but under no 
        circumstances to be used in an MCMC!!! 
        
        Can set seed to ensure the same quasi random samples are generated.
        """
        kernel = par[0]**2 * RBF(par[1]**2)
        cov = kernel(np.column_stack((self.midpoints, self.midpoints)))
        log_gp = MultivariateNormalQMC(
            mean=np.log(self.fid_pz+np.nextafter(np.float32(0),np.float32(1))),
            cov=cov,
            seed=seed
            )
        samples = np.exp(log_gp.random(num_samp))
        samples = np.array([el/np.trapz(el, self.midpoints) for el in samples])
        return samples
    
    def _cost(self, par):
        samples = self.quasi_random_sample(par, self.num_samp)
        mean_list = np.array([np.trapz(el*self.midpoints, self.midpoints) for el in samples])
        res = (np.std(mean_list) - self.std_mean_goal)**2 
    
        return res

def gen_Pz_base(mean, var, gridsize=50):
    '''Function to generate underlying redshift distribution'''
    data = np.random.normal(mean, var, size=10000)

    grid = np.linspace(0.1, 3.0, gridsize)
    midpoints = grid[:-1] + (grid[1] - grid[0]) / 0.5
    pdf_true = np.histogram(data, grid)[0]
    pz = pdf_true/np.trapz(pdf_true, midpoints)

    return pz, midpoints

def gen_Pz_samples(pz, midpoints, qrd_samples, seed=None, shift=0.01, fit_samples=2**10):
    '''Function to generate different realisations
    of redshift distribution using a gaussian process'''
    model = PZErrorRBFModel(pz, midpoints, std_mean_goal=shift, num_samp=fit_samples)
    res = model.fit_log_gp_model()
    qrand_sample = model.quasi_random_sample(res.x, qrd_samples, seed=seed)
    true_mean = np.trapz(pz*midpoints, midpoints)

    return qrand_sample, true_mean

def gen_hypercube(Omega_b, Omega_c, h, sigma8, n_s, n_samples=2**10):
    '''Function to generate hypercube of cosmological parameters'''

    # Initialise hypercube
    hyperframe = LatinHypercube(d=5)
    hyperunits = hyperframe.random(n_samples)

    # Rescale the hypercube for provided param ranges
    l_bounds = [Omega_b[0], Omega_c[0], h[0], sigma8[0], n_s[0]]
    u_bounds = [Omega_b[1], Omega_c[1], h[1], sigma8[1], n_s[1]]
    hypercube = qmc.scale(hyperunits, l_bounds, u_bounds)

    return hypercube

def gen_Cgg_autocorr(cosmo, ell, z, pz):

    shearTracer = ccl.WeakLensingTracer(cosmo, dndz=(z, pz))
    C_ell = ccl.angular_cl(cosmo, shearTracer, shearTracer, ell)

    return C_ell