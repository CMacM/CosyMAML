import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import emcee
import torch
import os
import arviz as az

import src.training as training
import src.models as models

from time import time
from importlib import reload
reload(training)
reload(models)
from IPython.display import clear_output

class EmulatorHook():
    def __init__(self, model, ScalerX, ScalerY, weights=None, device=None):
        self.model = model
        self.ScalerX = ScalerX
        self.ScalerY = ScalerY
        self.weights = weights

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def __call__(self, theta_batch):
        # Assume theta_batch has shape (n_walkers, n_params)
        # Set to eval mode
        self.model.eval()

        # Send inputs to PyTorch tensor and apply scaling
        theta_tensor = torch.tensor(theta_batch, dtype=torch.float32).to(self.device)
        theta_tensor = self.ScalerX.transform(theta_tensor)

        # Compute the model vector in a batch
        with torch.no_grad():
            model_vectors = self.model(theta_tensor, params=self.weights) # Shape: (n_walkers, output_dim)
            model_vectors = torch.exp(self.ScalerY.inverse_transform(model_vectors))  # Inverse transform

        # Transfer results to CPU and convert to numpy
        model_vectors = model_vectors.cpu().detach().numpy()

        return model_vectors
    
# Define the likelihood function to use emulator with batching
def log_likelihood(theta_batch, data, inv_cov, ClHook):

    # Call to emulator hook
    model_vectors = ClHook(theta_batch)
    
    # Compute the log likelihood for each walker
    diffs = data - model_vectors  # Shape: (n_walkers, output_dim)
    log_likes = -0.5 * np.einsum("ij,ij->i", diffs, np.dot(inv_cov, diffs.T).T)
    
    return log_likes  # Return a vector of log-likelihoods

def log_prior(theta, priors):
    # Vectorized check of priors
    in_bounds = np.array([all(prior[0] < t < prior[1] for prior, t in zip(priors, theta_row)) for theta_row in theta])
    return np.where(in_bounds, 0.0, -np.inf)

def log_probability(theta_batch, priors, data, inv_cov, ClHook):
    lp = log_prior(theta_batch, priors)
    # Only proceed with log-likelihood computation for valid prior cases
    mask = np.isfinite(lp)
    ll = np.full_like(lp, -np.inf)  # Initialize ll with -inf
    ll[mask] = log_likelihood(theta_batch[mask], data, inv_cov, ClHook)  # Compute only for valid priors
    
    return lp + ll  # Returns an array of log-probabilities

def run_emcee(
        ClHook,
        backend_file,
        ndim,
        priors,
        pos,
        data_vector,
        inv_cov,
        nwalkers=128, 
        n_check=1000,
        max_iter=50000, 
        clobber=False
    ):

    #Check for existing backend file
    if os.path.exists(backend_file) and clobber:
        print("Clobbering existing backend file")
        os.remove(backend_file)

    if os.path.exists(backend_file):
        backend = emcee.backends.HDFBackend(backend_file)
        if backend.iteration > 0: # If the file is empty, start from scratch
            pos = None # backend will resume from previous position by default
            print("Resuming from existing backend file at iteration {}".format(backend.iteration))
        else:
            backend.reset(nwalkers, ndim) # Reset the backend
            print("Existing backend file found but is empty. Starting from scratch")
    else:
        backend = emcee.backends.HDFBackend(backend_file)
        backend.reset(nwalkers, ndim) # Reset the backend
        print("No existing backend file found. Starting from scratch")

    converged = False # Convergence flag

    # Set up the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(priors, data_vector, inv_cov, ClHook),
        vectorize=True,
        backend=backend
    )

    while not converged:
        clear_output(wait=True)
        # Sample 
        sampler.run_mcmc(pos, n_check, progress=True)

        # Check convergence
        try:
            tau = sampler.get_autocorr_time(tol=0)
            print("Current iteration: {}".format(sampler.iteration))
            print("Rounded autocorrelation times: {}".format((tau * 50).astype(int)))
        except emcee.autocorr.AutocorrError:
            print("Autocorrelation time could not be estimated. Continuing...")

        chain = sampler.get_chain()
        R_hat = az.rhat(chain)

        if np.all(tau * 50 < sampler.iteration) and R_hat < 1.1:
            print("Convergence achieved!")
            converged = True

        if sampler.iteration >= max_iter:
            print("Maximum number of iterations reached without convergence. Exiting...")
            break

    print("Sampling completed in {} iterations".format(sampler.iteration))