import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import emcee
import torch
import os

import h5py

import src.training as training
import src.models as models

from time import time
from emcee.autocorr import integrated_time, AutocorrError
from tqdm import tqdm

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
        tau_factor=50,
        clobber=False,
        progress=False
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
        sampler.run_mcmc(pos, n_check, progress=progress)

        # Check convergence
        try:
            tau = sampler.get_autocorr_time(tol=0)
            converged = np.all(tau * tau_factor < sampler.iteration)
            if progress:
                print("Current iteration: {}".format(sampler.iteration))
                print("Autocorrelation time: {}".format(tau * tau_factor))
        except emcee.autocorr.AutocorrError:
            print("Autocorrelation time could not be estimated. Continuing...")

        if sampler.iteration >= max_iter:
            print("Maximum number of iterations reached without convergence. Exiting...")
            break

    print("Sampling completed in {} iterations".format(sampler.iteration))

def run_batched_mh(
        ClHook,
        backend_file,
        ndim,
        priors,
        initial_pos,
        data_vector,
        inv_cov,
        batch_size=128,
        max_iter=50000,
        n_check=1000,
        tau_factor=50,
        proposal_scale=0.01,
        clobber=True,
        progress=False
    ):

    if clobber and os.path.exists(backend_file):
        os.remove(backend_file)

    current = np.array(initial_pos)
    chain = np.zeros((max_iter, ndim), dtype=np.float32)
    converged = False
    tau_factor = 50

    # Evaluate initial log prob
    current_log_prob = log_probability(current[None, :], priors, data_vector, inv_cov, ClHook)[0]

    # Initialise tqdm progress bar
    if progress:
        pbar = tqdm(total=max_iter, desc="MCMC Sampling", unit="step")
        pbar.update(0)

    with h5py.File(backend_file, "w") as f:
        dset = f.create_dataset("chain", shape=(max_iter, ndim), dtype="f4")
        dlogp = f.create_dataset("log_prob", shape=(max_iter,), dtype="f4")

        for step in range(max_iter):
            proposals = current + proposal_scale * np.random.randn(batch_size, ndim)
            log_probs = log_probability(proposals, priors, data_vector, inv_cov, ClHook)

            # Softmax-based sampling to pick a new state
            log_ratios = log_probs - current_log_prob
            weights = np.exp(log_ratios - np.max(log_ratios))  # stability
            weights /= np.sum(weights)

            idx = np.random.choice(batch_size, p=weights)
            current = proposals[idx]
            current_log_prob = log_probs[idx]

            # Save sample
            chain[step] = current
            dset[step] = current
            dlogp[step] = current_log_prob

            # Update progress bar
            if progress:
                pbar.update(1)
                pbar.set_postfix({"log_prob": current_log_prob})
                pbar.refresh()

            # Convergence check
            if step >= n_check and step % n_check == 0:
                try:
                    tau = integrated_time(chain[:step], tol=0)
                    converged = np.all(tau * tau_factor < step)
                    if converged:
                        print(f"Converged at step {step} with tau = {tau}")
                except AutocorrError:
                    print("Autocorrelation time could not be estimated. Continuing...")

    if progress:
        pbar.close()