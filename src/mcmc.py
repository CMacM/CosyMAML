import os
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
        self.ScalerX = ScalerX  # Expecting Torch-compatible scaler
        self.ScalerY = ScalerY  # Expecting Torch-compatible scaler
        self.weights = weights

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def __call__(self, theta_tensor):
        if isinstance(theta_tensor, np.ndarray):
            theta_tensor = torch.tensor(theta_tensor, dtype=torch.float32, device=self.device)

        # theta_tensor shape: (batch_size, n_params), already a torch tensor on device
        self.model.eval()

        # Apply scaling (Torch version expected)
        theta_scaled = self.ScalerX.transform(theta_tensor)

        with torch.no_grad():
            model_vectors = self.model(theta_scaled, params=self.weights)  # shape: (batch_size, output_dim)
            model_vectors = torch.exp(self.ScalerY.inverse_transform(model_vectors))  # Inverse transform in log-space

        return model_vectors  # Still a torch tensor (batch_size, output_dim)

def log_likelihood(theta_batch, data_vector, inv_cov, ClHook):
    model_vectors = ClHook(theta_batch)  # Torch tensor output

    diffs = data_vector.unsqueeze(0) - model_vectors
    diffs = diffs @ inv_cov
    log_likes = -0.5 * torch.sum(diffs * (data_vector.unsqueeze(0) - model_vectors), dim=1)

    return log_likes

def log_prior(theta_batch, priors, device):

    lower_bounds = torch.tensor([p[0] for p in priors], device=device)
    upper_bounds = torch.tensor([p[1] for p in priors], device=device)

    in_bounds = (theta_batch > lower_bounds) & (theta_batch < upper_bounds)
    valid_mask = torch.all(in_bounds, dim=1)

    lp = torch.full((theta_batch.shape[0],), -float('inf'), device=device)
    lp[valid_mask] = 0.0

    return lp

def log_probability(theta_batch, priors, data_vector, inv_cov, ClHook):
    device = ClHook.device

    # Convert theta_batch if needed
    if isinstance(theta_batch, np.ndarray):
        theta_batch = torch.tensor(theta_batch, dtype=torch.float32, device=device)

    lp = log_prior(theta_batch, priors, device)

    if not torch.any(torch.isfinite(lp)):
        return lp.cpu().numpy()  # For emcee compatibility

    valid_mask = torch.isfinite(lp)
    ll = torch.full_like(lp, -float('inf'))

    ll[valid_mask] = log_likelihood(theta_batch[valid_mask], data_vector, inv_cov, ClHook)

    log_prob = lp + ll

    return log_prob.cpu().numpy()  # Return as numpy for emcee compatibility

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

# def run_batched_mh(
#         ClHook,
#         backend_file,
#         ndim,
#         priors,
#         initial_pos,
#         data_vector,
#         inv_cov,
#         batch_size=128,
#         max_iter=500000,
#         n_check=5000,
#         tau_factor=50,
#         proposal_scale=0.001,
#         window_size=5000,
#         clobber=True,
#         progress=False
#     ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if clobber and os.path.exists(backend_file):
#         os.remove(backend_file)

#     # Convert data_vector & inv_cov to torch tensors (once)
#     if not isinstance(data_vector, torch.Tensor):
#         data_vector = torch.tensor(data_vector, dtype=torch.float32, device=device)
#     if not isinstance(inv_cov, torch.Tensor):
#         inv_cov = torch.tensor(inv_cov, dtype=torch.float32, device=device)

#     # Initialize current position
#     current = torch.tensor(initial_pos, dtype=torch.float32, device=device).unsqueeze(0)  # (1, ndim)
#     current_log_prob = log_probability(current, priors, data_vector, inv_cov, ClHook)[0]

#     # Preallocate chains (CPU side for HDF5 writing)
#     chain = np.zeros((max_iter, ndim), dtype=np.float32)
#     log_probs_chain = np.zeros((max_iter,), dtype=np.float32)

#     accepted = 0

#     # Progress bar
#     if progress:
#         pbar = tqdm(total=max_iter, desc="MCMC Sampling", unit="step")

#     with h5py.File(backend_file, "w") as f:
#         dset = f.create_dataset("chain", shape=(max_iter, ndim), dtype="f4")
#         dlogp = f.create_dataset("log_prob", shape=(max_iter,), dtype="f4")

#         for step in range(max_iter):
#             # Batched proposals on GPU
#             proposals = current + proposal_scale * torch.randn((batch_size, ndim), device=device)

#             # Compute log probabilities for proposals
#             log_probs_proposals = torch.tensor(
#                 log_probability(proposals, priors, data_vector, inv_cov, ClHook),
#                 dtype=torch.float32,
#                 device=device
#             )

#             # Compute log acceptance ratios
#             log_acceptance_ratio = log_probs_proposals - current_log_prob  # (batch_size,)

#             # Generate uniform randoms in log-space
#             log_uniforms = torch.log(torch.rand(batch_size, device=device))

#             # Determine acceptances
#             accept_mask = log_acceptance_ratio >= log_uniforms

#             if accept_mask.any():
#                 # Randomly select one of the accepted proposals
#                 accepted_indices = torch.nonzero(accept_mask, as_tuple=False).squeeze(1)
#                 chosen_idx = accepted_indices[torch.randint(len(accepted_indices), (1,)).item()]

#                 # Accept this proposal
#                 current = proposals[chosen_idx:chosen_idx+1]  # keep shape (1, ndim)
#                 current_log_prob = log_probs_proposals[chosen_idx]
#                 accepted += 1
#                 accepted_this_step = True
#             else:
#                 accepted_this_step = False  # Remain at current

#             # Save current state
#             current_np = current.squeeze(0).cpu().numpy()
#             chain[step] = current_np
#             log_probs_chain[step] = current_log_prob.item()
#             dset[step] = current_np
#             dlogp[step] = current_log_prob.item()

#             # Update progress bar
#             if progress:
#                 pbar.update(1)
#                 if step % (n_check // 10) == 0:
#                     acc_rate_so_far = accepted / (step + 1)
#                     pbar.set_postfix({"log_prob": current_log_prob.item(), "acc_rate": acc_rate_so_far})

#             # Convergence check every n_check steps
#             if step >= n_check and step % n_check == 0:
#                 try:
#                     tau = integrated_time(chain[:step], tol=0, has_walkers=False)
#                     acc_rate = accepted / (step + 1)
#                     print(f"Step {step}: Acceptance Rate = {acc_rate:.3f}, Tau = {tau * tau_factor}")

#                     if np.all(tau * tau_factor < step):
#                         print(f"Converged at step {step} with tau = {tau}")
#                         break
#                 except AutocorrError:
#                     print(f"Step {step}: AutocorrError — continuing sampling...")

#     # Final acceptance rate
#     final_acceptance_rate = accepted / max_iter
#     print(f"Final Acceptance Rate: {final_acceptance_rate:.3f}")

#     if progress:
#         pbar.close()