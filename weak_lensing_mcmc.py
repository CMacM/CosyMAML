import os

# set the environment variable to control the number of threads
# NEEDS TO BE DONE BEFORE CCL IS IMPORTED
original_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
os.environ['OMP_NUM_THREADS'] = '1'

import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
import emcee
import argparse
import cProfile
import pstats

from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.linalg import solve

from multiprocessing import Pool, cpu_count

from time import time

def Smail_dndz(z, z0, alpha):
    return z**2 * np.exp(-(z/z0)**alpha)

def convolve_photoz(sigma, zs, dndz_spec):

     # Convolve with photo-z
    sigma_z = sigma * (1 + zs)

    z_ph = np.linspace(0.05, 4.0, 300)

    # find probability of galaxy with true redshift z_s to be measured at redshift z_ph
    integrand1 = np.zeros([len(zs),len(z_ph)])
    p_zs_zph = np.zeros([len(zs),len(z_ph)])
    for j in range(len(zs)):
        p_zs_zph[j,:] =  (1. / (np.sqrt(2. * np.pi) * sigma_z[j])) * np.exp(-((z_ph - zs[j])**2) / (2. * sigma_z[j]**2))

    integrand1 = p_zs_zph * dndz_spec[:,None]   

    # integrate over z_s to get dN
    integral1 = simps(integrand1, zs, axis=0)
    dN = integral1
    
    dz_ph = simps(dN, z_ph)
    
    return z_ph, dN/dz_ph

def setup_redshifts(z0, alpha, n_bins):

    # Generate the redshift distribution
    z = np.linspace(0.2, 3.5, 300)

    dndz_s = Smail_dndz(z, z0, alpha)

    # Normalize the distribution
    area = simps(dndz_s, z)  # Integrate dndz_s over z to get the area under the curve
    pdf = dndz_s / area  # Normalize to make it a PDF

    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(pdf) * (z[1] - z[0])  # Approximate the integral to get the CDF

    # Interpolate the CDF to find the bin edges
    inverse_cdf = interp1d(cdf, z, fill_value="extrapolate")

    # Define the CDF values for the bin edges
    cdf_values = np.linspace(0, 1, n_bins+1)

    # Find the corresponding z values (bin edges) for these CDF values
    bin_edges = inverse_cdf(cdf_values)

    zs = []
    dndz_spec_bins = []
    dndz_ph_bins = []
    for i in range(n_bins):
        zs.append(np.linspace(bin_edges[i], bin_edges[i+1], len(z)))
        dndz_spec_bins.append(Smail_dndz(zs[i], z0, alpha)) # Store spec bins to produce slighty different photo-z bins later

        z_ph, dndz_ph = convolve_photoz(0.05, zs[i], dndz_spec_bins[i])
        dndz_ph_bins.append(dndz_ph) # fiducial photo-z bins

    return z_ph, dndz_ph_bins

def compute_spectra(cosmo, dndz_ph_bins, z_ph, ells):
    
        n_bins = len(dndz_ph_bins)
        indices = np.tril_indices(n_bins)
        zipped_inds = list(zip(*indices))
    
        c_ells = np.empty((len(zipped_inds), len(ells)))
        for i, arg in enumerate(zipped_inds):
            j, k = arg
            tracer1 = ccl.WeakLensingTracer(cosmo, dndz=(z_ph[j], dndz_ph_bins[j]))
            tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z_ph[k], dndz_ph_bins[k]))
            c_ells[i,:] = ccl.angular_cl(cosmo, tracer1, tracer2, ells, l_limber='auto')

        return c_ells

def compute_covariance(noise):
    return np.diag(noise**2)

# Define the likelihood function
def log_likelihood(theta, data, inv_cov, dndz_ph_bins, z_ph, ells):
    cosmo = ccl.Cosmology(
        Omega_c=theta[0],
        Omega_b=theta[1],
        h=theta[2],
        sigma8=theta[3],
        n_s=theta[4],
        matter_power_spectrum='halofit'
    )

    # Shift mean of photo-z bins based on theta
    # Can comment out if not using shifts
    shifted_z = [] 
    for i in range(len(dndz_ph_bins)):
        mean_z = np.average(z_ph, weights=dndz_ph_bins[i])
        shifted_z.append(z_ph + theta[5+i]*(1 + mean_z))
        # set any negative values to 0
        shifted_z[i][shifted_z[i] < 0] = 0

    model_vector = compute_spectra(cosmo, dndz_ph_bins, shifted_z, ells)
    model_vector = model_vector.flatten() # Flatten the array to make it a vector

    # Compute the log likelihood
    diff = data - model_vector
    log_like = -0.5 * np.dot(diff, np.dot(inv_cov, diff))

    return log_like, model_vector

def log_prior(theta, priors):
    for i, prior in enumerate(priors):
        if not prior[0] < theta[i] < prior[1]:
            return -np.inf
    return 0.0

def log_probability(theta, priors, data, inv_cov, dndz_ph_bins, z_ph, ells):
    lp = log_prior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf, None
    ll, model_vector = log_likelihood(theta, data, inv_cov, dndz_ph_bins, z_ph, ells)
    if not np.isfinite(ll):
        return -np.inf, None
    log_prob = lp + ll
    return log_prob, model_vector

def main(args):

    # Create the output directory
    mcmc_dir = args.mcmc_dir
    if not os.path.exists(mcmc_dir):
        os.makedirs(mcmc_dir)

    # Set up the redshift bins
    z0 = args.z0
    alpha = args.alpha
    n_bins = args.n_bins
    z_ph, dndz_ph_bins = setup_redshifts(z0, alpha, n_bins)

    print("Redshift bins set up")

    # Define the cosmology
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
        matter_power_spectrum='halofit'
    )

    ells = np.geomspace(2, 2000, 30)

    # Compute the spectra
    zph_list = [z_ph for i in range(n_bins)]
    c_ells = compute_spectra(cosmo, dndz_ph_bins, zph_list, ells)

    data_vector = c_ells.flatten() # Flatten the array to make it a vector

    # Add noise
    noise = 0.1 * data_vector
    data_vector += np.random.normal(0, noise)

    # generate covariance matrix
    cov = compute_covariance(noise)
    inv_cov = np.linalg.inv(cov)
    print("Data vector and covariance matrix set up")
    
    # Set the random seed for reproducibility
    np.random.seed(args.seed)

    # test the likelihood function
    theta = [0.27, 0.045, 0.67, 0.83, 0.96]

    # Comment out if not using shifts
    for i in range(n_bins):
        theta.append(0.0)

    # Initialize the walkers
    nwalkers = args.n_walkers
    ndim = len(theta)
    chain_len = args.chain_len

    # Define the priors
    priors = [
        (0.1, 0.9), # Omega_c
        (0.01, 0.1), # Omega_b
        (0.5, 0.9), # h
        (0.7, 1.0), # sigma8
        (0.8, 1.1) # n_s
    ]

    # Comment out if not using shifts
    delta_z = 0.002 # LSST Y1 mean uncertainty
    for i in range(n_bins):
        priors.append((-delta_z, delta_z)) # Shifts for each redshift bin

    # Define the initial positions
    pos = [
        theta[0] + 1e-2 * np.random.randn(nwalkers), # Omega_c
        theta[1] + 1e-3 * np.random.randn(nwalkers), # Omega_b
        theta[2] + 1e-2 * np.random.randn(nwalkers), # h
        theta[3] + 1e-2 * np.random.randn(nwalkers), # sigma8
        theta[4] + 1e-2 * np.random.randn(nwalkers), # n_s
    ]

    # Comment out if not using shifts
    for i in range(n_bins):
        pos += (theta[5+i] + 1e-4 * np.random.randn(nwalkers),)
    pos = np.array(pos).T

    converged = False # Convergence flag

    # Check for existing backend file
    backend_file = os.path.join(mcmc_dir, '{}walkers_chain_outputs.h5'.format(nwalkers))

    if os.path.exists(backend_file) and args.clobber:
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

    # Initialise a pool
    n_threads = min(cpu_count(), args.n_threads)
    with Pool(n_threads) as pool:

        # set up the blob type
        btype = [('model_vector', float, len(data_vector))]

        # Set up the sampler
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=(priors, data_vector, inv_cov, dndz_ph_bins, z_ph, ells),
            pool=pool,
            backend=backend,
            blobs_dtype=btype
        )
        
        while not converged:
            # Sample 
            start = time()
            sampler.run_mcmc(pos, chain_len, progress=True)
            end = time()

            # Check convergence
            try:
                tau = sampler.get_autocorr_time(tol=0)
                converged = np.all(tau * 50 < sampler.iteration)
                print("Current iteration: {}".format(sampler.iteration))
                print("Rounded autocorrelation times: {}".format((tau * 50).astype(int)))
            except emcee.autocorr.AutocorrError:
                print("Autocorrelation time could not be estimated. Continuing...")

            if sampler.iteration >= args.max_iter:
                print("Maximum number of iterations reached without convergence. Exiting...")
                break

            # Estimate completion time
            time_per_iter = (end - start) / chain_len
            time_left = time_per_iter * (np.max(tau) * 50 - sampler.iteration)
            print("Estimated time left: {}mins".format(time_left/60))

            if args.profile:
                # Check data has saved correctly
                samples = backend.get_chain()
                log_prob = backend.get_log_prob()
                blobs = backend.get_blobs()
                model_vectors = blobs['model_vector']
                assert samples.shape == (sampler.iteration, nwalkers, ndim)
                assert log_prob.shape == (sampler.iteration, nwalkers)
                assert model_vectors.shape == (sampler.iteration, nwalkers, len(data_vector))

                break # Break after one iteration if profiling

    print("Sampling completed in {} iterations".format(sampler.iteration))

if __name__ == '__main__':
    # Add command line arguments for the script
    parser = argparse.ArgumentParser(description='Run MCMC to estimate cosmological parameters')
    parser.add_argument('--mcmc_dir', type=str, default='mcmc', help='Directory to save MCMC output')
    parser.add_argument('--z0', type=float, default=0.11, help='Smail parameter z0')
    parser.add_argument('--alpha', type=float, default=0.71, help='Smail parameter alpha')
    parser.add_argument('--n_bins', type=int, default=5, help='Number of redshift bins')
    parser.add_argument('--n_walkers', type=int, default=48, help='Number of walkers')
    parser.add_argument('--chain_len', type=int, default=1000, help='Length of chain to run between convergence checks')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_threads', type=int, default=cpu_count(), help='Number of threads to use')
    parser.add_argument('--profile', action='store_true', help='Run the profiler')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files')
    args = parser.parse_args() # Parse the arguments

    if args.profile: # If profiling is enabled
        pr = cProfile.Profile()
        pr.enable()
        main(args)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('cumulative')
        ps.print_stats(20)
    else: # Otherwise run the main function
        main(args)