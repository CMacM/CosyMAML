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
import psutil
import sacc
import src.simulate as sim

from tjpcov.covariance_calculator import CovarianceCalculator
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.linalg import solve

from multiprocessing import Pool, cpu_count

from time import time

def log_likelihood(theta, data_vector, inv_cov, dndz_ph_bins, z_ph, ells, C_ell_func):

    # Call C_ell function
    c_ells = C_ell_func(theta, dndz_ph_bins, z_ph, ells)

    # Compute the log-likelihood
    diffs = data_vector - c_ells
    log_likes = -0.5 * np.dot(diffs, np.dot(inv_cov, diffs))

    return log_likes # Return a vector of log-likelihoods

def log_prior(theta, priors):
    for i, prior in enumerate(priors):
        if not prior[0] < theta[i] < prior[1]:
            return -np.inf
    return 0.0

def log_probability(theta, priors, data_vector, inv_cov, dndz_ph_bins, z_ph, ells, C_ell_func):
    lp = log_prior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood(theta, data_vector, inv_cov, dndz_ph_bins, z_ph, ells, C_ell_func)
    return lp + ll

def main(args):

    if args.for_cluster:
        C_ell_func = sim.compute_spectra_cluster
        sacc_type = 'cl_00'
        print('Using clustering spectra')
    else:
        C_ell_func = sim.compute_spectra_cosmicshear
        sacc_type = 'cl_ee'
        print('Using cosmic shear spectra')

    # Create the output directory
    mcmc_dir = args.mcmc_dir
    if not os.path.exists(mcmc_dir):
        os.makedirs(mcmc_dir)

    # Set up the redshift bins
    n_bins = args.n_bins
    kwargs = {'z0': args.z0,'alpha': args.alpha,}
    
    # Set up the redshift distributions and bin
    z = np.linspace(0.05, 3.5, 300)
    z_bin, dndz_bin = sim.bin_dndz(args.n_bins, z, sim.Smail_dndz, **kwargs)

    # Convolve photo-z errors
    dndz_bin_ph = np.zeros((args.n_bins, len(z)))
    for j in range(args.n_bins):
        z_ph, dndz_bin_ph[j] = sim.convolve_photoz(
            sigma=args.sigma_pz, 
            zs=z_bin[j], 
            dndz_spec=dndz_bin[j]
        )
    print("Redshift bins set up")

    # Fiducial cosmological parameters (DES Y3)
    theta = [0.27, 0.045, 0.67, 0.83, 0.96]

    cosmo = ccl.Cosmology(
        Omega_c=theta[0],
        Omega_b=theta[1],
        h=theta[2],
        sigma8=theta[3],
        n_s=theta[4],
        matter_power_spectrum='halofit'
    )

    for i in range(n_bins):
        theta.append(0.0)

    # Construct ell vector
    ells = np.geomspace(2, 5000, 50)

    # Load fiducial spectra and cov or generate if needed
    sacc_path = os.path.join(mcmc_dir, '{}_fiducial_sacc.fits'.format(sacc_type))
    if (not args.clobber_sacc and os.path.exists(sacc_path)):
        print('Existing fiducial data found. Loading...')
        # Load in fiducial data
        S = sacc.Sacc.load_fits(sacc_path)
        print(S.mean.size)

        # Define tracer combs
        tracer_combs = S.get_tracer_combinations()

        # Extract C_ell and covariance blocks
        c_ells = []
        for comb in tracer_combs:
            _, cell = S.get_ell_cl(
                data_type=sacc_type,
                tracer1=comb[0],
                tracer2=comb[1],
                return_cov=False
            )
            c_ells.append(cell)

        # get covariance matrix
        cov = S.covariance.covmat
        # prepare data for MCMC
        data_vector = np.concatenate(c_ells)
    else:
        # Compute the spectra
        c_ells = C_ell_func(theta, dndz_bin_ph, z_ph, ells)

        # Store C_ells in a Sacc file
        s = sacc.Sacc()
        for i in range(n_bins):
            s.add_tracer('NZ', 'src{}'.format(i), z=z_ph, nz=dndz_bin_ph[i])

        # Set up the indices for the C_ell matrix
        if args.for_cluster:
            indices = []
            for i in range(n_bins):
                for j in range(n_bins):
                    if i == j:
                        indices.append((i, j))
        else:
            indices = np.tril_indices(n_bins)
            indices = list(zip(*indices))

        # Reshape back to 2D to store in Sacc file
        c_ells = c_ells.reshape((len(indices), len(ells)))
        print(c_ells.shape)

        for i, arg in enumerate(indices):
            j, k = arg
            s.add_ell_cl(sacc_type, 'src{}'.format(j), 'src{}'.format(k), ells, c_ells[i])

        data_vector = c_ells.flatten() # Flatten the array to make it a vector

        # generate covariance matrix with tjpcov
        cov = sim.compute_covariance(cosmo, s, n_bins)

        # Add covariance to Sacc file
        s.add_covariance(cov)
        s.save_fits(sacc_path, overwrite=True)

    # Invert the covariance matrix
    inv_cov = np.linalg.inv(cov)
    print("Data vector and covariance matrix set up")
    
    # Set the random seed for reproducibility
    np.random.seed(args.seed)

    # Initialize the walkers
    nwalkers = args.n_walkers
    ndim = len(theta)

    if nwalkers*2 < ndim:
        raise ValueError("Number of walkers must be greater than half the number of parameters")

    # Define the priors
    priors = [
        (0.17, 0.4), # Omega_c
        (0.03, 0.07), # Omega_b
        (0.4, 1.1), # h
        (0.65, 1.0), # sigma8
        (0.8, 1.1) # n_s
    ]

    # Comment out if not using shifts
    delta_z = 0.004 # LSST Y1 mean uncertainty
    for i in range(n_bins):
        priors.append((-delta_z, delta_z)) # Shifts for each redshift bin

    # 10% spread around the known good point
    spreads = 0.1 * np.array(theta)
    for i in range(n_bins):
        spreads[n_bins+i] += 4e-4
    pos = [theta + spreads * np.random.randn(ndim) for _ in range(nwalkers)]

        # Define the initial positions
    # pos = [
    #     theta[0] + 1e-2 * np.random.randn(nwalkers), # Omega_c
    #     theta[1] + 1e-3 * np.random.randn(nwalkers), # Omega_b
    #     theta[2] + 1e-2 * np.random.randn(nwalkers), # h
    #     theta[3] + 1e-2 * np.random.randn(nwalkers), # sigma8
    #     theta[4] + 1e-2 * np.random.randn(nwalkers), # n_s
    # ]
    pos = np.array(pos)

    converged = False # Convergence flag

    # Construct path for backend file
    backend_file = ('{}_{}walkers_{}tau'.format(
        sacc_type, nwalkers, args.tau_factor)
        +args.backend_file
    )
    backend_file = os.path.join(mcmc_dir, backend_file)

    # Check for existing backend file
    if os.path.exists(backend_file) and args.clobber_chain:
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

    chain_len = args.chain_len
    max_iter = args.max_iter

    # Initialise a pool
    n_threads = min(args.n_walkers, args.n_threads)
    profile = args.profile
    with Pool(n_threads) as pool:

        # set up the blob type
        btype = [('model_vector', float, len(data_vector))]

        # Set up the sampler
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=(priors, data_vector, inv_cov, dndz_bin_ph, z_ph, ells, C_ell_func),
            pool=pool,
            backend=backend,
            blobs_dtype=btype,
        )
        
        start = time()
        while not converged:
            # Sample 
            sampler.run_mcmc(pos, chain_len, progress=True)

            # Check convergence
            try:
                tau = sampler.get_autocorr_time(tol=0)
                converged = np.all(tau * args.tau_factor < sampler.iteration)
                print("Current iteration: {}".format(sampler.iteration))
                print("Rounded autocorrelation times: {}".format((tau * args.tau_factor).astype(int)))
            except emcee.autocorr.AutocorrError:
                print("Autocorrelation time could not be estimated. Continuing...")

            if sampler.iteration >= max_iter:
                print("Maximum number of iterations reached without convergence. Exiting...")
                break

            if profile:
                # Check data has saved correctly
                samples = backend.get_chain()
                log_prob = backend.get_log_prob()
                blobs = backend.get_blobs()
                model_vectors = blobs['model_vector']
                assert samples.shape == (sampler.iteration, nwalkers, ndim)
                assert log_prob.shape == (sampler.iteration, nwalkers)
                assert model_vectors.shape == (sampler.iteration, nwalkers, len(data_vector))

                break # Break after one iteration if profiling
    
    end = time()
    print("Sampling completed in {} iterations".format(sampler.iteration))
    print("Writing sampling metadata to file...")

    # Calculate cpu hours
    cpu_hours = (end - start) * n_threads / 3600
    print(f"CPU count: {psutil.cpu_count(logical=False)}")
    print(f"CPU usage per core: {psutil.cpu_percent(percpu=True)}")

    # Save the metadata in a text file
    with open(os.path.join(mcmc_dir, '{}walkers_metadata.txt'.format(nwalkers)), 'w') as f:
        f.write("Sampling completed in {} iterations\n".format(sampler.iteration))
        f.write("Total time taken: {} seconds\n".format(end - start))
        f.write("Number of threads used: {}\n".format(n_threads))
        f.write("Total CPU hours: {}\n".format(cpu_hours))

if __name__ == '__main__':
    # Add command line arguments for the script
    parser = argparse.ArgumentParser(description='Run MCMC to estimate cosmological parameters')
    parser.add_argument('--mcmc_dir', type=str, default='mcmc_chains', help='Directory to save MCMC output')
    parser.add_argument('--z0', type=float, default=0.13, help='Smail parameter z0')
    parser.add_argument('--alpha', type=float, default=0.78, help='Smail parameter alpha')
    parser.add_argument('--sigma_pz', type=float, default=0.04, help='Photo-z error')
    parser.add_argument('--n_bins', type=int, default=5, help='Number of redshift bins')
    parser.add_argument('--n_walkers', type=int, default=76, help='Number of walkers')
    parser.add_argument('--chain_len', type=int, default=1000, help='Length of chain to run between convergence checks')
    parser.add_argument('--max_iter', type=int, default=100000, help='Maximum number of iterations')
    parser.add_argument('--tau_factor', type=int, default=50, help='Factor to multiply by the autocorrelation time to determine convergence')
    parser.add_argument('--seed', type=int, default=14, help='Random seed')
    parser.add_argument('--n_threads', type=int, default=cpu_count(), help='Number of threads to use')
    parser.add_argument('--profile', action='store_true', help='Run the profiler')
    parser.add_argument('--clobber_chain', action='store_true', help='Overwrite existing MCMC file')
    parser.add_argument('--clobber_sacc', action='store_true', help='Overwrite existing Sacc file')
    parser.add_argument('--backend_file', type=str, default='CCL_chain.h5')
    parser.add_argument('--for_cluster', action='store_true', help='Run MCMC for clustering spectra')
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