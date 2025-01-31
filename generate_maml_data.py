import os

# set the environment variable to control the number of threads
# NEEDS TO BE DONE BEFORE CCL IS IMPORTED
original_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
os.environ['OMP_NUM_THREADS'] = '1'

import pyccl as ccl
import numpy as np
import h5py as h5

import scipy.stats.qmc as qmc
from scipy.interpolate import interp1d
from scipy.integrate import simps

from multiprocessing import Pool, cpu_count
from parallelbar import progress_starmap

import argparse

# Begin by defining functions needed to generat the data

# define various functions to produce redshift distributions
def Smail_dndz(z, z0=0.13, alpha=0.78):
    '''
    Smail type redshift distribution.
    Args:
    z: redshift (array-like)
    z0: characteristic redshift, peak of the distribution (float)
    alpha: power law index, slope of the distribution (float)
    '''
    return z**2 * np.exp(-(z/z0)**alpha)

def Gaussian_dndz(z, z0=0.7, sigma=0.4):
    '''
    Gaussian redshift distribution.
    Args:
    z: redshift (array-like)
    z0: mean redshift (float)
    sigma: standard deviation of the distribution (float)
    '''
    return np.exp(-0.5 * ((z - z0) / sigma)**2)

def add_noise(z, dndz, noise=0.04):
    '''
    Add noise to the redshift distribution.
    Args:
    z: redshift (array-like)
    dndz: redshift distribution (array-like)
    noise: standard deviation of the noise (float)
    '''
    dndz_noisy = dndz + noise / (1 + z) * np.random.randn(len(dndz))
    dndz_noisy = np.clip(dndz_noisy, 0, None) # negative values are unphysical
    return dndz_noisy

def bin_dndz(n_bins, z, dndz_func, **kwargs):
    '''
    Bin the redshift distribution.
    Args:
    n_bins: number of bins (int)
    z: redshift (array-like)
    dndz_func: redshift distribution function (callable)
    kwargs: additional arguments for dndz_func
    '''

    # Compute the redshift distribution
    dndz = dndz_func(z, **kwargs)

    # Normalize the distribution
    area = simps(dndz, z)  # Integrate dndz_s over z to get the area under the curve
    pdf = dndz / area  # Normalize to make it a PDF

    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(pdf) * (z[1] - z[0])  # Approximate the integral to get the CDF

    # Interpolate the CDF to find the bin edges
    inverse_cdf = interp1d(cdf, z, fill_value="extrapolate")

    # Define the CDF values for the bin edges
    cdf_values = np.linspace(0, 1, n_bins+1)

    # Find the corresponding z values (bin edges) for these CDF values
    bin_edges = inverse_cdf(cdf_values)

    z_bin = np.zeros((n_bins, len(z)))
    dndz_bin = np.zeros((n_bins, len(z)))
    for i in range(n_bins):
        z_bin[i] = np.linspace(bin_edges[i], bin_edges[i+1], len(z))
        dndz_bin[i] = dndz_func(z_bin[i], **kwargs)

    return z_bin, dndz_bin

def shift_mean(z, dndz, delta_z=0.02):
    '''
    Shift the mean redshift of the distribution.
    Args:
    z: redshift (array-like)
    dndz: redshift distribution (array-like)
    delta_z: shift in the mean redshift (float)
    '''
    mean_z = np.average(z, weights=dndz)
    delta_z *= (1 + mean_z)
    z_shift = z + delta_z
    # Ensure the redshift distribution is non-negative.
    # Clip to a very small value instead of zero 
    # to avoid division by zero
    for i in range(len(z_shift)):
        if z_shift[i] <= 0:
            z_shift[i] = 1e-10*i
    return z_shift

def convolve_photoz(sigma, zs, dndz_spec, return_2d=False):
    '''
    Convolve the true redshift distribution with the photo-z error.
    Args:
    sigma: photo-z error (float)
    zs: true redshift (array-like)
    dndz_spec: true redshift distribution (array-like)
    return_2d: return the 2D probability distribution (bool)
    '''

     # Convolve with photo-z
    sigma_z = sigma * (1 + zs)

    z_ph = np.linspace(0.0, 4.0, 300)

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

    if return_2d:
        return z_ph, dN/dz_ph, p_zs_zph
    
    return z_ph, dN/dz_ph

# NEXT STEP IS TO GENERATE THE SPECTRA
global compute_spectra
def compute_spectra(theta, dndz_ph_bins, z_ph, ells):
    '''
    Compute the angular power spectra for the given redshift distributions.
    Args:
    theta: cosmological parameters (array-like)
    dndz_ph_bins: redshift distributions (list of array-like)
    z_ph: redshift (array-like)
    ells: multipoles (array-like)
    shift_mean: shift in the mean redshift of the distribution (float)'''

    cosmo = ccl.Cosmology(
        Omega_c=theta[0],
        Omega_b=theta[1],
        h=theta[2],
        sigma8=theta[3],
        n_s=theta[4],
        matter_power_spectrum='halofit'
    )

    n_bins = len(dndz_ph_bins)
    inds = list(zip(*np.tril_indices(n_bins)))

    z_ph_shifted = np.empty((n_bins, len(z_ph)))
    for i in range(n_bins):
        z_ph_shifted[i] = shift_mean(
            z_ph, 
            dndz_ph_bins[i],
            delta_z=theta[5+i]
        )

    c_ells = np.empty((len(inds), len(ells)))
    for i, arg in enumerate(inds):
        j, k = arg
        tracer1 = ccl.WeakLensingTracer(
            cosmo,
            dndz=(z_ph_shifted[j], dndz_ph_bins[j])
        )
        tracer2 = ccl.WeakLensingTracer(
            cosmo,
            dndz=(z_ph_shifted[k], dndz_ph_bins[k])
        )
        c_ells[i,:] = ccl.angular_cl(cosmo, tracer1, tracer2, ells)
    
    return c_ells.flatten()

def wrap_errors(theta, dndz_ph_bins, z_ph, ells):
    '''
    Wrapper function to catch errors in parallel computation.
    Args:
    func: function to be called (callable)
    args: arguments for the function (tuple)
    '''
    try:
        return compute_spectra(theta, dndz_ph_bins, z_ph, ells)
    except Exception as e:
        # Dump data and re-raise the exception
        print('Caught exception in worker thread:')
        print('Dumping data for debugging')
        np.savez(
            os.path.join(
                'data', 'debug_worker{}.npz'.format(os.getpid())
                ),
            theta=theta,
            dndz_ph_bins=dndz_ph_bins,
            z_ph=z_ph,
            ells=ells
        )
        raise e

def main(args):
    # Set the random seed
    np.random.seed(args.seed)

    # Define the redshift range
    z = np.linspace(args.z_min, args.z_max, 300)

    ell_bins = 50
    inds = list(zip(*np.tril_indices(args.n_bins)))

    # Loop through the number of tasks
    y_train = np.empty((
        args.n_tasks,
        args.n_samples,
        ell_bins * len(inds)
    ))
    X_train = np.empty((
        args.n_tasks,
        args.n_samples,
        10
    ))
    dndz_save = np.empty((
        args.n_tasks,
        args.n_bins,
        len(z)
    ))
    z_save = np.empty((
        args.n_tasks,
        args.n_bins,
        len(z)
    ))
    dndz_params = []
    model_type = []

    # Construct latin hypercube of redshift parameters
    hyperframe = qmc.LatinHypercube(d=2)

    z0_smail = np.array([0.1, 0.2])
    alpha_smail = np.array([0.6, 1.0])

    z0_gaussian = np.array([0.2, 1.5])
    sigma_gaussian = np.array([0.2, 0.6])

    hyperunits_smail = hyperframe.random(args.n_tasks//2)
    hyperunits_gaussian = hyperframe.random(args.n_tasks//2)

    l_bounds_smail = np.array([z0_smail[0], alpha_smail[0]])
    u_bounds_smail = np.array([z0_smail[1], alpha_smail[1]])
    hypercube_smail = qmc.scale(hyperunits_smail, l_bounds_smail, u_bounds_smail)

    l_bounds_gaussian = np.array([z0_gaussian[0], sigma_gaussian[0]])
    u_bounds_gaussian = np.array([z0_gaussian[1], sigma_gaussian[1]])
    hypercube_gaussian = qmc.scale(hyperunits_gaussian, l_bounds_gaussian, u_bounds_gaussian)

    smail_i = 0
    gaussian_i = 0
    for i in range(args.n_tasks):
        
        if i % 2 == 0:
            dndz_func = Smail_dndz
            z0 = hypercube_smail[smail_i, 0]
            alpha = hypercube_smail[smail_i, 1]
            kwargs = {'z0': z0, 'alpha': alpha}
            smail_i += 1
        else:
            dndz_func = Gaussian_dndz
            z0 = hypercube_gaussian[gaussian_i, 0]
            sigma = hypercube_gaussian[gaussian_i, 1]
            kwargs = {'z0': z0, 'sigma': sigma}
            gaussian_i += 1

        # Bin the redshift distribution
        z_bin, dndz_bin = bin_dndz(args.n_bins, z, dndz_func, **kwargs)

        # Convolve with photo-z and add noise to the redshift distribution
        dndz_bin_ph = np.zeros((args.n_bins, len(z)))
        noise_std = np.random.uniform(0, args.noise_lim)
        for j in range(args.n_bins):
            z_ph, dndz_bin_ph[j] = convolve_photoz(
                sigma=args.sigma_pz, 
                zs=z_bin[j], 
                dndz_spec=dndz_bin[j]
            )

            dndz_bin_ph[j] = add_noise(
                z_ph,
                dndz_bin_ph[j],
                noise_std
            )

        # Now we compute samples in parallel
        # Start by constructing the latin hypercube
        # Construct parameter hypercube from DES Y3 priors
        Omega_m = np.array([0.05, 0.95])
        Omega_b = np.array([0.025, 0.075])
        Omega_c = Omega_m - Omega_b

        h = np.array([0.55, 0.91])
        n_s = np.array([0.87, 1.07])
        sigma8 = np.array([0.6, 0.9])

        delta_z = np.array([-0.02, 0.02])# times 5 bins

        hyperframe = qmc.LatinHypercube(d=10)
        hyperunits = hyperframe.random(args.n_samples)

        # Rescale the hypercube for provided param ranges
        l_bounds = np.array([
            Omega_c[0], Omega_b[0], h[0], sigma8[0], n_s[0],
            delta_z[0], delta_z[0], delta_z[0], delta_z[0], delta_z[0]
        ])
        u_bounds = np.array([
            Omega_c[1], Omega_b[1], h[1], sigma8[1], n_s[1],
            delta_z[1], delta_z[1], delta_z[1], delta_z[1], delta_z[1]
        ])
        hypercube = qmc.scale(hyperunits, l_bounds, u_bounds)

        # Construct arglist for parallel computation of
        # the angular power spectra
        ells = np.geomspace(2, args.ell_max, ell_bins)
        arglist = [
            (hypercube[i], dndz_bin_ph, z_ph, ells)
            for i in range(args.n_samples)
        ]

        # Compute the angular power spectra in parallel
        with Pool(args.n_threads) as pool:
            c_ells = progress_starmap(
                wrap_errors,
                arglist,
                n_cpu=args.n_threads
            )

        # Collect the data into lists for saving
        y_train[i] = np.array(c_ells)
        X_train[i] = hypercube
        dndz_save[i] = dndz_bin_ph
        z_save[i] = z_ph
        if i % 2 == 0:
            dndz_params.append([z0, alpha])
        else:
            dndz_params.append([z0, sigma])

        model_type.append(dndz_func.__name__)

    # Save the data as h5 file
    filename = '{}tasks_{}samples_seed{}.h5'.format(args.n_tasks, args.n_samples, args.seed)
    with h5.File(os.path.join(args.output, filename), 'w') as f:
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('dndz', data=dndz_save)
        f.create_dataset('z', data=z_save)
        f.create_dataset('dndz_params', data=dndz_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for photo-z estimation')
    parser.add_argument('--n_bins', type=int, default=5, help='Number of redshift bins')
    parser.add_argument('--n_tasks', type=int, default=50, help='Number of tasks')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--gaussian_prob', type=float, default=0.5, help='Probability of Gaussian distribution')
    parser.add_argument('--noise_lim', type=float, default=0.1, help='Standard deviation of the noise')
    parser.add_argument('--shift', type=float, default=0.01, help='Max shift in the mean redshift')
    parser.add_argument('--sigma_pz', type=float, default=0.04, help='Photo-z error')
    parser.add_argument('--z_min', type=float, default=0.05, help='Minimum redshift')
    parser.add_argument('--z_max', type=float, default=3.5, help='Maximum redshift')
    parser.add_argument('--n_threads', type=int, default=cpu_count(), help='Number of threads')
    parser.add_argument('--ell_max', type=int, default=5000, help='Maximum multipole')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    args = parser.parse_args()

    main(args)