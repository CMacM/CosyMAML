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
import src.simulate as sim

def main(args):
    # Set the random seed
    np.random.seed(args.seed)

    # Determine whether to generate clustering spectra or cosmic shear spectra
    if args.for_cluster:
        func = sim.compute_spectra_cluster
        tag = 'CU'
        print('Computing clustering spectra')
    else:
        func = sim.compute_spectra_cosmicshear
        tag = 'CS'
        print('Computing cosmic shear spectra')

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

    print('Constructing N(z) tasks...')

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
            dndz_func = sim.Smail_dndz
            z0 = hypercube_smail[smail_i, 0]
            alpha = hypercube_smail[smail_i, 1]
            kwargs = {'z0': z0, 'alpha': alpha}
            smail_i += 1
        else:
            dndz_func = sim.Gaussian_dndz
            z0 = hypercube_gaussian[gaussian_i, 0]
            sigma = hypercube_gaussian[gaussian_i, 1]
            kwargs = {'z0': z0, 'sigma': sigma}
            gaussian_i += 1

        # Bin the redshift distribution
        z_bin, dndz_bin = sim.bin_dndz(args.n_bins, z, dndz_func, **kwargs)

        # Convolve with photo-z and add noise to the redshift distribution
        dndz_bin_ph = np.zeros((args.n_bins, len(z)))
        noise_std = np.random.uniform(0, args.noise_lim)
        for j in range(args.n_bins):
            z_ph, dndz_bin_ph[j] = sim.convolve_photoz(
                sigma=args.sigma_pz, 
                zs=z_bin[j], 
                dndz_spec=dndz_bin[j]
            )

            dndz_bin_ph[j] = sim.add_noise(
                z_ph,
                dndz_bin_ph[j],
                noise_std
            )

        hypercube = sim.cosmo_hypercube(n_samples=args.n_samples)

        # Construct arglist for parallel computation of
        # the angular power spectra
        ells = np.geomspace(2, args.ell_max, args.ell_bins)
        arglist = [
            (hypercube[i], dndz_bin_ph, z_ph, ells)
            for i in range(args.n_samples)
        ]

        # Instantiate class to catch errors
        SpectraWrapper = sim.SpectraWrapper(func)

        print('Computing spectra for task {}...'.format(i))
        # Compute the angular power spectra in parallel
        with Pool(args.n_threads) as pool:
            c_ells = progress_starmap(
                SpectraWrapper,
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
    filename = '{}_{}tasks_{}samples_seed{}.h5'.format(tag, args.n_tasks, args.n_samples, args.seed)
    with h5.File(os.path.join(args.output, filename), 'w') as f:
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('dndz', data=dndz_save)
        f.create_dataset('z', data=z_save)
        f.create_dataset('dndz_params', data=dndz_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for photo-z estimation')
    parser.add_argument('--n_bins', type=int, default=5, help='Number of redshift bins')
    parser.add_argument('--n_tasks', type=int, default=30, help='Number of tasks')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--gaussian_prob', type=float, default=0.5, help='Probability of Gaussian distribution')
    parser.add_argument('--noise_lim', type=float, default=0.1, help='Standard deviation of the noise')
    parser.add_argument('--shift', type=float, default=0.01, help='Max shift in the mean redshift')
    parser.add_argument('--sigma_pz', type=float, default=0.04, help='Photo-z error')
    parser.add_argument('--z_min', type=float, default=0.05, help='Minimum redshift')
    parser.add_argument('--z_max', type=float, default=3.5, help='Maximum redshift')
    parser.add_argument('--n_threads', type=int, default=cpu_count(), help='Number of threads')
    parser.add_argument('--ell_max', type=int, default=5000, help='Maximum multipole')
    parser.add_argument('--ell_bins', type=int, default=50, help='Number of multipole bins')
    parser.add_argument('--seed', type=int, default=456, help='Random seed')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--for_cluster', action='store_true', help='Generate clustering spectra')
    args = parser.parse_args()

    main(args)