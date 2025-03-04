import os

# set the environment variable to control the number of threads
# NEEDS TO BE DONE BEFORE CCL IS IMPORTED
original_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
os.environ['OMP_NUM_THREADS'] = '1'

import pyccl as ccl
import numpy as np
import sacc
import scipy.stats.qmc as qmc
import h5py as h5
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps

from multiprocessing import Pool, cpu_count
from parallelbar import progress_starmap
from time import time

import argparse

import src.simulate as sim

def main(args):

    # Check which type of spectra we are computing
    if args.for_cluster:
        func = sim.compute_spectra_cluster
        tag = 'cl_00'
        print('Computing clustering spectra')
    else:
        func = sim.compute_spectra_cosmicshear
        tag = 'cl_ee'
        print('Computing cosmic shear spectra')

    # Flag to say if we are generating samples representative of the MCMC data
    if args.for_mcmc:
        print('Loading MCMC N(z)...')
        if args.fiducial_file is None:
            sacc_path = 'mcmc_chains/{}_fiducial_sacc.fits'.format(tag)
            sacc_file = sacc.Sacc.load_fits(sacc_path)
        else:
            sacc_file = sacc.Sacc.load_fits(args.fiducial_file)

        # get redshift bins
        bins = sacc_file.tracers
        # convert into lists of redshifts and dndz
        z = []
        dndz = []
        for key in bins:
            z.append(bins[key].z)
            dndz.append(bins[key].nz)
    elif args.for_multipeak:
        print('Generating multi-peak N(z)...')
        peaks = [
            (0.5, 0.1, 0.3),   # Peak 1: centered at z=0.5, width=0.1, weight=0.6
            (0.75, 0.3, 0.1),   # Peak 2: centered at z=1.5, width=0.2, weight=0.3
            (3.0, 0.2, 0.6),   # Peak 3: centered at z=3.0, width=0.3, weight=0.1
        ]
        kwargs = {'peaks': peaks}
        dndz_func = sim.multi_peak_nz
        # Generate redshift values
        z_s = np.linspace(0.05, 3.5, 300)  # Redshift range from 0 to 4
        n_bins=5
        noise_lim = 0.1
        sigma_pz = 0.05
        # Bin the redshift distribution
        z_bin, dndz_bin = sim.bin_dndz(n_bins, z_s, dndz_func, **kwargs)

        # Convolve with photo-z and add noise to the redshift distribution
        dndz_bin_ph = np.zeros((n_bins, len(z_s)))
        noise_std = np.random.uniform(0, noise_lim)
        for j in range(n_bins):
            z_ph, dndz_bin_ph[j] = sim.convolve_photoz(
                sigma=sigma_pz, 
                zs=z_bin[j], 
                dndz_spec=dndz_bin[j]
            )

            dndz_bin_ph[j] = sim.add_noise(
                z_ph,
                dndz_bin_ph[j],
                noise_std
            )
        z = [z_ph, z_ph, z_ph, z_ph, z_ph]
        dndz = dndz_bin_ph
    else:
        print('Loading random N(z)...')
        #### OR choose another dndz ####
        # Load a distribution from original training sample
        filepath = args.multitask_file
        f = h5.File(filepath, 'r')

        data_ind = 3
        print('Using data index {}'.format(data_ind))
        z = f['z'][data_ind]
        dndz = f['dndz'][data_ind]

    # check all z arrays are the same
    assert all([np.allclose(z[0], z[i]) for i in range(1, len(z))])

    print('Generating {} hypercube samples...'.format(args.n_samples))
    hypercube = sim.cosmo_hypercube(n_samples=args.n_samples)

    # Construct arglist for parallel computation of
    # the angular power spectra
    ells = np.geomspace(2, args.ell_max, args.ell_bins)
    arglist = [
        (hypercube[i], dndz, z[0], ells)
        for i in range(args.n_samples)
    ]

    # Instantiate class to catch errors
    SpectraWrapper = sim.SpectraWrapper(func)

    print('Computing spectra, this may take a while...')
    start = time()
    # Compute the angular power spectra in parallel
    n_threads = cpu_count()
    c_ells = progress_starmap(
        SpectraWrapper,
        arglist,
        n_cpu=n_threads
    )

    end = time()
    print('Time taken: {}s'.format(end - start))

    # Write C_ells and cosmology to h5 file
    if args.for_mcmc:
        filename = 'spectra_data/{}_mcmc_dndz_nsamples={}.h5'.format(tag,args.n_samples)
    elif args.for_multipeak:
        filename = 'spectra_data/{}_multi_peak_dndz_nsamples={}.h5'.format(tag,args.n_samples)
    else:
        filename = 'spectra_data/{}_{}_dndz_nsamples={}.h5'.format(tag,data_ind,args.n_samples)

    with h5.File(filename, 'w') as f:
        f.create_dataset('c_ells', data=c_ells)
        f.create_dataset('hypercube', data=hypercube)
        f.create_dataset('ells', data=ells)
        f.create_dataset('z', data=z[0])
        f.create_dataset('dndz', data=dndz) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a set of spectra for single task redshift distribution')
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--ell_max', type=int, default=5000)
    parser.add_argument('--ell_bins', type=int, default=50)
    parser.add_argument('--for_cluster', action='store_true')
    parser.add_argument('--for_mcmc', action='store_true')
    parser.add_argument('--for_multipeak', action='store_true')
    parser.add_argument('--multitask_file', type=str, default='spectra_data/60tasks_2000samples_seed456.h5')
    parser.add_argument('--fiducial_file', type=str, default=None)
    args = parser.parse_args()

    main(args)