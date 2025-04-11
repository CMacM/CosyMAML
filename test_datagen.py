import os

# set the environment variable to control the number of threads
# NEEDS TO BE DONE BEFORE CCL IS IMPORTED
original_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import h5py as h5

from multiprocessing import Pool, cpu_count
from parallelbar import progress_starmap
from time import time

import argparse

import src.simulate as sim

def main(args):

    # Define a redshift distribution
    z = np.linspace(0.01, 3.0, 100)
    z_bin, dndz = sim.bin_dndz(5, z, sim.Smail_dndz)

    print('Generating {} hypercube samples...'.format(args.n_samples))
    hypercube = sim.cosmo_hypercube(n_samples=args.n_samples)

    # Construct arglist for parallel computation of
    # the angular power spectra
    ells = np.geomspace(2, args.ell_max, args.ell_bins)
    arglist = [
        (hypercube[i], dndz, z, ells)
        for i in range(args.n_samples)
    ]

    print('Computing spectra, this may take a while...')
    start = time()
    # Compute the angular power spectra in loop
    c_ells = progress_starmap(
        sim.compute_spectra_cosmicshear,
        arglist,
        n_cpu=args.n_threads
    )
    end = time()
    print('Time taken: {}s'.format(end - start))

    with h5.File(f'/exafs/400NVX2/cmacmahon/spectra_data/test_c_ells_{args.n_samples}.h5', 'w') as f:
        f.create_dataset('c_ells', data=c_ells)
        f.create_dataset('hypercube', data=hypercube)
        f.create_dataset('ells', data=ells)
        f.create_dataset('z', data=z)
        f.create_dataset('dndz', data=dndz)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a set of spectra for single task redshift distribution')
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--ell_max', type=int, default=5000)
    parser.add_argument('--ell_bins', type=int, default=50)
    parser.add_argument('--n_threads', type=int, default=cpu_count())
    args = parser.parse_args()

    main(args)