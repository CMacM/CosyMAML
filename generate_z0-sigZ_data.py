import os

# set the environment variable to control the number of threads
# NEEDS TO BE DONE BEFORE CCL IS IMPORTED
original_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
import src.simulate as sim
import multiprocessing as mp
import h5py as h5
from parallelbar import progress_starmap

from argparse import ArgumentParser

def build_nz_bins(sigma_z, n_bins=5):
    """Build redshift bins and convolve with photo-z errors."""
    # Construct the redshift distribution
    z = np.linspace(0.05, 3.5, 300)

    # Bin the redshift distribution with equal area per bin
    kwargs = {'z0': 0.13, 'alpha': 0.78}
    z_bins, dndz_bins = sim.bin_dndz(n_bins, z, sim.Smail_dndz, **kwargs)

    # Convolve photo-z errors
    dndz_bin_ph = np.zeros((n_bins, len(z)))
    for j in range(n_bins):
        z_ph, dndz_bin_ph[j] = sim.convolve_photoz(
            sigma=sigma_z, 
            zs=z_bins[j], 
            dndz_spec=dndz_bins[j]
        )

    return z_ph, dndz_bin_ph

# Define NLA IA models
def nla_ia(z, a_ia, eta, z0):
    """Non-linear alignment intrinsic alignment model."""
    return a_ia * ((1 + z) / (1 + z0))**eta

def compute_cl_ia(cosmo, z_ph, ells, dndz_bin_ph, A_ia_model):
    n_bins = dndz_bin_ph.shape[0]
    inds = list(zip(*np.tril_indices(n_bins)))

    c_ells = np.empty((len(inds), len(ells)))
    for i, arg in enumerate(inds):
        j, k = arg
        tracer1 = ccl.WeakLensingTracer(
            cosmo,
            dndz=(z_ph, dndz_bin_ph[j]),
            ia_bias=(z_ph, A_ia_model)
        )
        tracer2 = ccl.WeakLensingTracer(
            cosmo,
            dndz=(z_ph, dndz_bin_ph[k]),
            ia_bias=(z_ph, A_ia_model)
        )
        c_ells[i,:] = ccl.angular_cl(cosmo, tracer1, tracer2, ells)

    return c_ells.flatten()

def main(args):

    # Construct the redshift distribution
    z = np.linspace(0.05, 3.5, 300)
    #nz /= np.trapz(nz, z)  # Normalize

    # Bin the redshift distribution with equal area per bin
    n_bins = 5

    # Define multipoles
    ells = np.geomspace(2, 2000, 50)

    # Seed
    np.random.seed(14)

    # Meta parameters for IA sampling
    n_tasks = args.n_tasks

    # Sample z0 and sigma_z values for tasks
    z0_range = np.random.uniform(0.42, 0.82, n_tasks)
    sigma_z_range = np.random.uniform(0.02, 0.07, n_tasks)

    # create list to hold dndz bins for each task
    dndz_bin_ph = []
    for i in range(n_tasks):
        z_ph, dndz_bin_ph_task = build_nz_bins(sigma_z=sigma_z_range[i], n_bins=n_bins)
        dndz_bin_ph.append(dndz_bin_ph_task)

    n_shots = args.n_shots

    c_ells_array = np.empty((
        n_tasks,
        n_shots,
        (n_bins * (n_bins + 1)) // 2 * len(ells)
    ))

    hypercube_array = np.empty((
        n_tasks,
        n_shots,
        7
    ))

    for i in range(n_tasks):
        print("Computing spectra for task {} of {}".format(i+1, n_tasks))

        # Sample cosmologies and IA params
        hypercube = sim.cosmo_hypercube(n_samples=n_shots, use_ia=True)
        # remove dndz_shifts from hypercube
        hypercube = hypercube[:,:-5]

        cosmologies = [
            ccl.Cosmology(
                Omega_c=hypercube[j][0],
                Omega_b=hypercube[j][1],
                h=hypercube[j][2],
                sigma8=hypercube[j][3],
                n_s=hypercube[j][4]
            )
            for j in range(n_shots)
        ]

        A_ia_samples = [
            nla_ia(
                z_ph,
                a_ia=hypercube[j,5],
                eta=hypercube[j,6],
                z0=z0_range[i]
            ) for j in range(n_shots)
        ]

        # construct arglist for parallel computation of power spectra
        arglist = [
            (cosmologies[j], z_ph, ells, dndz_bin_ph[i], A_ia_samples[j])
            for j in range(n_shots)
        ]

        # Instantiate error wrapper
        #SpectraWrapper = sim.SpectraWrapper(compute_cl_ia)

        print("Starting parallel computation...")
        with mp.Pool(mp.cpu_count()//2) as pool:
            c_ells = progress_starmap(
                compute_cl_ia,
                arglist,
                n_cpu=mp.cpu_count()//2
            )

        c_ells_array[i] = np.array(c_ells)
        hypercube_array[i] = hypercube

    # Save the data as h5 file
    filename = 'cl_ee_z0_sigZ_{}tasks_{}samples_seed{}.h5'.format(n_tasks, n_shots, 14)
    with h5.File(filename, 'w') as f:
        f.create_dataset('cosmo_params', data=hypercube_array)
        f.create_dataset('c_ells', data=c_ells_array)
        f.create_dataset('dndz', data=dndz_bin_ph)
        f.create_dataset('z', data=z_ph)
        f.create_dataset('z0_samples', data=z0_range)
        f.create_dataset('sigma_z_samples', data=sigma_z_range)
    print("Data saved to {}".format(filename))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_tasks', type=int, default=10, help='Number of tasks to generate')
    parser.add_argument('--n_shots', type=int, default=50, help='Number of samples per task')
    args = parser.parse_args()

    main(args)