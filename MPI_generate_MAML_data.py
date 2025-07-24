import os

# set the environment variable to control the number of threads
# NEEDS TO BE DONE BEFORE CCL IS IMPORTED
original_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import h5py as h5
from scipy.stats import qmc

from mpi4py import MPI
import argparse
import src.simulate as sim

def main(args):
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    np.random.seed(args.seed)

    if rank == 0:
        print(f'Running with {size} MPI ranks...')
                
        if args.for_cluster:
            tag = 'cl_00'
            func = sim.compute_spectra_cluster
            inds = [(i, j) for i in range(args.n_bins) for j in range(args.n_bins) if i == j]
        else:
            tag = 'cl_ee'
            func = sim.compute_spectra_cosmicshear
            inds = list(zip(*np.tril_indices(args.n_bins)))

    z = np.linspace(args.z_min, args.z_max, 300)
    ell_bins = args.ell_bins

    # Latin hypercube setup for N(z) (done on rank 0 and broadcast)
    if rank == 0:
        hyperframe = qmc.LatinHypercube(d=2)
        z0_smail = np.array([0.1, 0.2])
        alpha_smail = np.array([0.6, 1.0])

        z0_gaussian = np.array([0.2, 1.5])
        sigma_gaussian = np.array([0.2, 0.6])

        hyperunits_smail = hyperframe.random(args.n_tasks // 2)
        hyperunits_gaussian = hyperframe.random(args.n_tasks // 2)

        l_bounds_smail = np.array([z0_smail[0], alpha_smail[0]])
        u_bounds_smail = np.array([z0_smail[1], alpha_smail[1]])
        hypercube_smail = qmc.scale(hyperunits_smail, l_bounds_smail, u_bounds_smail)

        l_bounds_gaussian = np.array([z0_gaussian[0], sigma_gaussian[0]])
        u_bounds_gaussian = np.array([z0_gaussian[1], sigma_gaussian[1]])
        hypercube_gaussian = qmc.scale(hyperunits_gaussian, l_bounds_gaussian, u_bounds_gaussian)
    else:
        hypercube_smail = None
        hypercube_gaussian = None

    # Broadcast hypercube data to all ranks
    hypercube_smail = comm.bcast(hypercube_smail, root=0)
    hypercube_gaussian = comm.bcast(hypercube_gaussian, root=0)

    smail_i, gaussian_i = 0, 0

    if rank == 0:
        y_train = np.empty((args.n_tasks, args.n_samples, ell_bins * len(inds)))
        X_train = np.empty((args.n_tasks, args.n_samples, 10))
        dndz_save = np.empty((args.n_tasks, args.n_bins, len(z)))
        z_save = np.empty((args.n_tasks, args.n_bins, len(z)))
        dndz_params = []
        model_type = []

    for task_id in range(args.n_tasks):
        if task_id % 2 == 0:
            dndz_func = sim.Smail_dndz
            z0, alpha = hypercube_smail[smail_i]
            kwargs = {'z0': z0, 'alpha': alpha}
            smail_i += 1
        else:
            dndz_func = sim.Gaussian_dndz
            z0, sigma = hypercube_gaussian[gaussian_i]
            kwargs = {'z0': z0, 'sigma': sigma}
            gaussian_i += 1

        if rank == 0:
            z_bin, dndz_bin = sim.bin_dndz(args.n_bins, z, dndz_func, **kwargs)
            dndz_bin_ph = np.zeros((args.n_bins, len(z)))

            noise_std = np.random.uniform(0, args.noise_lim)
            for j in range(args.n_bins):
                z_ph, dndz_bin_ph[j] = sim.convolve_photoz(sigma=args.sigma_pz, zs=z_bin[j], dndz_spec=dndz_bin[j])
                dndz_bin_ph[j] = sim.add_noise(z_ph, dndz_bin_ph[j], noise_std)
        else:
            dndz_bin_ph = None
            z_ph = None

        dndz_bin_ph = comm.bcast(dndz_bin_ph, root=0)
        z_ph = comm.bcast(z_ph, root=0)

        ells = np.geomspace(2, args.ell_max, args.ell_bins)
        hypercube = sim.cosmo_hypercube(n_samples=args.n_samples)

        sample_indices = np.array_split(range(args.n_samples), size)
        my_indices = sample_indices[rank]

        SpectraWrapper = sim.SpectraWrapper(func)
        my_results = [SpectraWrapper(hypercube[i], dndz_bin_ph, z_ph, ells) for i in my_indices]

        gathered_results = comm.gather(my_results, root=0)

        if rank == 0:
            c_ells = [item for sublist in gathered_results for item in sublist]
            y_train[task_id] = np.array(c_ells)
            X_train[task_id] = hypercube
            dndz_save[task_id] = dndz_bin_ph
            z_save[task_id] = z_ph
            dndz_params.append([z0, alpha] if task_id % 2 == 0 else [z0, sigma])
            model_type.append(dndz_func.__name__)

            print(f'Finished task {task_id + 1} of {args.n_tasks}...')

    if rank != 0:
        return

    if rank == 0:
        filename = f'{tag}_{args.n_tasks}tasks_{args.n_samples}samples_seed{args.seed}.h5'
        with h5.File(os.path.join(args.output, filename), 'w') as f:
            f.create_dataset('X_train', data=X_train)
            f.create_dataset('y_train', data=y_train)
            f.create_dataset('dndz', data=dndz_save)
            f.create_dataset('z', data=z_save)
            f.create_dataset('dndz_params', data=dndz_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for photo-z estimation')
    parser.add_argument('--n_bins', type=int, default=5)
    parser.add_argument('--n_tasks', type=int, default=30)
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--gaussian_prob', type=float, default=0.5)
    parser.add_argument('--noise_lim', type=float, default=0.1)
    parser.add_argument('--shift', type=float, default=0.01)
    parser.add_argument('--sigma_pz', type=float, default=0.04)
    parser.add_argument('--z_min', type=float, default=0.05)
    parser.add_argument('--z_max', type=float, default=3.5)
    parser.add_argument('--ell_max', type=int, default=5000)
    parser.add_argument('--ell_bins', type=int, default=50)
    parser.add_argument('--seed', type=int, default=456)
    parser.add_argument('--output', type=str, default='/exafs/400NVX2/cmacmahon/spectra_data/')
    parser.add_argument('--for_cluster', action='store_true')
    args = parser.parse_args()

    main(args)