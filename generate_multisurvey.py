import numpy as np
import pyccl as ccl

import matplotlib.pyplot as plt

import scipy.stats.qmc as qmc
import scipy.stats as stats
from scipy.interpolate import interp1d

from tqdm import trange, tqdm
from time import time
from multiprocessing import Pool, cpu_count
from parallelbar import progress_starmap

import src.datamaker as datamaker

import argparse
import os

def main(n_tasks, n_samples, seed):
    
    start = time()
    
    print("Initializing...")
    print("Producing data for {} tasks, {} samples".format(n_tasks, n_samples))

    # ell bins for C_ell
    # Define the range and the number of points
    # ell_bao = np.arange(2, 200)
    # ell_bins = np.concatenate((ell_bao, np.geomspace(200, 4000, 220)))
    ell_bins = np.unique(np.geomspace(2, 4000, 30))

    np.random.seed(seed)
    gridsize = 50
    z = np.linspace(0.01, 3.0, gridsize)

    task_means = np.random.uniform(0.8, 1.6, n_tasks)
    task_vars = np.random.uniform(0.2, 0.6, n_tasks)

    survey_pz = np.empty((n_tasks, gridsize-1))
    survey_midpoints = np.empty((n_tasks, gridsize-1))
    
    print("Generating P(z) for each task...")

    for i in range(n_tasks):    
        survey_pz[i], survey_midpoints = datamaker.gen_Pz_base(
            task_means[i],
            task_vars[i],
            grid=z
        )

    print("Generating P(z) realizations...")

    true_means = np.empty(n_tasks)
    shifts = np.empty((n_tasks, n_samples))
    qrd_pz = np.empty((n_tasks, n_samples, gridsize-1))
    for i in trange(n_tasks):
        qrd_pz[i], true_means[i] = datamaker.gen_Pz_samples(
                                            survey_pz[i],
                                            survey_midpoints,
                                            seed=14,
                                            shift=0.01,
                                            qrd_samples=n_samples
                                            )
        for j in range(n_samples):
            mean = np.trapz(qrd_pz[i, j]*survey_midpoints, survey_midpoints)
            shifts[i, j] = mean - true_means[i]

    # Priors from DES Y3 Cosmic shear fits
    OmM = np.array([0.1, 0.9])
    OmB = np.array([0.03, 0.07])
    OmC = OmM - OmB

    h = np.array([0.55, 0.91])
    n_s = np.array([0.87, 1.07])
    sigma8 = np.array([0.6, 0.9])

    print("Constructing Cosmology hypercubes...")

    inputs = 6
    X_train = np.empty((n_tasks, n_samples, inputs))

    for i in trange(n_tasks):
        # Generate new Hypercube for each task to randomise the cosmology samples
        cosmo_hypercube = datamaker.gen_hypercube(OmC, OmB, h, n_s, sigma8, n_samples)
        for j in range(inputs):
            if j != inputs-1:
                X_train[i, :, j] = cosmo_hypercube[:, j]
            else:
                X_train[i, :, j] = shifts[i, :]

    print("Computing C_ell samples, this may take a while...")

    progbar = tqdm(total=n_tasks*n_samples)

    Y_train = np.empty((n_tasks, n_samples, len(ell_bins)))
    for i in range(n_tasks):
        for j in range(n_samples):
            cosmo = ccl.Cosmology(
                Omega_c=X_train[i, j, 0],
                Omega_b=X_train[i, j, 1],
                h=X_train[i, j, 2],
                n_s=X_train[i, j, 3],
                sigma8=X_train[i, j, 4],
            )

            # Get the pz for this task and sample
            pz = qrd_pz[i, j]
            z = survey_midpoints

            # Upsample the pz to ensure accurate calculation of Cgg
            z_up = np.linspace(z[0], z[-1], 300)
            interpolator = interp1d(z, pz, kind='cubic')
            pz_up = interpolator(z_up)

            Y_train[i, j] = datamaker.gen_Cgg_autocorr(cosmo, ell_bins, z_up, pz_up)

            progbar.update(1)
    progbar.close()

    print("Collecting results and saving...")

    # Save the data
    np.savez("Cgg_samples_{}tasks_{}samples.npz".format(n_tasks, n_samples),
             X_train=X_train,
             Y_train=Y_train,
             ell_bins=ell_bins,
             survey_midpoints=survey_midpoints,
             survey_pz=survey_pz,
             qrd_pz=qrd_pz,
            )
    
    print("Finished in {:.2f} minutes".format((time()-start)/60))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-task training data for cosmological analysis.")
    parser.add_argument("--n_tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--n_samples", type=int, default=2**5, help="Number of samples per task")
    parser.add_argument("--seed", type=int, default=14, help="Random seed for reproducibility")

    args = parser.parse_args()
    
    main(args.n_tasks, args.n_samples, args.seed)