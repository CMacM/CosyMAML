import numpy as np
import pyccl as ccl

import matplotlib.pyplot as plt

import scipy.stats.qmc as qmc
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.integrate import simps, quad

from tqdm import trange, tqdm
from time import time
from multiprocessing import Pool, cpu_count
from parallelbar import progress_starmap

import CosyMAML.src.simulate as simulate

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
    
    print("Generating P(z) for each task...")

    survey_pz = np.empty((n_tasks, gridsize-1))
    n_bins = 4
    true_means = np.empty((n_tasks, n_bins))
    shifts = np.empty((n_tasks, n_bins, n_samples))
    qrd_pz = np.empty((n_tasks, n_bins, n_samples, gridsize-1))
    for i in trange(n_tasks):    
        survey_pz[i], z_mid = simulate.gen_Pz_base(
            task_means[i],
            task_vars[i],
            grid=z
        )

        # Normalize the distribution
        area = simps(survey_pz[i], z_mid)  # Integrate dndz_s over z to get the area under the curve
        pdf = survey_pz[i] / area  # Normalize to make it a PDF

        # Compute the cumulative distribution function (CDF)
        cdf = np.cumsum(pdf) * (z_mid[1] - z_mid[0])  # Approximate the integral to get the CDF

        # Ensure CDF starts at 0 and ends at 1 exactly
        # cdf = np.concatenate(([0], cdf))
        # cdf[-1] = 1.0

        # Interpolate the CDF to find the bin edges
        inverse_cdf = interp1d(cdf, z_mid, fill_value="extrapolate")

        # Define the CDF values for the bin edges
        cdf_values = np.linspace(0, 1, n_bins+1)

        # Find the corresponding z values (bin edges) for these CDF values
        bin_edges = inverse_cdf(cdf_values)
        if np.isnan(bin_edges[0]) or np.isinf(bin_edges[0]):
            bin_edges[0] = z_mid[0]
        if np.isnan(bin_edges[-1]) or np.isinf(bin_edges[-1]):
            bin_edges[-1] = z_mid[-1]

        print("Generating P(z) realizations...")
        for j in range(n_bins):
            zs = np.linspace(bin_edges[j], bin_edges[j+1], gridsize)
            spec_bin, zs = simulate.gen_Pz_base(
                            task_means[i],
                            task_vars[i],
                            grid=zs
                        )
            
            # Convolve with photo-z
            sigma_z = 0.05 * (1 + zs)

            z_ph = np.linspace(0, 3.5, len(zs))

            # find probability of galaxy with true redshift z_s to be measured at redshift z_ph
            integrand1 = np.zeros([len(zs),len(z_ph)])
            p_zs_zph = np.zeros([len(zs),len(z_ph)])
            for k in range(len(zs)):
                p_zs_zph[k,:] =  (1. / (np.sqrt(2. * np.pi) * sigma_z[k])) * np.exp(-((z_ph - zs[k])**2) / (2. * sigma_z[k]**2))

            integrand1 = p_zs_zph * spec_bin[:,None]   

            # integrate over z_s to get dN
            integral1 = simps(integrand1, zs, axis=0)
            dN = integral1
            
            dz_ph = simps(dN, z_ph)
            dndz_ph = dN/dz_ph

            qrd_pz[i,j], true_means[i,j] = simulate.gen_Pz_samples(
                                                        dndz_ph,
                                                        zs,
                                                        seed=14,
                                                        shift=0.005,
                                                        qrd_samples=n_samples
                                                    )
            
            for k in range(n_samples):
                mean = np.trapz(qrd_pz[i, j, k]*zs, zs)
                shifts[i, j, k] = mean - true_means[i, j]

    # Priors from DES Y3 Cosmic shear fits
    OmM = np.array([0.1, 0.9])
    OmB = np.array([0.03, 0.07])
    OmC = OmM - OmB

    h = np.array([0.55, 0.91])
    n_s = np.array([0.87, 1.07])
    sigma8 = np.array([0.6, 0.9])

    print("Constructing Cosmology hypercubes...")

    inputs = 7
    X_train = np.empty((n_tasks, n_samples, inputs))

    for i in trange(n_tasks):
        # Generate new Hypercube for each task to randomise the cosmology samples
        cosmo_hypercube = simulate.gen_hypercube(OmC, OmB, h, n_s, sigma8, n_samples)
        for j in range(inputs):
            if j < inputs-2:
                X_train[i, :, j] = cosmo_hypercube[:, j]

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

            z_up = np.linspace(z_mid[0], z_mid[-1], 300)

            #### Randomly choose one combination of C_ell for each sample ####
            bins = np.arange(n_bins)
            bin1, bin2 = np.random.choice(bins, size=2, replace=False)
            if bin1 > bin2:
                bin1, bin2 = bin2, bin1

            pz1 = qrd_pz[i, bin1, j]
            interpolator = interp1d(z_mid, pz1, kind='cubic', fill_value="extrapolate")
            pz1 = interpolator(z_up)

            pz2 = qrd_pz[i, bin2, j]
            interpolator = interp1d(z_mid, pz2, kind='cubic', fill_value="extrapolate")
            pz2 = interpolator(z_up)
            
            shearTracer1 = ccl.WeakLensingTracer(cosmo, dndz=(z_up, pz1))
            shearTracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z_up, pz2))

            Y_train[i, j, :] = ccl.angular_cl(cosmo, shearTracer1, shearTracer2, ell_bins)
            X_train[i, j, 5] = shifts[i, bin1, j]
            X_train[i, j, 6] = shifts[i, bin2, j]

            progbar.update(1)

            #### Generate every possible combination of C_ell for each sample ####
            # for k in range(n_bins):
            #     # Get the pz for this task and sample
            #     pz1 = qrd_pz[i, k, j]
            #     interpolator = interp1d(z_mid, pz1, kind='cubic', fill_value="extrapolate")
            #     pz1 = interpolator(z_up)
            #     for l in range(n_bins):
            #         pz2 = qrd_pz[i, l, j]
            #         interpolator = interp1d(z_mid, pz2, kind='cubic', fill_value="extrapolate")
            #         pz2 = interpolator(z_up)
            #         if k == l:
            #             shearTracer = ccl.WeakLensingTracer(cosmo, dndz=(z_up, pz1))
            #             Y_train[i, k, j] = ccl.angular_cl(cosmo, shearTracer, shearTracer, ell_bins)
            #             X_train[i, j, 5] = shifts[i, k, j]
            #             X_train[i, j, 6] = shifts[i, l, j]
            #         elif l > k:
            #             shearTracer1 = ccl.WeakLensingTracer(cosmo, dndz=(z_up, pz1))
            #             shearTracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z_up, pz2))
            #             Y_train[i, k, j] = ccl.angular_cl(cosmo, shearTracer1, shearTracer2, ell_bins)
            #             X_train[i, j, 5] = shifts[i, k, j]
            #             X_train[i, j, 6] = shifts[i, l, j]
            #         else:
            #             pass
            #   progbar.update(1)

    progbar.close()

    print("Collecting results and saving...")

    # Save the data
    np.savez("Cgg_samples_{}tasks_{}samples.npz".format(n_tasks, n_samples),
             X_train=X_train,
             Y_train=Y_train,
             ell_bins=ell_bins,
             survey_midpoints=z,
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