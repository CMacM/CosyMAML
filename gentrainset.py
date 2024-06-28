import pyccl as ccl
import numpy as np
import tqdm
import argparse

# training data in this case needs different source dists
def main(n_samples, n_tasks, seed):

    z = np.linspace(0., 3., 256)
    rng = np.random.RandomState(seed)
        
    l_arr = np.unique(np.geomspace(2, 60000, 30).astype(int))

    C_ells = np.empty((n_tasks, n_samples, len(l_arr)))
    z0 = np.empty(n_tasks)
    dndz = np.empty((n_tasks, len(z)))

    Omega_b_arr = rng.uniform(0.04, 0.06, n_samples)
    Omega_c_arr = rng.uniform(0.25, 0.35, n_samples)
    h_arr = rng.uniform(0.6, 0.8, n_samples)
    sigma8_arr = rng.uniform(0.7, 0.9, n_samples)
    n_s_arr = rng.uniform(0.9, 1.1, n_samples)

    for i in tqdm.trange(n_tasks):
        z0[i] = rng.uniform(0.1, 0.3)
        dndz[i,:] = 1./(2.*z0[i]) * (z / z0[i])**2 * np.exp(-z/z0[i])
        for j in range(n_samples):

            cosmo = ccl.Cosmology(Omega_c=Omega_c_arr[j], Omega_b=Omega_b_arr[j],
                                h=h_arr[j], sigma8=sigma8_arr[j], n_s=n_s_arr[j])
            lensTracer = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[i,:]))
            C_ells[i,j,:] = ccl.angular_cl(cosmo, lensTracer, lensTracer, l_arr)

    np.savez('Cgg_data_multi_task.npz', C_ells=C_ells, l_arr=l_arr, Omega_b_arr=Omega_b_arr,
                Omega_c_arr=Omega_c_arr, h_arr=h_arr, sigma8_arr=sigma8_arr, n_s_arr=n_s_arr,
                z0=z0, dndz=dndz)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-task training data for cosmological analysis.")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples per task")
    parser.add_argument("--n_tasks", type=int, default=500, help="Number of tasks")
    parser.add_argument("--seed", type=int, default=14, help="Random seed for reproducibility")

    args = parser.parse_args()
    
    main(args.n_samples, args.n_tasks, args.seed)