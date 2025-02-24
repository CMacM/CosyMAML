import numpy as np
import os
import pyccl as ccl

from scipy.integrate import simps
from scipy.interpolate import interp1d
import scipy.stats.qmc as qmc

class SpectraWrapper():
    def __init__(self, func):
        self.func = func

    def __call__(self, theta, dndz_ph_bins, z_ph, ells):
        try:
            return self.func(theta, dndz_ph_bins, z_ph, ells)
        except Exception as e:
            # Dump data and re-raise the exception
            print('Caught exception in worker thread:')
            print('Dumping data for debugging')
            np.savez(
                os.path.join(
                    'spectra_data', 'debug_worker{}.npz'.format(os.getpid())
                    ),
                theta=theta,
                dndz_ph_bins=dndz_ph_bins,
                z_ph=z_ph,
                ells=ells
            )
            raise e

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

def multi_peak_nz(z, peaks):
    '''
    Generate a redshift distribution with multiple peaks.
    Args:
    z: redshift (array-like)
    peaks: peak locations (array-like)
    '''
    # Calculate the multi-peaked distribution
    dndz = np.zeros_like(z)
    for center, width, weight in peaks:
        dndz += weight * np.exp(-0.5 * ((z - center) / width) ** 2)
    return dndz

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

def cosmo_hypercube(
        Omega_c=np.array([0.165, 0.45]),
        Omega_b=np.array([0.025, 0.075]),
        h=np.array([0.35, 1.15]),
        n_s=np.array([0.75, 1.15]),
        sigma8=np.array([0.6, 1.05]),
        delta_z=np.array([-0.0045, 0.0045]),
        n_samples=1000,
        dim=10
    ):

    hyperframe = qmc.LatinHypercube(d=dim)
    hyperunits = hyperframe.random(n_samples)

    # Rescale the hypercube for provided param ranges
    l_bounds = [
        Omega_c[0], Omega_b[0], h[0], sigma8[0], n_s[0]
    ]
    u_bounds = [
        Omega_c[1], Omega_b[1], h[1], sigma8[1], n_s[1]
    ]

    for _ in range(dim-5):
        l_bounds.append(delta_z[0])
        u_bounds.append(delta_z[1])

    return qmc.scale(hyperunits, np.array(l_bounds), np.array(u_bounds))

def compute_spectra_cosmicshear(theta, dndz_ph_bins, z_ph, ells):
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

def compute_spectra_cluster(theta, dndz_ph_bins, z_ph, ells, bias=1.5):
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
        tracer1 = ccl.NumberCountsTracer(
            cosmo,
            dndz=(z_ph_shifted[j], dndz_ph_bins[j]),
            bias=(z_ph_shifted[j], bias*np.ones_like(z_ph_shifted[j])),
            has_rsd=False
        )
        tracer2 = ccl.NumberCountsTracer(
            cosmo,
            dndz=(z_ph_shifted[k], dndz_ph_bins[k]),
            bias=(z_ph_shifted[k], bias*np.ones_like(z_ph_shifted[k])),
            has_rsd=False
        )
        c_ells[i,:] = ccl.angular_cl(cosmo, tracer1, tracer2, ells)
    
    return c_ells.flatten()

