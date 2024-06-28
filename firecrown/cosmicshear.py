""" Define the likelihood factory function for the cosmic shear likelihood. """

import os
import sacc

import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian

def build_likelihood(_):
    """ Create firecrown likelihood for cosmic shear analysis"""
    # Sources map to a section of SACC file.

    # A photo-z shift bias (const shift in dNdz) is added to the sources.
    source0 = wl.WeakLensing(
        sacc_tracer='trc0', systematics=[wl.PhotoZShift(sacc_tracer='trc0')]
    )
    source1 = wl.WeakLensing(
        sacc_tracer='trc1', systematics=[wl.PhotoZShift(sacc_tracer='trc1')]
    )

    # Now we instantiate the two-point functions
    stats = [
        TwoPoint('galaxy_shear_cl_ee', source0, source0),
        TwoPoint('galaxy_shear_cl_ee', source0, source1),
        TwoPoint('galaxy_shear_cl_ee', source1, source1),
    ]

    # Next instantiate the Gaussian likelihoods
    likelihood = ConstGaussian(statistics=stats)

    # Load SACC file
    saccfile = os.path.expanduser(
        os.path.expandvars('cosmicshear.fits')
    )
    sacc_data = sacc.Sacc.load_fits(saccfile)

    # two-point functions recieve appropriate sections of SACC file
    # and sources their respective dndz
    likelihood.read(sacc_data)

    # Script is loaded by connector, framework then calls factory function and returns a likelihood instance
    return likelihood