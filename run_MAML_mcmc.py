import os
import numpy as np
import torch
import sacc
import os
from torch.utils.data import DataLoader

import src.training as training
import src.models as models
import src.mcmc as mcmc

from time import time
from importlib import reload
reload(training)
reload(models)

import argparse

masterseed = 363
np.random.seed(masterseed)
torch.manual_seed(masterseed)
torch.cuda.manual_seed(masterseed)

def main(args):

    print('Initializing MAML model...')
    # set device for all tensors
    device = args.device

    if args.for_cluster:
        print('Computing clustering spectra')
        sacc_type = 'cl_00'
    else:
        print('Computing cosmic shear spectra')
        sacc_type = 'cl_ee'

    print(f'Training / fine-tuning emulators with {args.n_finetune} samples...')
    # Load training data for MCMC dndz
    train_data, test_data, X_val, y_val, ScalerY, ScalerX = training.load_train_test_val(
        filepath=args.mcmc_train_file, n_train=args.n_finetune, n_val=args.n_val, n_test=None, seed=masterseed,
        device=device
    )
    X_train, y_train = train_data[:]
    del test_data # we don't need this here

    # Obtain dimensions of input and output
    in_size = X_train.shape[1]
    out_size = y_train.shape[1]

        # Load MAML model
    maml_model = models.FastWeightCNN(
        input_size=in_size,
        latent_dim=(16,16),
        output_size=out_size,
        dropout_rate=0.2
    )

    # Create a new MetaLearner instance
    metalearner = training.MetaLearner(
        model=maml_model,
        outer_lr=0.01,
        inner_lr=0.001,
        loss_fn=torch.nn.MSELoss,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        seed=masterseed,
        device=device
    )

    # Load metaleraner weights
    metalearner.model.load_state_dict(
        torch.load(
            '{}_{}batch_{}samples_{}tasks_MAML_weights.pt'.format(sacc_type, 5, 1000, 20)
        )
    )

    # Finetune the MAML model to get new weights
    if args.n_epochs > 0:
        task_weights, _ = metalearner.finetune(
            X_train,
            y_train,
            adapt_steps=args.n_epochs,
            use_new_adam=True
        )
    else:
        task_weights, _ = metalearner.finetune(
            X_train,
            y_train,
            x_val=X_val,
            y_val=y_val,
            use_new_adam=True
        )

    # Load in fiducial data
    sacc_path = 'mcmc_chains/{}_fiducial_sacc.fits'.format(sacc_type)
    S = sacc.Sacc.load_fits(sacc_path)

    # Define tracer combs
    tracer_combs = S.get_tracer_combinations()

    # Extract C_ell and covariance blocks
    c_ells = []
    for comb in tracer_combs:
        _, cell = S.get_ell_cl(
            data_type='cl_ee',
            tracer1=comb[0],
            tracer2=comb[1],
            return_cov=False
        )
        c_ells.append(cell)

    # get covariance matrix
    cov_full = S.covariance.covmat

    # prepare data for MCMC
    theta = [0.27, 0.045, 0.67, 0.83, 0.96]

    n_bins = len(S.tracers)

    # Comment out if not using shifts
    for i in range(n_bins):
        theta.append(0.0)

    inv_cov = np.linalg.inv(cov_full)
    data_vector = np.concatenate(c_ells)

    # Initialize the walkers
    nwalkers = args.n_walkers
    ndim = len(theta)

    # Define the priors
    priors = [
        (0.17, 0.4), # Omega_c
        (0.03, 0.07), # Omega_b
        (0.4, 1.1), # h
        (0.65, 1.0), # sigma8
        (0.8, 1.1) # n_s
    ]

    # Comment out if not using shifts
    delta_z = 0.004 # LSST Y1 mean uncertainty
    for i in range(n_bins):
        priors.append((-delta_z, delta_z)) # Shifts for each redshift bin

    # 10% spread around the known good point
    spreads = 0.1 * np.array(theta)
    for i in range(n_bins):
        spreads[n_bins+i] += 4e-4
    pos = [theta + spreads * np.random.randn(ndim) for _ in range(nwalkers)]

        # Define the initial positions
    # pos = [
    #     theta[0] + 1e-2 * np.random.randn(nwalkers), # Omega_c
    #     theta[1] + 1e-3 * np.random.randn(nwalkers), # Omega_b
    #     theta[2] + 1e-2 * np.random.randn(nwalkers), # h
    #     theta[3] + 1e-2 * np.random.randn(nwalkers), # sigma8
    #     theta[4] + 1e-2 * np.random.randn(nwalkers), # n_s
    # ]

    pos = np.array(pos)

    # Construct a hook class to the MAML model
    MAMLHook = mcmc.EmulatorHook(
        metalearner.model,
        ScalerX,
        ScalerY,
        weights=task_weights,
        device=device
    )
    # Define backend file for emcee
    maml_backend = f'mcmc/{nwalkers}_maml_emulator_mcmc_samples.h5'

    # Run the emcee sampler
    print('Running emcee with MAML emulator...')
    start = time()
    mcmc.run_emcee(
        ClHook=MAMLHook,
        backend_file=maml_backend,
        ndim=ndim,
        priors=priors,
        pos=pos,
        data_vector=data_vector,
        inv_cov=inv_cov,
        nwalkers=nwalkers,
        n_check=args.n_check,
        max_iter=args.max_iter,
        tau_factor=args.tau_factor,
        clobber=True,
        progress=True
    )
    maml_mcmc_time = time() - start

    # Print out the MCMC times in minutes
    print('MAML model MCMC time:', maml_mcmc_time/60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mcmc_train_file', type=str, default='data/mcmc_dndz_nsamples=30000.h5')
    parser.add_argument('--fiducial_file', type=str, default='mcmc/sacc_fiducial_data.fits')
    parser.add_argument('--n_finetune', type=int, default=500)
    parser.add_argument('--n_val', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=64)
    parser.add_argument('--n_walkers', type=int, default=76)
    parser.add_argument('--n_check', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tau_factor', type=float, default=50)
    args = parser.parse_args()
    main(args)