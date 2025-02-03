import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

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

masterseed = 14
np.random.seed(masterseed)
torch.manual_seed(masterseed)
torch.cuda.manual_seed(masterseed)
torch.use_deterministic_algorithms(False)

def main(args):

    # set device for all tensors
    device = args.device

    print('Pretraining emualtor with standard method...')
    # Load pretraining data for standard training emulator
    train_data, test_data, _, _ = training.load_train_test_val(
        filepath=args.pretrain_file, n_train=args.n_train, n_val=None, n_test=None, seed=masterseed,
        device=device
    )

    # Pretrain an emulator in the standard way
    train_loader = DataLoader(train_data, batch_size=5000, shuffle=False)
    X_val, y_val = test_data[:]
    pretrain_model, pretrain_time = training.train_standard_emulator(train_loader, X_val, y_val, device=device)

    # Load MAML model
    maml_model = models.FastWeightCNN(
        input_size=10,
        latent_dim=(16,16),
        output_size=750,
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
            '{}batch_{}samples_{}tasks_metalearner_weights.pt'.format(5, 1000, 20)
        )
    )

    print(f'Training / fine-tuning emulators with {args.n_finetune} samples...')
    # Load training data for MCMC dndz
    train_data, test_data, X_val, y_val, ScalerY, ScalerX = training.load_train_test_val(
        filepath=args.mcmc_train_file, n_train=args.n_finetune, n_val=4000, n_test=6000, seed=masterseed,
        device=device
    )
    del test_data # we don't need this here

    # Train a fresh emulator
    train_loader = DataLoader(train_data, batch_size=5000, shuffle=False)
    fresh_model, fresh_time = training.train_standard_emulator(train_loader, X_val, y_val, device=device)

    # finetune pretrained model
    start = time()
    training.finetune_pretrained(
        model = pretrain_model,
        n_epochs = args.n_epochs,
        train_loader = train_loader,
        device=device
    )
    pretrain_ft_time = time() - start

    # finetune MAML model
    X_train, y_train = train_data[:]
    start = time()
    task_weights, _ = metalearner.finetune(
        X_train, y_train, adapt_steps=args.n_epochs, use_new_adam=True
    )
    maml_ft_time = time() - start

    # Load in fiducial data
    S = sacc.Sacc.load_fits(args.fiducial_file)
    print(S.mean.size)

    # Define tracer combs
    tracer_combs = S.get_tracer_combinations()

    # Extract C_ell and covariance blocks
    c_ells = []
    for comb in tracer_combs:
        ell, cell = S.get_ell_cl(
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
        (0.1, 0.9), # Omega_c
        (0.01, 0.1), # Omega_b
        (0.5, 0.9), # h
        (0.7, 1.0), # sigma8
        (0.8, 1.1) # n_s
    ]

    # Comment out if not using shifts
    delta_z = 0.002 # LSST Y1 mean uncertainty
    for i in range(n_bins):
        priors.append((-delta_z, delta_z)) # Shifts for each redshift bin

    # Define the initial positions of walkers
    pos = [
        theta[0] + 1e-2 * np.random.randn(nwalkers), # Omega_c
        theta[1] + 1e-3 * np.random.randn(nwalkers), # Omega_b
        theta[2] + 1e-2 * np.random.randn(nwalkers), # h
        theta[3] + 1e-2 * np.random.randn(nwalkers), # sigma8
        theta[4] + 1e-2 * np.random.randn(nwalkers), # n_s
    ]

    # Comment out if not using shifts
    for i in range(n_bins):
        pos += (theta[5+i] + 1e-4 * np.random.randn(nwalkers),)
    pos = np.array(pos).T

    # Construct a hook class to the pretrained model
    PreTrainHook = mcmc.EmulatorHook(pretrain_model, ScalerX, ScalerY, device=device)
    # Define backend file for emcee
    pretrain_backend = f'mcmc/{nwalkers}_pretrain_emulator_mcmc_samples.h5'

    # Run the emcee sampler
    print('Running emcee with pretrained emulator...')
    start = time()
    mcmc.run_emcee(
        ClHook=PreTrainHook,
        backend_file=pretrain_backend,
        ndim=ndim,
        priors=priors,
        pos=pos,
        data_vector=data_vector,
        inv_cov=inv_cov,
        nwalkers=nwalkers,
        n_check=args.n_check,
        max_iter=args.max_iter,
        clobber=True
    )
    pretrain_mcmc_time = time() - start

    # Construct a hook class to the fresh model
    FreshHook = mcmc.EmulatorHook(fresh_model, ScalerX, ScalerY, device=device)
    # Define backend file for emcee
    fresh_backend = f'mcmc/{nwalkers}_fresh_emulator_mcmc_samples.h5'

    # Run the emcee sampler
    print('Running emcee with fresh emulator...')
    start = time()
    mcmc.run_emcee(
        ClHook=FreshHook,
        backend_file=fresh_backend,
        ndim=ndim,
        priors=priors,
        pos=pos,
        data_vector=data_vector,
        inv_cov=inv_cov,
        nwalkers=nwalkers,
        n_check=args.n_check,
        max_iter=args.max_iter,
        clobber=True
    )
    fresh_mcmc_time = time() - start

    # Construct a hook class to the MAML model
    MAMLHook = mcmc.EmulatorHook(metalearner.model, ScalerX, ScalerY, weights=task_weights, device=device)
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
        clobber=True
    )
    maml_mcmc_time = time() - start

    # Print out the MCMC times in minutes
    print('Pretrained model MCMC time:', pretrain_mcmc_time/60)
    print('Fresh model MCMC time:', fresh_mcmc_time/60)
    print('MAML model MCMC time:', maml_mcmc_time/60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_file', type=str, default='data/42_dndz_nsamples=24000.h5')
    parser.add_argument('--mcmc_train_file', type=str, default='data/mcmc_dndz_nsamples=30000.h5')
    parser.add_argument('--fiducial_file', type=str, default='mcmc/sacc_fiducial_data.fits')
    parser.add_argument('--n_train', type=int, default=20000)
    parser.add_argument('--n_finetune', type=int, default=500)
    parser.add_argument('--n_epochs', type=int, default=64)
    parser.add_argument('--n_walkers', type=int, default=76)
    parser.add_argument('--n_check', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=200000)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)

