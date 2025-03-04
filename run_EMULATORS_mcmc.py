import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
import sacc
import os
from torch.utils.data import DataLoader, TensorDataset

import src.training as training
import src.models as models
import src.mcmc as mcmc
import h5py as h5

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
    train_data, _, X_val, y_val, _, _ = training.load_train_test_val(
        filepath=args.pretrain_file, n_train=args.n_train, n_val=4000, n_test=None, seed=masterseed,
        device=device
    )

    # Pretrain an emulator in the standard way
    train_loader = DataLoader(train_data, batch_size=5000, shuffle=False)
    print('Pre-training emulator with {} samples...'.format(len(train_data)))
    pretrain_model, pretrain_time, _, _ = training.train_standard_emulator(train_loader, X_val, y_val, device=device)
    print('Pre-training time:', pretrain_time)
    
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
            'weights/cl_ee_5batch_1000samples_20tasks_metalearner_weights.pt'.format(5, 1000, 20)
        )
    )

    print(f'Training / fine-tuning emulators with {args.n_finetune} samples...')
    # Load training data for MCMC dndz
    train_data, _, X_val, y_val, ScalerY, ScalerX = training.load_train_test_val(
        filepath=args.mcmc_train_file, n_train=args.n_finetune, n_val=100, n_test=None, seed=masterseed,
        device=device
    )

    # Train a fresh emulator
    train_loader = DataLoader(train_data, batch_size=5000, shuffle=False)
    fresh_model, fresh_time, losses, _ = training.train_standard_emulator(train_loader, X_val, y_val, device=device)
    print('Fresh training time:', fresh_time)
    print('Final loss:', losses[-1])

    # finetune pretrained model
    start = time()
    losses = training.finetune_pretrained(
        model = pretrain_model,
        #x_val=X_val,
        #y_val=y_val,
        n_epochs=64,
        train_loader = train_loader,
        device=device
    )
    print('Pretrained finetuning time:', time() - start)
    print('Final loss:', losses[-1])

    # finetune MAML model
    X_train, y_train = train_data[:]
    start = time()
    task_weights, losses = metalearner.finetune(
        X_train, 
        y_train, 
        #x_val=X_val,
        #y_val=y_val,
        adapt_steps=64,
        use_new_adam=True
    )
    print('MAML finetuning time:', time() - start)
    print('Final loss:', losses[-1])

    # Load in fiducial data
    S = sacc.Sacc.load_fits(args.fiducial_file)
    print(S.mean.size)

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

    pos = np.array(pos)

    # Construct a hook class to the pretrained model
    PreTrainHook = mcmc.EmulatorHook(pretrain_model, ScalerX, ScalerY, device=device)
    # Define backend file for emcee
    pretrain_backend = f'mcmc_chains/{nwalkers}_pretrain_emulator_mcmc_{args.tau_factor}tau.h5'

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
        tau_factor=args.tau_factor,
        clobber=True,
        progress=True
    )
    pretrain_mcmc_time = time() - start

    # Construct a hook class to the fresh model
    FreshHook = mcmc.EmulatorHook(fresh_model, ScalerX, ScalerY, device=device)
    # Define backend file for emcee
    fresh_backend = f'mcmc_chains/{nwalkers}_fresh_emulator_mcmc_{args.tau_factor}tau.h5'

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
        tau_factor=args.tau_factor,
        clobber=True,
        progress=True
    )
    fresh_mcmc_time = time() - start

    # Construct a hook class to the MAML model
    MAMLHook = mcmc.EmulatorHook(metalearner.model, ScalerX, ScalerY, weights=task_weights, device=device)
    # Define backend file for emcee
    maml_backend = f'mcmc_chains/{nwalkers}_maml_emulator_mcmc_{args.tau_factor}tau.h5'

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
    print('Pretrained model MCMC time:', pretrain_mcmc_time/60)
    print('Fresh model MCMC time:', fresh_mcmc_time/60)
    print('MAML model MCMC time:', maml_mcmc_time/60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_file', type=str, default='spectra_data/cl_ee_27_dndz_nsamples=30000.h5')
    parser.add_argument('--mcmc_train_file', type=str, default='spectra_data/cl_ee_mcmc_dndz_nsamples=30000.h5')
    parser.add_argument('--fiducial_file', type=str, default='mcmc_chains/cl_ee_fiducial_sacc.fits')
    parser.add_argument('--n_train', type=int, default=20000)
    parser.add_argument('--n_finetune', type=int, default=500)
    parser.add_argument('--n_walkers', type=int, default=76)
    parser.add_argument('--n_check', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=200000)
    parser.add_argument('--tau_factor', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)

