import numpy as np
import torch
import sacc
import os
import sys
from time import time

import src.models as models
import src.training as training
import src.mcmc as mcmc
import argparse

from importlib import reload
reload(training)
reload(models)

def main(args):
    device = args.device
    print(f'Using device: {device}')

    # Check if the device is a GPU
    if device == 'cuda':
        torch.cuda.set_device(args.chain_id)

    # Use chain ID to seed the random number generator
    # This ensures that each chain starts with a different random seed
    seed = 14 + 14*args.chain_id

    np.random.seed(seed)
    torch.manual_seed(seed)

    print('Finetuning model to MCMC N(z) task...')

    # Load in data for finetuning the MAML model
    #### I/O KEY READ POINT ####
    trainfile_path = os.path.join(args.data_dir, args.mcmc_trainfile)
    train_data, _, ScalerY, ScalerX = training.load_train_test_val(
        filepath=trainfile_path, n_train=args.n_finetune, n_val=None, n_test=None, seed=seed,
        device=device
    )
    X_train, y_train = train_data[:]
    in_size = X_train.shape[1]
    out_size = y_train.shape[1]

        # Construct model architecture
    model = models.FastWeightCNN(
        input_size=in_size,
        latent_dim=(16,16),
        output_size=out_size,
        dropout_rate=0.2
    )

    # Initialise a MetaLearner for training
    metalearner = training.MetaLearner(
        model=model,
        outer_lr=0.01,
        inner_lr=0.001,
        loss_fn=torch.nn.MSELoss,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        seed=seed,
        device=device
    )

    # Load metaleraner weights
    #### I/O KEY READ POINT ####
    weights_path = os.path.join(args.data_dir,'model_weights/5batch_500samples_20tasks_metalearner_weights.pt')

    # Load the weights into the model
    metalearner.model.load_state_dict(torch.load(weights_path))

    # Finetune the MAML model to get new weights
    task_weights, _ = metalearner.finetune(
        X_train,
        y_train,
        adapt_steps=args.n_ft_epochs,
        use_new_adam=True
    )

    # Load in fiducial data
    #### I/O KEY READ POINT ####
    sacc_path = os.path.join(args.data_dir, 'spectra_data/cl_ee_fiducial_sacc.fits')
    S = sacc.Sacc.load_fits(sacc_path)

    # Define tracer combs
    tracer_combs = S.get_tracer_combinations()

    # Extract C_ell from SACC
    c_ells = []
    for comb in tracer_combs:
        _, cell = S.get_ell_cl(
            data_type='cl_ee',
            tracer1=comb[0],
            tracer2=comb[1],
            return_cov=False
        )
        c_ells.append(cell)

    # Get the covariance matrix from SACC
    cov_full = S.covariance.covmat
    # Check covariance matrix is not singular
    if np.linalg.cond(cov_full) > 1 / np.finfo(cov_full.dtype).eps:
        raise ValueError("Covariance matrix is singular or nearly singular.")

    # Specify fiducial cosmology and number of z-bins
    theta = [0.27, 0.045, 0.67, 0.83, 0.96]
    n_bins = len(S.tracers)

    # Comment out if not using shifts
    for i in range(n_bins):
        theta.append(0.0)

    # Prepare data and cov for MCMC
    inv_cov = np.linalg.inv(cov_full)
    inv_cov = torch.tensor(inv_cov, dtype=torch.float32, device=device)

    data_vector = np.concatenate(c_ells)
    data_vector = torch.tensor(data_vector, dtype=torch.float32, device=device)

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

    # Define spreads as before
    spreads = 0.1 * np.array(theta)
    for i in range(n_bins):
        spreads[n_bins + i] += 4e-4

    pos = [theta + spreads * np.random.randn(ndim) for _ in range(args.nwalkers)]
    pos = np.array(pos)

    # Construct a hook to call the MAML model during MCMC
    MAMLHook = mcmc.EmulatorHook(
        metalearner.model,
        ScalerX,
        ScalerY,
        weights=task_weights, #Weights from finetuning
        device=device
    )

    # Define backend file for emcee to store samples
    ### I/O KEY WRITE POINT ### 
    # Need to understand how the emcee backend works with writes
    backend_file = os.path.join(args.data_dir, f'chains/chain_{args.chain_id}.h5')
    print('Writing to backend file:', backend_file)

    # Run the emcee sampler, pos needs to be an array of shape (nwalkers, ndim)
    start = time()
    mcmc.run_emcee(
        ClHook=MAMLHook,
        backend_file=backend_file,
        ndim=ndim,
        priors=priors,
        pos=pos,
        data_vector=data_vector,
        inv_cov=inv_cov,
        nwalkers=args.nwalkers,
        max_iter=args.max_iter,
        n_check=args.n_check,
        tau_factor=args.tau_factor,
        clobber=True,
        progress=args.progress
    )
    maml_mcmc_time = time() - start

    # Print out the MCMC times in minutes
    print(f'Chain {args.chain_id} converged in {maml_mcmc_time/60:.2f} minutes')

    # Exit the script
    print('Exiting script')
    sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dir', type=str, default='/exafs/400NVX2/cmacmahon/')
    parser.add_argument('--mcmc_trainfile', type=str, default='spectra_data/cl_ee_mcmc_dndz_nsamples=30000.h5')
    parser.add_argument('--n_finetune', type=int, default=100)
    parser.add_argument('--n_ft_epochs', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_iter', type=int, default=500000)
    parser.add_argument('--n_check', type=int, default=1000)
    parser.add_argument('--tau_factor', type=int, default=50)
    parser.add_argument('--nwalkers', type=int, default=128)
    parser.add_argument('--chain_id', type=int, default=0)
    parser.add_argument('--progress', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
