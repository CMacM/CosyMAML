import numpy as np
import torch
import sacc
import os
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Finetuning model to MCMC N(z) task...')

    # Load in data for finetuning the MAML model
    #### I/O KEY READ POINT ####
    train_data, _, ScalerY, ScalerX = training.load_train_test_val(
        filepath=args.mcmc_trainfile, n_train=args.n_finetune, n_val=None, n_test=None, seed=args.seed,
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
        seed=14,
        device=device
    )

    # Load metaleraner weights
    #### I/O KEY READ POINT ####
    metalearner.model.load_state_dict(
        torch.load(
            '/exafs/400NVX2/cmacmahon/weights/{}batch_{}samples_{}tasks_metalearner_weights.pt'.format(
                args.batch_size, args.n_samples, args.n_tasks
            )
        )
    )

    # Finetune the MAML model to get new weights
    task_weights, _ = metalearner.finetune(
        X_train,
        y_train,
        adapt_steps=args.n_ft_epochs,
        use_new_adam=True
    )

    # Load in fiducial data
    #### I/O KEY READ POINT ####
    sacc_path = '/exafs/400NVX2/cmacmahon/spectra_data/cl_ee_fiducial_sacc.fits'
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

    # Specify fiducial cosmology and number of z-bins
    theta = [0.27, 0.045, 0.67, 0.83, 0.96]
    n_bins = len(S.tracers)

    # Comment out if not using shifts
    for i in range(n_bins):
        theta.append(0.0)

    # Prepare data and cov for MCMC
    inv_cov = np.linalg.inv(cov_full)
    data_vector = np.concatenate(c_ells)

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

    np.random.seed(42 + args.chain_id)  # Ensure different but reproducible initializations

    # Define spreads as before
    spreads = 0.1 * np.array(theta)
    for i in range(n_bins):
        spreads[n_bins + i] += 4e-4

    # Generate a single starting position for this chain
    pos = theta + spreads * np.random.randn(len(theta))

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
    data_dir = '/exafs/400NVX2/cmacmahon/chains/'
    backend_file = os.path.join(data_dir, f'chain_{args.chain_id}.h5')

    # Run the emcee sampler
    print('Running emcee with MAML emulator...')
    start = time()
    mcmc.run_batched_mh(
        ClHook=MAMLHook,
        backend_file=backend_file,
        ndim=ndim,
        priors=priors,
        initial_pos=pos,
        data_vector=data_vector,
        inv_cov=inv_cov,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        n_check=args.n_check,
        tau_factor=args.tau_factor,
        proposal_scale=args.proposal_scale
    )
    maml_mcmc_time = time() - start

    # Print out the MCMC times in minutes
    print(f'Chain converged after: {maml_mcmc_time/60:.2f} minutes')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mcmc_trainfile', type=str, 
        default='/exafs/400NVX2/cmacmahon/spectra_data/cl_ee_mcmc_dndz_nsamples=30000.h5'
    )
    parser.add_argument('--n_finetune', type=int, default=100)
    parser.add_argument('--n_ft_epochs', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_iter', type=int, default=50000)
    parser.add_argument('--n_check', type=int, default=1000)
    parser.add_argument('--tau_factor', type=int, default=50)
    parser.add_argument('--proposal_scale', type=float, default=0.01)
