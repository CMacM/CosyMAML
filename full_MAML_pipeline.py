import numpy as np
import h5py as h5
import torch
import os
import sacc
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from time import time

import src.training as training
import src.models as models
import src.mcmc as mcmc

from importlib import reload
reload(training)
reload(models)
reload(mcmc)

import argparse

def main(args):

    device = args.device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_inds = np.random.choice(200, args.sn_tasks+1, replace=False)
    sample_inds = np.random.choice(5000, args.n_samples, replace=False)
    # sort the indices
    task_inds.sort()

    with h5.File(args.trainfile, 'r') as f:
        X_train = f['X_train'][task_inds[:-1]]
        y_train = f['y_train'][task_inds[:-1]]
        X_val = f['X_train'][task_inds[-1]]
        y_val = f['y_train'][task_inds[-1]]

    X_train = X_train[:,sample_inds,:]
    y_train = y_train[:,sample_inds,:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    print(X_train.shape, y_train.shape)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_val, y_val, train_size=100, random_state=args.seed)

    X_val_train = torch.tensor(X_val_train, dtype=torch.float32).to(device)
    y_val_train = torch.tensor(y_val_train, dtype=torch.float32).to(device)
    X_val_test = torch.tensor(X_val_test, dtype=torch.float32).to(device)
    y_val_test = torch.tensor(y_val_test, dtype=torch.float32).to(device)

    # Scale
    y_val_train = torch.log(y_val_train)
    y_val_test = torch.log(y_val_test)

    ScalerX = training.TorchStandardScaler()
    ScalerY = training.TorchStandardScaler()

    X_val_train = ScalerX.fit_transform(X_val_train)
    y_val_train = ScalerY.fit_transform(y_val_train)
    X_val_test = ScalerX.transform(X_val_test)
    y_val_test = ScalerY.transform(y_val_test)

    in_size = X_train.shape[-1]
    out_size = y_train.shape[-1]

    # Load the model
    model = models.FastWeightCNN(
        input_size=in_size,
        latent_dim=(16,16),
        output_size=out_size,
        dropout_rate=0.2
    )

    # Initialise a MetaLearner
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

    # Train the model
    converged = False
    meta_losses = []
    val_losses = []
    epoch = 0
    best_val_loss = np.inf
    strike = 0
    force_stop = args.force_stop
    start = time()
    while not converged:
        epoch_loss = 0.0  # Accumulate loss for reporting purposes
        batch_count = 0   # Keep track of the number of batches
        metalearner.model.train()
        # Iterate through all batches in the dataloader
        for x_batch, y_batch in train_dataloader:
            x_batch = x_batch.to(metalearner.device)  # Move to GPU if available
            y_batch = y_batch.to(metalearner.device)

            # Take logarithm of y_batch
            y_batch = torch.log(y_batch)

            # Split the data into support and query sets
            x_spt, y_spt, x_qry, y_qry = training.support_query_split(
                x_batch, y_batch, spt_frac=0.6
            )

            # Perform one meta update step across the batch, scaling applied internally
            meta_loss = metalearner.meta_train(
                x_spt, y_spt, x_qry, y_qry, inner_steps=5
            )
            
            # Accumulate meta-loss for epoch statistics
            epoch_loss += meta_loss
            batch_count += 1

        # Compute average meta-loss for the epoch
        avg_meta_loss = epoch_loss / batch_count
        meta_losses.append(avg_meta_loss)
        epoch += 1

        # fine-tune the model on the validation set
        task_weights, _ = metalearner.finetune(X_val_train, y_val_train, adapt_steps=args.n_ft_epochs, use_new_adam=True)

        # Evaluate the model on the test set
        metalearner.model.eval()
        with torch.no_grad():
            y_pred = metalearner.model(X_val_test, params=task_weights)
            new_val_loss = metalearner.loss_fn(y_pred, y_val_test).item()
            val_losses.append(new_val_loss)

        # Check for convergence
        if best_val_loss - new_val_loss < 1e-5:
            strike += 1
            if strike >= 20 or epoch > force_stop:
                converged = True
                print('Validation loss has not improved for 20 epochs. Stopping training.')
        else:
            strike = 0

        # Update the best validation loss if necessary
        if new_val_loss < best_val_loss:
            best_val_loss = new_val_loss

        print(f'Epoch {epoch} - Val Loss: {new_val_loss} - Strike: {strike}')
        print('Took', time()-start, 'seconds')

    # Save loss history
    with h5.File(
        '/exafs/400NVX2/cmacmahon/weights/{}batch_{}samples_{}tasks_metalearner_losses.h5'.format(
            args.batch_size, args.n_samples, args.n_tasks
        ), 'w') as f:
        f.create_dataset('meta_losses', data=meta_losses)
        f.create_dataset('val_losses', data=val_losses)

    # Save meta weights
    torch.save(
        metalearner.model.state_dict(),
        '/exafs/400NVX2/cmacmahon/weights/{}batch_{}samples_{}tasks_metalearner_weights.pt'.format(
            args.batch_size, args.n_samples, args.n_tasks
        )
    )

    # Load metaleraner weights
    metalearner.model.load_state_dict(
        torch.load(
            '/exafs/400NVX2/cmacmahon/weights/{}batch_{}samples_{}tasks_metalearner_weights.pt'.format(
                args.batch_size, args.n_samples, args.n_tasks
            )
        )
    )

    train_data, test_data, X_val, y_val, ScalerY, ScalerX = training.load_train_test_val(
        filepath=args.mcmc_train_file, n_train=args.n_finetune, n_val=args.n_val, n_test=None, seed=args.seed,
        device=device
    )
    X_train, y_train = train_data[:]
    del test_data # we don't need this here

    # Finetune the MAML model to get new weights
    task_weights, _ = metalearner.finetune(
        X_train,
        y_train,
        adapt_steps=args.n_ft_epochs,
        use_new_adam=True
    )

    # Load in fiducial data
    sacc_path = '/exafs/400NVX2/cmacmahon/spectra_data/cl_ee_fiducial_sacc.fits'
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
    maml_backend = f'/exafs/400NVX2/cmacmahon/chains/{nwalkers}_maml_emulator_mcmc_samples.h5'

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
        progress=False
    )
    maml_mcmc_time = time() - start

    # Print out the MCMC times in minutes
    print('MAML model MCMC time:', maml_mcmc_time/60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--trainfile', type=str, default='/exafs/400NVX2/cmacmahon/weights/cl_ee_200tasks_5000samples_seed456.h5')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_tasks', type=int, default=20)
    parser.add_argument('--force_stop', type=int, default=100)
    parser.add_argument('--n_ft_epochs', type=int, default=64)
    parser.add_argument('--n_finetune', type=int, default=100)
    parser.add_argument('--n_check', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--n_walkers', type=int, default=76)
    parser.add_argument('--seed', type=int, default=14)
    parser.add_argument('--tau_factor', type=float, default=50)





