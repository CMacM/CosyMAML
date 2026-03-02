import numpy as np
import torch
import sacc
import h5py as h5
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

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

    filepath = '/home/b7009348/CosyMAML/cl_ee_IA_MCMC_task_samples.h5'
    with h5.File(filepath, 'r') as f:

        X_cosmo_train = f['cosmo_params'][:]
        y_cl_train = f['c_ells'][:]

        A_ia = f['A_ia_samples'][:]
        eta = f['eta_samples'][:]

    print('Using A_IA:{}, eta:{}'.format(np.mean(A_ia), np.mean(eta)))
    print('Train cosmologies shape:', X_cosmo_train.shape)
    print('Train C_ell shape:', y_cl_train.shape)

    # set device for all tensors
    device = args.device

    args.n_finetune = 100

    # Split test data into fine-tuning and evaluation sets
    if args.n_finetune < X_cosmo_train.shape[0]:
        X_finetune, _, y_finetune, _ = train_test_split(X_cosmo_train, y_cl_train, train_size=args.n_finetune, random_state=14)
    else:
        X_finetune = X_cosmo_train
        y_finetune = y_cl_train

    # convert to torch tensors
    X_finetune = torch.tensor(X_finetune, dtype=torch.float32).to(device)
    y_finetune = torch.tensor(y_finetune, dtype=torch.float32).to(device)

    # Scale the data
    y_finetune = torch.log(y_finetune)

    ScalerY = training.TorchStandardScaler()
    y_finetune = ScalerY.fit_transform(y_finetune)

    ScalerX = training.TorchStandardScaler()
    X_finetune = ScalerX.fit_transform(X_finetune)

    finetune_dataset = TensorDataset(X_finetune, y_finetune)
    finetune_loader = DataLoader(finetune_dataset, batch_size=args.n_finetune, shuffle=True)

    # Obtain dimensions of input and output
    in_size = X_finetune.shape[1]
    out_size = y_finetune.shape[1]

    ##### MAML MODEL #####

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
        outer_lr=0.005,
        inner_lr=0.001,
        loss_fn=torch.nn.MSELoss,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        seed=masterseed,
        device=device
    )

    # Load metalearner weights
    metalearner.model.load_state_dict(
        torch.load(
            'weights_ia_maml_20tasks_500samples_5batch.pth'
        )
    )

    # Finetune the MAML model to get new weights
    task_weights, _ = metalearner.finetune(
        X_finetune,
        y_finetune,
        adapt_steps=args.n_epochs,
        use_new_adam=True
    )
    print('MAML finetuning complete.')

    #### STANDARD MODEL #####

    # Load standard model
    pretrain_model = models.FastWeightCNN(
        input_size=in_size,
        latent_dim=(16,16),
        output_size=out_size,
        dropout_rate=0.2
    )

    pretrain_model.load_state_dict(
        torch.load(
            'weights_ia_standard_10000samples.pth'
        )
    )

    # Finetune the standard model to get new weights
    pretrain_model.to(device)
    _ = training.finetune_pretrained(
        model = pretrain_model,
        n_epochs=args.n_epochs,
        train_loader = finetune_loader,
        device=device
    )
    print('Standard model finetuning complete.')

    ##### FRESH MODEL #####

    # Load fresh model
    fresh_model = models.FastWeightCNN(
        input_size=in_size,
        latent_dim=(16,16),
        output_size=out_size,
        dropout_rate=0.2
    )

    fresh_model.to(device)
    _ = training.finetune_pretrained(
        model = fresh_model,
        n_epochs=args.n_epochs,
        train_loader = finetune_loader,
        device=device
    )
    print('Fresh model finetuning complete.')

    # Load in fiducial data
    sacc_path = os.path.join(args.mcmc_dir, 'cl_ee_IA_fiducial_sacc.fits')
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

    inv_cov = np.linalg.inv(cov_full)
    data_vector = np.concatenate(c_ells)

    # Initialize the walkers
    nwalkers = args.n_walkers
    ndim = len(theta)

    if nwalkers*2 < ndim:
        raise ValueError("Number of walkers must be greater than half the number of parameters")

    # Define the priors
    priors = [
        (0.17, 0.4), # Omega_c
        (0.03, 0.07), # Omega_b
        (0.4, 1.1), # h
        (0.65, 1.0), # sigma8
        (0.8, 1.1) # n_s
    ]

    # 10% spread around the known good point
    spreads = 0.1 * np.array(theta)
    pos = [theta + spreads * np.random.randn(ndim) for _ in range(nwalkers)]
    pos = np.array(pos)

    # # Construct a hook class to the MAML model
    # MAMLHook = mcmc.EmulatorHook(
    #     metalearner.model,
    #     ScalerX,
    #     ScalerY,
    #     weights=task_weights,
    #     device=device
    # )
    # # Define backend file for emcee
    # maml_backend = os.path.join(args.mcmc_dir, 
    #                             'cl_ee_IA_{}walkers_{}tau_MAML_chain.h5'
    #                             .format(nwalkers, args.tau_factor)
    #                         )

    # # Run the emcee sampler
    # print('Running emcee with MAML emulator...')
    # start = time()
    # mcmc.run_emcee(
    #     ClHook=MAMLHook,
    #     backend_file=maml_backend,
    #     ndim=ndim,
    #     priors=priors,
    #     pos=pos,
    #     data_vector=data_vector,
    #     inv_cov=inv_cov,
    #     nwalkers=nwalkers,
    #     n_check=args.n_check,
    #     max_iter=args.max_iter,
    #     tau_factor=args.tau_factor,
    #     clobber=True,
    #     progress=True
    # )
    # maml_mcmc_time = time() - start

    # # Print out the MCMC times in minutes
    # print('MAML model MCMC time:', maml_mcmc_time/60)

    # Construct a hook class to the standard model
    PretrainHook = mcmc.EmulatorHook(
        pretrain_model,
        ScalerX,
        ScalerY,
        device=device
    )

    # Define backend file for emcee
    pretrain_backend = os.path.join(args.mcmc_dir, 
                                   'cl_ee_IA_{}walkers_{}tau_pretrain_chain.h5'
                                   .format(nwalkers, args.tau_factor)
                                )
    
    # Run the emcee sampler
    print('Running emcee with pretrained emulator...')
    start = time()
    mcmc.run_emcee(
        ClHook=PretrainHook,
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

    # Print out the MCMC times in minutes
    print('Pretrained model MCMC time:', pretrain_mcmc_time/60)

    # Construct a hook class to the fresh model
    FreshHook = mcmc.EmulatorHook(
        fresh_model,
        ScalerX,
        ScalerY,
        device=device
    )

    # Define backend file for emcee
    fresh_backend = os.path.join(args.mcmc_dir, 
                                'cl_ee_IA_{}walkers_{}tau_fresh_chain.h5'
                                .format(nwalkers, args.tau_factor)
                            )
    
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

    # Print out the MCMC times in minutes
    print('Fresh model MCMC time:', fresh_mcmc_time/60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mcmc_dir', type=str, default='mcmc_chains', help='Directory to save MCMC output')
    parser.add_argument('--n_finetune', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_walkers', type=int, default=76)
    parser.add_argument('--n_check', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tau_factor', type=float, default=100)
    args = parser.parse_args()
    main(args)