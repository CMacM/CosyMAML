import numpy as np
import h5py as h5
import torch
from torch.utils.data import DataLoader
from IPython.display import clear_output
from copy import deepcopy

import src.training as training
import src.models as models

from time import time
from importlib import reload
reload(training)
reload(models)

import argparse

# Set up random seed for reproducibility
masterseed = 14
np.random.seed(masterseed)
torch.manual_seed(masterseed)
torch.cuda.manual_seed(masterseed)

# set device for all tensors
device = 'cuda'

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

    original_model = deepcopy(pretrain_model.state_dict())
    
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

    n_finetune = np.linspace(args.ft_min, args.ft_max, args.nft_sizes).astype(int)
    smooth = args.smooth

    pretrain_mapes = np.zeros((len(n_finetune), smooth))
    pretrain_frates = np.zeros((len(n_finetune), smooth))
    maml_mapes = np.zeros((len(n_finetune), smooth))
    maml_frates = np.zeros((len(n_finetune), smooth))
    pretrain_times = np.zeros((len(n_finetune), smooth))
    maml_times = np.zeros((len(n_finetune), smooth))
    seeds = np.zeros((len(n_finetune), smooth))

    for i, samples in enumerate(n_finetune):
        for j in range(smooth):

            seed = np.random.randint(10000)

            train_data, test_data, X_val, y_val, ScalerY, _ = training.load_train_test_val(
                filepath=args.finetune_file,
                n_train=samples,
                n_val=None,
                n_test=20000,
                seed=seed,
                device=device
            )
            train_loader = DataLoader(train_data, batch_size=5000, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=5000, shuffle=False)
            
            # Fine-tune the single-task emulator
            start = time()
            pretrain_model.load_state_dict(original_model)
            _ = training.finetune_pretrained(
                model = pretrain_model,
                #x_val=X_val,
                #y_val=y_val,
                n_epochs=32,
                train_loader = train_loader,
                device=device
            )
            pretrain_times[i,j] = time() - start

            # Fine-tune the MAML model
            start = time()
            X_train, y_train = train_data[:]
            task_weights, _ = metalearner.finetune(
                X_train, 
                y_train, 
                #x_val=X_val,
                #y_val=y_val,
                adapt_steps=32,
                use_new_adam=True
            )
            maml_times[i,j] = time() - start

            # Test the models
            pretrain_mape, pretrain_frate = training.test_model(
                pretrain_model, test_loader, ScalerY
            )

            maml_mape, maml_frate = training.test_model(
                maml_model, test_loader, ScalerY, weights=task_weights
            )

            # Save results
            pretrain_mapes[i,j] = np.mean(pretrain_mape)
            pretrain_frates[i,j] = pretrain_frate
            maml_mapes[i,j] = np.mean(maml_mape)
            maml_frates[i,j] = maml_frate

    # Write results to file
    fname = f'maml_compare_pretrained_nsmooth={args.smooth}.h5'
    with h5.File(fname, 'w') as f:
        f.create_dataset('n_finetune', data=n_finetune)
        f.create_dataset('pretrain_times', data=pretrain_times)
        f.create_dataset('pretrain_mapes', data=pretrain_mapes)
        f.create_dataset('pretrain_frates', data=pretrain_frates)
        f.create_dataset('maml_times', data=maml_times)
        f.create_dataset('maml_mapes', data=maml_mapes)
        f.create_dataset('maml_frates', data=maml_frates)
        f.create_dataset('seeds', data=seeds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare MAML vs Pretrained Emulators')
    parser.add_argument('--pretrain_file', type=str, default='spectra_data/cl_ee_27_dndz_nsamples=30000.h5', help='Path to pretraining data')
    parser.add_argument('--finetune_file', type=str, default='spectra_data/cl_ee_mcmc_dndz_nsamples=30000.h5', help='Path to finetuning data')
    parser.add_argument('--n_train', type=int, default=20000, help='Number of samples for pretraining')
    parser.add_argument('--ft_min', type=int, default=100, help='Minimum number of samples for finetuning')
    parser.add_argument('--ft_max', type=int, default=1000, help='Maximum number of samples for finetuning')
    parser.add_argument('--nft_sizes', type=int, default=50, help='Number of different finetuning sample sizes')
    parser.add_argument('--smooth', type=int, default=20, help='Number of smoothings to perform')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    args = parser.parse_args()
    main(args)
