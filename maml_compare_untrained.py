import numpy as np
import h5py as h5
import torch
from torch.utils.data import DataLoader
from IPython.display import clear_output

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

def main(args):
    
    # Define test parameters
    n_samples = np.linspace(
        args.n_samples_start, args.n_samples_end, args.n_samples_n
    ).astype(int)

    # Split first word of datafile to get dndz type
    dndz_type = args.datafile.split('_')[0].split('/')[-1]
    print(f'Running for dndz type: {dndz_type}')

    # Set up empty arrays to store results
    fresh_times = np.zeros((args.smooth, len(n_samples)))
    fresh_mapes = np.zeros((args.smooth, len(n_samples)))
    fresh_frates = np.zeros((args.smooth, len(n_samples)))
    maml_times = np.zeros((args.smooth, len(n_samples)))
    maml_mapes = np.zeros((args.smooth, len(n_samples)))
    maml_frates = np.zeros((args.smooth, len(n_samples)))
    seeds = np.zeros((args.smooth, len(n_samples)))

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

    # Loop over the number of samples
    for i, samples in enumerate(n_samples):
        # loop over the number of smoothings
        for j in range(args.smooth):
            clear_output(wait=True)
            print(f'Running {j+1}/{args.smooth} for {samples} samples')
            seed = np.random.randint(0, 1000)

            # Load the training data
            train_data, test_loader, X_val, y_val, ScalerY, _ = training.load_train_test_val(
                filepath=args.datafile, n_train=samples, n_val=4000, n_test=6000, seed=seed
            )

            # Train a fresh emulator
            train_loader = DataLoader(train_data, batch_size=5000, shuffle=False)
            fresh_model, fresh_time = training.train_standard_emulator(train_loader, X_val, y_val)

            # Sub-select 500 samples from the training set:
            X_train, y_train = train_data[:500]
            start = time()
            task_weights, _ = metalearner.finetune(
                X_train, y_train, adapt_steps=64, use_new_adam=True
            )
            maml_time = time() - start

            # test models on test set
            maml_mape, maml_frate = training.test_model(
                maml_model, test_loader, ScalerY, weights=task_weights
            )

            fresh_mape, fresh_frate = training.test_model(
                fresh_model, test_loader, ScalerY
            )

            # Save results
            fresh_times[j, i] = fresh_time
            fresh_mapes[j, i] = np.mean(fresh_mape)
            fresh_frates[j, i] = fresh_frate
            maml_times[j, i] = maml_time
            maml_mapes[j, i] = np.mean(maml_mape)
            maml_frates[j, i] = maml_frate
            seeds[j, i] = seed

    # Write results to file
    fname = f'{dndz_type}_maml_compare_untrained_nsamps={args.n_samples_n}_nsmooth={args.smooth}.h5'
    with h5.File(fname, 'w') as f:
        f.create_dataset('n_samples', data=n_samples)
        f.create_dataset('fresh_times', data=fresh_times)
        f.create_dataset('fresh_mapes', data=fresh_mapes)
        f.create_dataset('fresh_frates', data=fresh_frates)
        f.create_dataset('maml_times', data=maml_times)
        f.create_dataset('maml_mapes', data=maml_mapes)
        f.create_dataset('maml_frates', data=maml_frates)
        f.create_dataset('seeds', data=seeds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, default='data/mcmc_dndz_nsamples=30000.h5')
    parser.add_argument('--n_samples_start', type=int, default=500)
    parser.add_argument('--n_samples_end', type=int, default=20000)
    parser.add_argument('--n_samples_n', type=int, default=40)
    parser.add_argument('--smooth', type=int, default=20)
    args = parser.parse_args()
    main(args)


