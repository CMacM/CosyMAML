import numpy as np
import os
import h5py as h5
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.profiler
from sklearn.model_selection import train_test_split
from datetime import datetime
from time import time

import src.training as training
import src.models as models

from importlib import reload
reload(training)
reload(models)

import argparse

def main(args):

    device = args.device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_inds = np.random.choice(200, args.n_tasks+1, replace=False)
    sample_inds = np.random.choice(5000, args.n_samples, replace=False)
    # sort the indices
    task_inds.sort()

    print('Loading training data...')
    # Load the training data and split into train and validation sets
    #### I/O KEY READ POINT ####
    # check if file is hdf5 or npz
    start_read = datetime.now()
    if args.trainfile.endswith('.npz'):
        with np.load(args.trainfile) as data:
            X_train = data['X_train'][task_inds[:-1]]
            y_train = data['y_train'][task_inds[:-1]]
            X_val = data['X_train'][task_inds[-1]]
            y_val = data['y_train'][task_inds[-1]]
    elif args.trainfile.endswith('.h5'):
        with h5.File(args.trainfile, 'r') as f:
            X_train = f['X_train'][task_inds[:-1]]
            y_train = f['y_train'][task_inds[:-1]]
            X_val = f['X_train'][task_inds[-1]]
            y_val = f['y_train'][task_inds[-1]]
    end_read = datetime.now()

    start_prep = datetime.now()
    # Slice number of shots we want to train with
    X_train = X_train[:,sample_inds,:]
    y_train = y_train[:,sample_inds,:]

    # Convert to torch tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Preapre as a torch dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Split the validation set into train and test sets
    # This is required because MAML invovles a small fine-tuning step
    # before testing on the validation data
    X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(
        X_val, y_val, train_size=args.n_finetune, random_state=args.seed
    )

    # Convert to torch tensors
    X_val_train = torch.tensor(X_val_train, dtype=torch.float32).to(device)
    y_val_train = torch.tensor(y_val_train, dtype=torch.float32).to(device)
    X_val_test = torch.tensor(X_val_test, dtype=torch.float32).to(device)
    y_val_test = torch.tensor(y_val_test, dtype=torch.float32).to(device)

    # Log transform helps stabilise the training
    y_val_train = torch.log(y_val_train)
    y_val_test = torch.log(y_val_test)

    # Prepare the scalers
    ScalerX = training.TorchStandardScaler()
    ScalerY = training.TorchStandardScaler()

    # Fit the scalers to the training data and transform
    X_val_train = ScalerX.fit_transform(X_val_train)
    y_val_train = ScalerY.fit_transform(y_val_train)
    X_val_test = ScalerX.transform(X_val_test)
    y_val_test = ScalerY.transform(y_val_test)

    # Get size of input and output
    in_size = X_train.shape[-1]
    out_size = y_train.shape[-1]
    end_prep = datetime.now()

    ### Hyperparams currently hardcoded to what was used in the paper ###

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

    # Initialise training variables
    converged = False
    meta_losses = []
    val_losses = []
    epoch = 0
    best_val_loss = np.inf
    strike = 0
    force_stop = args.force_stop

    # Train the model
    print('Training started...')
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
        task_weights, _ = metalearner.finetune(
                X_val_train, y_val_train, adapt_steps=args.n_ft_epochs, use_new_adam=True
            )

        # Evaluate the model on the test set
        metalearner.model.eval()
        with torch.no_grad():
            y_pred = metalearner.model(X_val_test, params=task_weights)
            new_val_loss = metalearner.loss_fn(y_pred, y_val_test).item()
            val_losses.append(new_val_loss)

        # Stop training if the maximum number of epochs is reached
        if epoch > force_stop:
            converged = True
            print('Epoch limit reached. Stopping training.')

        # Check for convergence based on validation loss
        if best_val_loss - new_val_loss < 1e-5:
            strike += 1
            # Stop training if validation loss has not improved for 20 epochs
            if strike >= 20:
                converged = True
                print('Validation loss has not improved for 20 epochs. Stopping training.')
        else:
            strike = 0

        # Update the best validation loss if necessary
        if new_val_loss < best_val_loss:
            best_val_loss = new_val_loss

        print(f'Epoch {epoch} - Val Loss: {new_val_loss} - Strike: {strike}')
    
    total_time = time()-start
    print(f'Training took {total_time/60} minutes, saving model and losses...')

    # Save loss history
    #### I/O KEY WRITE POINT ####
    loss_filename = os.path.join(
        args.model_dir, '{}batch_{}samples_{}tasks_metalearner_losses.h5'.format(
            args.batch_size, args.n_samples, args.n_tasks
        )
    )
    start_write_loss = datetime.now()
    with h5.File(loss_filename, 'w') as f:
        f.create_dataset('meta_losses', data=meta_losses)
        f.create_dataset('val_losses', data=val_losses)
        f.create_dataset('total_time', data=total_time)
    end_write_loss = datetime.now()

    # Save meta weights
    #### I/O KEY WRITE POINT ####
    weights_filename = os.path.join(
        args.model_dir, '{}batch_{}samples_{}tasks_metalearner_weights.pt'.format(
            args.batch_size, args.n_samples, args.n_tasks
        )
    )

    start_write_weights = datetime.now()
    torch.save(metalearner.model.state_dict() ,weights_filename)
    end_write_weights = datetime.now()

    # Write timing information to file
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    timing_filename = os.path.join(args.log_dir, f'{job_id}_timing.txt')
    with open(timing_filename, 'a') as f:
        f.write('start_read,end_read,start_prep,end_prep,start_write_loss,end_write_loss,start_write_weights,end_write_weights\n')
        f.write(f'{start_read},{end_read},{start_prep},{end_prep},{start_write_loss},{end_write_loss},{start_write_weights},{end_write_weights}\n')

if __name__ == '__main__':
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--trainfile', type=str, 
        default='/exafs/400NVX2/cmacmahon/spectra_data/cl_ee_200tasks_5000samples_seed456.h5'
    )
    parser.add_argument('--model_dir', type=str, default='/exafs/400NVX2/cmacmahon/weights')
    parser.add_argument('--log_dir', type=str, default='/exafs/400NVX2/cmacmahon/logs')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_tasks', type=int, default=20)
    parser.add_argument('--n_finetune', type=int, default=100)
    parser.add_argument('--n_ft_epochs', type=int, default=64)
    parser.add_argument('--force_stop', type=int, default=100)
    parser.add_argument('--seed', type=int, default=14)

    # Parse the command-line arguments and run the main function
    args = parser.parse_args()
    main(args)