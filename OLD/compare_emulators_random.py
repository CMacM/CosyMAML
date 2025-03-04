import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import torch
import argparse
from IPython.display import clear_output
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from time import time
import src.training as training
import src.models as models

from importlib import reload
reload(training)
reload(models)

device = 'cuda'

def load_maml_data(batch_size, n_tasks, n_samples, seed):

    rng = np.random.RandomState(seed)
    task_inds = rng.choice(1000, n_tasks+1, replace=False)
    sample_inds = rng.choice(5000, n_samples, replace=False)
    # sort the indices
    task_inds.sort()

    # Load the data and construct a dataloader
    filepath = 'data/1000tasks_5000samples_14seed.h5'
    with h5.File(filepath, 'r') as f:
        X_train = f['X_train'][task_inds]
        y_train = f['y_train'][task_inds]
        X_val = f['X_train'][task_inds[-1]]
        y_val = f['y_train'][task_inds[-1]]

    X_train = X_train[:,sample_inds,:]
    y_train = y_train[:,sample_inds,:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(
        X_val, y_val, train_size=100, test_size=4000, random_state=seed
    )

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

    return train_dataloader, X_val_train, y_val_train, X_val_test, y_val_test

def load_standard_data(batch_size, train_size, seed):

    filepath = '42_dndz_nsamples=24000.h5'
    with h5.File(filepath, 'r') as f:
        X = f['hypercube'][:]
        y = f['c_ells'][:]

    # Take log of y
    y_log = np.log(y)

    # Split into test training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, train_size=train_size, random_state=seed
    )

    # Send data to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    #Scale the data
    ScalerX = training.TorchStandardScaler()
    X_train = ScalerX.fit_transform(X_train)
    X_test = ScalerX.transform(X_test)

    ScalerY = training.TorchStandardScaler()
    y_train = ScalerY.fit_transform(y_train)
    y_test = ScalerY.transform(y_test)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loader, X_test, y_test

def train_maml_model(train_dataloader, X_val_train, y_val_train, X_val_test, y_val_test, metalearner):

    # Train the model
    converged = False

    # Lists to store the losses
    meta_losses = []
    val_losses = []

    # Initialise convergence criteria
    epoch = 0
    best_val_loss = np.inf
    strike = 0

    # Loop until convergence
    while not converged:
        clear_output(wait=True)
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
                x_batch, y_batch, spt_frac=0.6, shuffle=False
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
        task_weights, _ = metalearner.finetune(X_val_train, y_val_train, adapt_steps=32, use_new_adam=True)

        # Evaluate the model on the test set
        metalearner.model.eval()
        with torch.no_grad():
            y_pred = metalearner.model(X_val_test, params=task_weights)
            new_val_loss = metalearner.loss_fn(y_pred, y_val_test).item()
            val_losses.append(new_val_loss)

        if best_val_loss - new_val_loss < 1e-4:
            strike += 1
            if strike >= 20:
                converged = True
                print('Validation loss has not improved for 20 epochs. Stopping training.')
        else:
            strike = 0

        # Update the best validation loss if necessary
        if new_val_loss < best_val_loss:
            best_val_loss = new_val_loss

        print(f'Epoch {epoch} - Val Loss: {new_val_loss} - Strike: {strike}')

    return meta_losses, val_losses

def train_standard_model(train_loader, X_test, y_test, model, optimizer):
    
    loss_fn = torch.nn.MSELoss()

    train_losses = []
    val_losses = []

    converged = False
    best_val_loss = np.inf
    strike = 0
    epoch = 0

    while not converged:
        clear_output(wait=True)
        epoch_loss = 0
        batch_count = 0
        model.train()
        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        # Check for convergence
        epoch += 1
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            val_loss = loss_fn(y_pred, y_test).item()
            val_losses.append(val_loss)

        if best_val_loss - val_loss < 1e-4:
            strike += 1
            if strike > 20:
                converged = True
                print('Validation loss has not improved for 20 epochs. Converged.')
        else:
            strike = 0
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        avg_epoch_loss = epoch_loss / batch_count
        train_losses.append(avg_epoch_loss)
        print(f'Epoch {epoch} - avg loss: {avg_epoch_loss} - Strike: {strike}')

    return train_losses, val_losses

def load_test_data(train_size, seed):

    filepath = 'mcmc_dndz_nsamples=20000.h5'
    with h5.File(filepath, 'r') as f:
        X = f['hypercube'][:]
        y = f['c_ells'][:]

    # Take log of y
    y_log = np.log(y)

    # Split into test training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, train_size=train_size, random_state=seed
    )

    # Send data to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    #Scale the data
    ScalerX = training.TorchStandardScaler()
    X_train = ScalerX.fit_transform(X_train)
    X_test = ScalerX.transform(X_test)

    ScalerY = training.TorchStandardScaler()
    y_train = ScalerY.fit_transform(y_train)

    # test on the test set in batches
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=2048, shuffle=True)

    return X_train, y_train, test_loader, ScalerY

def test_maml_model(metalearner, adapt_steps, X_train, y_train, test_loader, ScalerY):

    # Perform finetuning
    task_weights, _ = metalearner.finetune(X_train, y_train, adapt_steps=adapt_steps)

    ############# MAML TRAINING #############
    # Construct empty tensor to store 
    metalearner.model.eval()
    y_pred = torch.tensor([]).to(device)
    y_test = torch.tensor([]).to(device)
    for X_batch, y_batch in test_loader:
        with torch.no_grad(): # don't compute gradients during inference
            y_pred_batch = metalearner.model(X_batch, params=task_weights)
            y_pred = torch.cat((y_pred, y_pred_batch), dim=0)
            y_test = torch.cat((y_test, y_batch), dim=0)

    # Inverse transform the data
    y_pred = ScalerY.inverse_transform(y_pred)

    y_pred_np = y_pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # Exponentiate the data
    y_pred_np = np.exp(y_pred_np)
    y_test_np = np.exp(y_test_np)

    # Compute mean absolute percentage error along the test set
    apes = np.abs((y_test_np - y_pred_np) / y_test_np) * 100
    ell_ape = np.mean(apes, axis=1)
    maml_mape = np.mean(np.mean(apes, axis=0))
    maml_frate = len(ell_ape[ell_ape > 5]) / len(ell_ape)

    return maml_mape, maml_frate

def test_standard_model(model, adapt_steps, X_train, y_train, test_loader, ScalerY):

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(adapt_steps):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

    ############# STANDARD TRAINING #############
    # Construct empty tensor to store predictions
    model.eval()
    y_pred = torch.tensor([]).to(device)
    y_test = torch.tensor([]).to(device)
    for X_batch, y_batch in test_loader:
        with torch.no_grad(): # don't compute gradients during inference
            y_pred_batch = model(X_batch)
            y_pred = torch.cat((y_pred, y_pred_batch), dim=0)
            y_test = torch.cat((y_test, y_batch), dim=0)

    # Inverse transform the data
    y_pred = ScalerY.inverse_transform(y_pred)

    y_pred_np = y_pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # Exponentiate the data
    y_pred_np = np.exp(y_pred_np)
    y_test_np = np.exp(y_test_np)

    # Compute mean absolute percentage error along the test set
    apes = np.abs((y_test_np - y_pred_np) / y_test_np) * 100
    ell_ape = np.mean(apes, axis=1)
    standard_mape = np.mean(np.mean(apes, axis=0))
    standard_frate = len(ell_ape[ell_ape > 5]) / len(ell_ape)

    return standard_mape, standard_frate

def main(args):
    seeds = np.random.randint(0, 100000, size=args.n_runs)

    maml_mapes = []
    maml_frates = []
    standard_mapes = []
    standard_frates = []
    for seed in seeds:
        print(f'Running seed {seed}')
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Load MAML training data
        train_dataloader, X_val_train, y_val_train, X_val_test, y_val_test = load_maml_data(
            batch_size=5, n_tasks=20, n_samples=1000, seed=seed
        )

        # Define the model 
        maml_model = models.FastWeightCNN(
            input_size=10,
            latent_dim=(16,16),
            output_size=750,
            dropout_rate=0.2
        )

        # Initialise a MetaLearner
        metalearner = training.MetaLearner(
            model=maml_model,
            outer_lr=0.01,
            inner_lr=0.001,
            loss_fn=torch.nn.MSELoss,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            seed=seed,
            device=device
        )
        print('Training MAML model...')
        start = time()
        # Train the model
        _, _ = train_maml_model(
            train_dataloader, X_val_train, y_val_train, X_val_test, y_val_test, metalearner
        )
        maml_training_time = time() - start
        # Load standard training data
        train_loader, X_test, y_test = load_standard_data(
            batch_size=5000, train_size=20000, seed=seed
        )

        # Define the model
        model = models.FastWeightCNN(
            input_size=10,
            latent_dim=(16,16),
            output_size=750,
            dropout_rate=0.2
        ).to(device)

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print('Training standard model...')
        start = time()
        # Train the model
        _, _ = train_standard_model(
            train_loader, X_test, y_test, model, optimizer
        )
        standard_training_time = time() - start
        print('Testing models...')
        # Load test data
        X_train, y_train, test_loader, ScalerY = load_test_data(
            train_size=args.finetune_samples, seed=seed
        )

        # Test the MAML model
        start = time()
        maml_mape, maml_frate = test_maml_model(
            metalearner, args.finetune_steps, X_train, y_train, test_loader, ScalerY
        )
        maml_testing_time = time() - start
        # Test the standard model
        start = time()
        standard_mape, standard_frate = test_standard_model(
            model, args.finetune_steps, X_train, y_train, test_loader, ScalerY
        )
        standard_testing_time = time() - start
        # Append the results to the lists
        maml_mapes.append(maml_mape)
        maml_frates.append(maml_frate)
        standard_mapes.append(standard_mape)
        standard_frates.append(standard_frate)
        print(f'MAML MAPE: {maml_mape} - Standard MAPE: {standard_mape}')
        print(f'MAML FRATE: {maml_frate} - Standard FRATE: {standard_frate}')

    # Save the results
    filename = f"compare_emulators_{args.finetune_samples}samples_{args.finetune_steps}steps.h5"
    with h5.File(filename, 'w') as f:
        f.create_dataset('maml_mapes', data=maml_mapes)
        f.create_dataset('maml_frates', data=maml_frates)
        f.create_dataset('standard_mapes', data=standard_mapes)
        f.create_dataset('standard_frates', data=standard_frates)
        f.create_dataset('seeds', data=seeds)
        f.create_dataset('maml_training_time', data=maml_training_time)
        f.create_dataset('standard_training_time', data=standard_training_time)
        f.create_dataset('maml_testing_time', data=maml_testing_time)
        f.create_dataset('standard_testing_time', data=standard_testing_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--finetune_samples', type=int, default=100)
    parser.add_argument('--finetune_steps', type=int, default=32)
    args = parser.parse_args()
    main(args)




    