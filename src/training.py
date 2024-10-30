import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import h5py as h5

# Redundant imports used by old code
# import matplotlib.pyplot as plt
# from torch import autograd as ag
# import copy
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

def support_query_split(x_train, y_train, spt_frac=None, n_spt=None, shuffle=True):
    '''
    Function to split the data into support and query sets.

    Args:
    - x_train: the input data for a single task.
    - y_train: the target data for a single task.
    - spt_frac: the fraction of the data to be used for support.
    - n_spt: the number of samples to be used for support.

    Returns:
    - x_spt: the support input data.
    - y_spt: the support target data.
    - x_qry: the query input data.
    - y_qry: the query target data.
    '''

    n_samples = x_train.shape[1]

    if spt_frac is None and n_spt is None:
        raise ValueError('Either spt_frac or n_spt must be provided.')
    elif spt_frac is not None and n_spt is not None:
        raise ValueError('Only one of spt_frac or n_spt should be provided.')

    # If spt_frac is provided, calculate n_spt from it
    if spt_frac is not None:
        n_spt = int(n_samples * spt_frac)

    # Shuffle indices and split into support and query sets
    if shuffle:
        permu = np.random.permutation(n_samples)
    else:
        permu = np.arange(n_samples)
    
    x_spt = x_train[:, permu[:n_spt], :]
    y_spt = y_train[:, permu[:n_spt], :]
    x_qry = x_train[:, permu[n_spt:], :]
    y_qry = y_train[:, permu[n_spt:], :]

    return x_spt, y_spt, x_qry, y_qry

class HDF5Dataset(Dataset):
    '''
        Construct a PyTorch dataset from an HDF5 file.

        Args:
        - file_path: the path to the HDF5 file.
    '''
    def __init__(self, file_path):
        self.file = file_path
        with h5.File(file_path, 'r') as f:
            self.len = f['X_train'].shape[0] # number of tasks
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        with h5.File(self.file, 'r') as f:
            X = f['X_train'][idx].astype('float32') # Specify precision
            y = f['y_train'][idx].astype('float32') # Torch expects float32
        return X, y

class TorchStandardScaler():
    '''
        Standardising Scaler class which can be used 
        directly with PyTorch tensors.
        
        Attributes:
            mean (torch.Tensor): Mean of the data.
            std (torch.Tensor): Standard deviation of the data.
        
        Methods:
            fit: Compute the mean and standard deviation of the data.
            transform: Transform the data using the computed mean and standard deviation.
            inverse_transform: Inverse transform the data using the computed mean and standard deviation.
            fit_transform: Fit to the data and transform the data.
    '''
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        self.mean = X.mean(0, keepdim=True)
        self.std = X.std(0, unbiased=False, keepdim=True)
        return self
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def inverse_transform(self, X):
        return X * self.std + self.mean
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MetaLearner():
    '''
        Base class to meta-train a model using First Order MAML. The class also uses
        a shared Adam optimizer for the inner and outer loops. Testing has shown this
        to be more stable than using separate optimizers.

        Args:
        - model: the model to be trained. It must have a forward method that accepts
                    an input tensor and a dictionary of parameters
        - outer_lr: the learning rate for the outer loop.
        - inner_lr: the learning rate for the inner loop.
        - loss_fn: the loss function to be used.
        - beta1: the beta1 parameter for the Adam optimizer.
        - beta2: the beta2 parameter for the Adam optimizer.
        - epsilon: the epsilon parameter for the Adam optimizer.
        - seed: the seed for reproducibility.
        - device: the device to run the model on.
    '''
    def __init__(self, model, outer_lr=0.01, inner_lr=0.001, loss_fn=nn.MSELoss, 
                 beta1=0.9, beta2=0.999, epsilon=1e-8, seed=None, device=None):
        # Define constants and parameters
        self.model = model
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.loss_fn = loss_fn()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Set seed for reproducibility if provided
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        # Set device if provided or check for GPU availability
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model.to(self.device) # Move model to device

        # Initialize Adam parameters
        self.adam_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.adam_params[name] = {
                    'm': torch.zeros_like(param).to(self.device),
                    'v': torch.zeros_like(param).to(self.device),
                    't': 0
                }

    def _inner_update(self, x, y, fast_weights, steps):
        '''
        Perform an inner update on the model.
        
        Args:
        - x: the input data.
        - y: the target data.
        - fast_weights: the model parameters as a dictionary.
        - steps: the number of steps to update the model.
        '''

        # Check for dropout layers and if present enable
        if hasattr(self.model, 'dropout'):
            self.model.train()

        for _ in range(steps):
            self.model.zero_grad()
            y_pred = self.model(x, params=fast_weights)
            loss_spt = self.loss_fn(y_pred, y)
            grad = torch.autograd.grad(loss_spt, fast_weights.values())

            # update fast weights using Adam optimizer
            for (name, param), grad in zip(fast_weights.items(), grad):
                adam_state = self.adam_params[name]
                adam_state['t'] += 1
                adam_state['m'] = self.beta1 * adam_state['m'] + (1 - self.beta1) * grad
                adam_state['v'] = self.beta2 * adam_state['v'] + (1 - self.beta2) * (grad ** 2)
                m_hat = adam_state['m'] / (1 - self.beta1 ** adam_state['t'])
                v_hat = adam_state['v'] / (1 - self.beta2 ** adam_state['t'])
                fast_weights[name] = param - self.inner_lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

        return fast_weights

    def meta_train(self, x_spt, y_spt, x_qry, y_qry, inner_steps=10):

        # Check size of batch provided
        batch_size = x_spt.shape[0]
        task_batch = self.rng.permutation(batch_size) # permute order of tasks
        meta_loss = 0
        for task in task_batch:
            x_spt_task = x_spt[task].to(self.device)
            y_spt_task = y_spt[task].to(self.device)
            x_qry_task = x_qry[task].to(self.device)
            y_qry_task = y_qry[task].to(self.device)

            # Apply scaling
            scaler_x = TorchStandardScaler()
            x_spt_task = scaler_x.fit_transform(x_spt_task)
            x_qry_task = scaler_x.transform(x_qry_task)

            scaler_y = TorchStandardScaler()
            y_spt_task = scaler_y.fit_transform(y_spt_task)
            y_qry_task = scaler_y.transform(y_qry_task)
        
            # Initialize fast weights for task specific updates
            fast_weights = {name: param.clone().requires_grad_(True) 
                            for name, param in self.model.named_parameters()}
            
            # Update fast weights using inner loop
            fast_weights = self._inner_update(
                x_spt_task, y_spt_task, fast_weights, inner_steps
            )

            # check for dropout layers and if present enable
            if hasattr(self.model, 'dropout'):
                self.model.train()
            
            # Zero gradients before training
            self.model.zero_grad()
            y_pred = self.model(x_qry_task, params=fast_weights)
            loss_qry = self.loss_fn(y_pred, y_qry_task)

            # Accumulate meta loss
            meta_loss += loss_qry

        # Average meta loss across the batch
        meta_loss /= batch_size

        # Compute gradients of meta loss and update meta parameters
        grad = torch.autograd.grad(meta_loss, self.model.parameters())
        # Update meta parameters using Adam optimizer
        for (name, param), grad in zip(self.model.named_parameters(), grad):
            adam_state = self.adam_params[name]
            adam_state['t'] += 1
            adam_state['m'] = self.beta1 * adam_state['m'] + (1 - self.beta1) * grad
            adam_state['v'] = self.beta2 * adam_state['v'] + (1 - self.beta2) * (grad ** 2)
            m_hat = adam_state['m'] / (1 - self.beta1 ** adam_state['t'])
            v_hat = adam_state['v'] / (1 - self.beta2 ** adam_state['t'])
            param.data -= self.outer_lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

        return meta_loss.detach().item()

    def finetune(self, x_spt, y_spt, adapt_steps):
        '''
        Fine-tune the model on the support data.

        Args:
        - x_spt: the support input data.
        - y_spt: the support target data.
        - adapt_steps: the number of steps to fine-tune the model.
        '''

        # Initialize fast weights for task specific updates
        fast_weights = {name: param.clone().requires_grad_(True) 
                        for name, param in self.model.named_parameters()}

        # Check for dropout layers and if present enable
        if hasattr(self.model, 'dropout'):
            self.model.train()

        # Use a separate Adam optimizer for fine-tuning
        finetune_adam_params = {}
        for name, param in fast_weights.items():
            finetune_adam_params[name] = {
                'm': torch.zeros_like(param).to(self.device),
                'v': torch.zeros_like(param).to(self.device),
                't': 0
            }

        # Update fast weights using inner loop
        for _ in range(adapt_steps):
            self.model.zero_grad()
            y_pred = self.model(x_spt, params=fast_weights)
            loss_spt = self.loss_fn(y_pred, y_spt)
            grad = torch.autograd.grad(loss_spt, fast_weights.values())

            # update fast weights using Adam optimizer
            for (name, param), grad in zip(fast_weights.items(), grad):
                adam_state = finetune_adam_params[name]
                adam_state['t'] += 1
                adam_state['m'] = self.beta1 * adam_state['m'] + (1 - self.beta1) * grad
                adam_state['v'] = self.beta2 * adam_state['v'] + (1 - self.beta2) * (grad ** 2)
                m_hat = adam_state['m'] / (1 - self.beta1 ** adam_state['t'])
                v_hat = adam_state['v'] / (1 - self.beta2 ** adam_state['t'])
                fast_weights[name] = param - self.inner_lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

        
        return fast_weights



#### OLD CODE TO BE KEPT FOR REFERENCE ####

# class MAML():
#     '''
#         Class for MAML implemntation of an angular power spectrum emulator.
#         SGD is currently hardcoded, need to provide different optimizers.
#     '''
#     def __init__(self, model, seed=14,
#                  ):
#         self.model = model
#         self.seed = seed
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)
#         self.rng = np.random.RandomState(seed)

#     # convert numpy array to torch tensor
#     def to_torch(self, arr):
#         return ag.Variable(torch.tensor(arr, dtype=torch.float32).to(self.device))
    
#     # send individual data batch to model for training step
#     def train_on_batch(self, x ,y, step, loss_fn):
#         x = self.to_torch(x)
#         y = self.to_torch(y)
#         self.model.zero_grad()
#         y_pred = self.model(x)
#         loss = loss_fn(y_pred, y)
#         loss.backward()
#         for param in self.model.parameters():
#             param.data -= step * param.grad.data

#     # obtain predictions from model
#     # Includes built in scaling 
#     def predict(self, x):
#         x = self.to_torch(x)
#         return self.model(x).detach().cpu().numpy()
    
#     # meta train the model
#     def meta_train(self, x_train, y_train, 
#                    inner_lr, outer_lr, loss_fn, 
#                    spt_frac, outer_epochs, inner_epochs,
#                    inner_decay=1e-4, outer_decay=1e-3, n_shots=5,
#                    rec_loss=True, plot_prog=False, scale=True):
        
#         # ascertaining the number of tasks and samples
#         n_tasks = y_train.shape[0]
#         n_samples = y_train.shape[1]
#         tasks = np.arange(n_tasks)
#         tasks = self.rng.permutation(tasks) # permute the order tasks are trained

#         # isolate a random task for plotting the progress of training
#         if plot_prog:
#             task_plot = self.rng.choice(tasks, size=1)
#             print('Plotting task {}'.format(task_plot))
#             tasks= np.delete(tasks, task_plot)
#             n_tasks = len(tasks)
#             rand_inds = self.rng.choice(n_samples, size=n_shots+1, replace=False)

#             # print('Training samples:', rand_inds[:-1])
#             # print('Test samples:', rand_inds[-1])

#             scaler_x_plot = StandardScaler()
#             scaler_y_plot = StandardScaler()

#             x_shot = x_train[task_plot, rand_inds[:-1]]
#             y_shot = y_train[task_plot, rand_inds[:-1]]

#             # Fit scaler on support data and transform both support and test data
#             x_shot = scaler_x_plot.fit_transform(x_shot)
#             x_test = x_train[task_plot, rand_inds[-1]]
#             x_test = scaler_x_plot.transform(x_test)

#             y_shot = scaler_y_plot.fit_transform(y_shot)
#             y_test = y_train[task_plot, rand_inds[-1]]
#             y_test = scaler_y_plot.transform(y_test)

#         # repeat tasks so total meta train epochs is satisfied
#         if n_tasks < (outer_epochs*n_tasks):
#             tasks = np.tile(tasks, int(np.ceil(outer_epochs*n_tasks/n_tasks))) 
        
#         loss_rec = []
#         # Outer loop
#         progress = tqdm(range(outer_epochs*n_tasks*inner_epochs*n_samples))
#         for i, task in enumerate(tasks):
#             # Create a deepcopy of the model to reset after both inner and outer loops
#             weights_before = copy.deepcopy(self.model.state_dict())

#             # Shuffle indices and split into support and query sets
#             spt_size = int(spt_frac * n_samples)
#             permu = self.rng.permutation(n_samples)
#             spt_inds = permu[:spt_size]
#             qry_inds = permu[spt_size:]

#             # Select support and query data, then scale appropriately
#             x_spt_raw = x_train[task][spt_inds, :]
#             y_spt_raw = y_train[task][spt_inds, :]
#             x_qry_raw = x_train[task][qry_inds, :]
#             y_qry_raw = y_train[task][qry_inds, :]

#             # Scaling should be fit only on support set
#             if scale:
#                 scaler_x = StandardScaler()
#                 scaler_y = StandardScaler()
#                 x_spt = scaler_x.fit_transform(x_spt_raw)
#                 y_spt = scaler_y.fit_transform(y_spt_raw)
#                 x_qry = scaler_x.transform(x_qry_raw)
#                 y_qry = scaler_y.transform(y_qry_raw)
#             else:
#                 x_spt, y_spt = x_spt_raw, y_spt_raw
#                 x_qry, y_qry = x_qry_raw, y_qry_raw

#             # Inner loop: Train on support data
#             for j in range(inner_epochs):
#                 innerstep = inner_lr * (1 - j * inner_decay)
#                 self.train_on_batch(x_spt, y_spt, innerstep, loss_fn)
#                 progress.update(1)

#             # Outer loop: Evaluate and update using query data
#             outerstep = outer_lr * (1 - i * outer_decay)
#             self.model.zero_grad()
#             y_pred = self.model(self.to_torch(x_qry))
#             loss = loss_fn(y_pred, self.to_torch(y_qry))
#             loss.backward()
#             loss_rec.append(loss.item())

#             # Reload pre-inner loop weights for the next task
#             self.model.load_state_dict(weights_before)

#             # Update weights
#             for param in self.model.parameters():
#                 param.data -= outerstep * param.grad.data

#             # Plot progress every n_tasks/10 outer epochs
#             # Progress is measured as how quickly and accurately 
#             # the model can adapt to a new task
#             if plot_prog and (i+1) % int(len(tasks)/10) == 0:
#                 plt.cla()
#                 #plt.ylim([0.8,1.2])
#                 plt.title('Epoch: %d, Shots: %d' % (i+1, n_shots))
#                 weights_before = copy.deepcopy(self.model.state_dict())
#                 y_pred = self.predict(x_test)
#                 plt.plot(y_pred[0]/y_test[0], label='MAML iter 0', ls='-')
#                 for inneriter in range(32):
#                     innerstep = inner_lr * (1 - inneriter * inner_decay)
#                     self.train_on_batch(x_shot, y_shot, innerstep, loss_fn)
#                     if (inneriter+1) % 8 == 0:
#                         y_pred = self.predict(x_test)
#                         plt.plot(y_pred[0]/y_test[0],
#                                  label='MAML iter %d' % (inneriter+1),
#                                  ls='-'
#                                  )
#                 plt.plot(y_test[0]/y_test[0], label='Truth', ls='--')
#                 y_pred_final = self.predict(x_test)
#                 loss = loss_fn(self.to_torch(y_pred_final),
#                                self.to_torch(y_test)
#                                ).item()
#                 plt.legend()
#                 plt.xlabel('Output index')
#                 plt.ylabel('Predicted/Truth')
#                 plt.ylim([0.8, 1.2])
#                 plt.savefig('maml_ratio_final.pdf', bbox_inches='tight')
#                 plt.pause(0.01)
#                 self.model.load_state_dict(weights_before)
#                 print('Loss:',loss)

#         y_pred_final = y_pred_final.reshape(1, -1)
#         y_pred_final = scaler_y_plot.inverse_transform(y_pred_final)
#         y_test = y_test.reshape(1, -1)
#         y_test = scaler_y_plot.inverse_transform(y_test)

#         plt.figure()
#         plt.title('Epoch: %d, Shots: %d' % (i+1, n_shots))
#         plt.plot(y_pred_final[0], label='Predicted')
#         plt.plot(y_test[0], ls='--', label='Truth')
#         #plt.yscale('log')
#         plt.legend()
#         plt.savefig('maml_output_final.pdf', bbox_inches='tight')

#         if rec_loss:
#             return loss_rec
        
#     def finetune_predict(self, x_train, y_train, x_test, adapt_steps):
#         weights_before = copy.deepcopy(self.model.state_dict())
#         for i in range(adapt_steps):
#             innerstep = self.inner_lr * (1 - i * self.inner_decay)
#             self.train_on_batch(x_train, y_train, innerstep)
#         y_pred = self.predict(x_test)
#         self.model.load_state_dict(weights_before)
#         return y_pred

# class Adam_MAML():
#     '''
#         Class for MAML implementation of an angular power spectrum emulator.
#         Uses Adam optimizer instead of SGD.
#     '''
#     def __init__(self,
#                  model,
#                  loss_fn=torch.nn.MSELoss(),
#                  inner_lr=0.001,
#                  outer_lr=0.01,
#                  inner_decay=1e-5,
#                  outer_decay=1e-4,
#                  seed=14,
#                  beta1=0.9,
#                  beta2=0.999,
#                  epsilon=1e-8,
#                  device=None
#                  ):
        
#         self.model = model
#         self.seed = seed

#         if device is None:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         else:
#             self.device = device
#         self.model.to(self.device)

#         self.rng = np.random.RandomState(seed)
        
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.epsilon = epsilon

#         self.adam_params = self._init_adam_params()

#         # Initialise scalers for training
#         self.scaler_x = TorchStandardScaler()
#         self.scaler_y = TorchStandardScaler()

#         # Initialise scalers for plotting
#         self.scaler_x_test = TorchStandardScaler()
#         self.scaler_y_test = TorchStandardScaler()

#         # Initialise loss function
#         self.loss_fn = loss_fn

#         # Initialise learning rates and decay
#         self.inner_lr = inner_lr
#         self.outer_lr = outer_lr
#         self.inner_decay = inner_decay
#         self.outer_decay = outer_decay

#     def _init_adam_params(self):
#         """Initialize the Adam optimizer parameters for each model parameter."""
#         adam_params = {}
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 adam_params[name] = {
#                     'm': torch.zeros_like(param.data).to(self.device),
#                     'v': torch.zeros_like(param.data).to(self.device),
#                     't': 0
#                 }
#         return adam_params

#     # convert numpy array to torch tensor
#     def _to_torch(self, arr):
#         return ag.Variable(torch.tensor(arr, dtype=torch.float32).to(self.device))
    
#     def _split_spt_qry(self, x_train, y_train, task, spt_frac):

#         n_samples = y_train.shape[1]

#         # Shuffle indices and split into support and query sets
#         spt_size = int(spt_frac * n_samples)
#         permu = self.rng.permutation(n_samples)
#         spt_inds = permu[:spt_size]
#         qry_inds = permu[spt_size:]

#         # Select support and query data, then scale appropriately
#         x_spt_raw = x_train[task][spt_inds, :]
#         y_spt_raw = y_train[task][spt_inds, :]
#         x_qry_raw = x_train[task][qry_inds, :]
#         y_qry_raw = y_train[task][qry_inds, :]

#         x_spt = self.scaler_x.fit_transform(x_spt_raw)
#         y_spt = self.scaler_y.fit_transform(y_spt_raw)

#         x_qry = self.scaler_x.transform(x_qry_raw)
#         y_qry = self.scaler_y.transform(y_qry_raw)

#         return x_spt, y_spt, x_qry, y_qry
    
#     def _split_shot_test(self, x_train, y_train, test_task, rand_inds):
        
#         test_inds = np.arange(len(x_train[0]))
#         test_inds = np.delete(test_inds, rand_inds)

#         x_shot = x_train[test_task, rand_inds]
#         y_shot = y_train[test_task, rand_inds]
        
#         # Fit scaler on support data and transform both support and test data
#         x_shot = self.scaler_x_test.fit_transform(x_shot)
#         y_shot = self.scaler_y_test.fit_transform(y_shot)

#         x_test = x_train[test_task, test_inds]
#         x_test = self.scaler_x_test.transform(x_test)
#         y_test = y_train[test_task, test_inds]
#         y_test = self.scaler_y_test.transform(y_test)
        
#         return x_shot, y_shot, x_test, y_test
    
#     def _update_adam_params(self, step):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 adam_state = self.adam_params[name]
#                 grad = param.grad.data
#                 # Update biased first moment estimate
#                 adam_state['m'] = self.beta1 * adam_state['m'] + (1 - self.beta1) * grad
#                 # Update biased second raw moment estimate
#                 adam_state['v'] = self.beta2 * adam_state['v'] + (1 - self.beta2) * (grad ** 2)
#                 adam_state['t'] += 1

#                 # Compute bias-corrected first moment estimate
#                 m_hat = adam_state['m'] / (1 - self.beta1 ** adam_state['t'])
#                 # Compute bias-corrected second raw moment estimate
#                 v_hat = adam_state['v'] / (1 - self.beta2 ** adam_state['t'])

#                 # Update parameters
#                 param.data -= step * m_hat / (torch.sqrt(v_hat) + self.epsilon)
    
#     # send individual data batch to model for training step
#     def _train_on_batch(self, x, y, step):
        
#         # Check if model has dropout layers and if so enable dropout
#         if hasattr(self.model, 'dropout'):
#             self.model.train()

#         # Zero gradients before training
#         self.model.zero_grad()
#         y_pred = self.model(x)
#         loss = self.loss_fn(y_pred, y)
#         loss.backward(retain_graph=True) # backpropagate loss

#         # Update model parameters using Adam optimizer
#         self._update_adam_params(step)

#     def _finetune(self, x_train, y_train, adapt_steps, safe=True, x_test=None, n_pred=None):
#         # In safe mode, model is reset to pre-finetuning weights after fine-tuning
#         if safe:
#             weights_before = copy.deepcopy(self.model.state_dict())
#             # save the adam state before fine-tuning
#             adam_params_before = copy.deepcopy(self.adam_params)

#         # Fine-tune model on shot data
#         for i in range(adapt_steps):
#             innerstep = self.inner_lr * (1 - i * self.inner_decay)
#             self._train_on_batch(x_train, y_train, innerstep)

#         # Reset model to pre-finetuning weights and return a prediction
#         if safe:
#             with torch.no_grad():
#                 if n_pred is not None:
#                     self.model.train()
#                     y_pred = []
#                     for i in range(n_pred):
#                         y_pred.append(self.model(x_test))
#                     y_pred = torch.stack(y_pred)
#                 else:
#                     self.model.eval()
#                     y_pred = self.model(x_test)
#             self.model.load_state_dict(weights_before)
#             # reset the adam state after fine-tuning
#             self.adam_params = adam_params_before
#             del adam_params_before, x_test
#             return y_pred

#     def _plot_progress(self, i, x_shot, y_shot, x_test, y_test, ell_bins):

#         # Create a deepcopy of the model to reset after fine-tuning
#         weights_before = copy.deepcopy(self.model.state_dict())
#         adam_params_before = copy.deepcopy(self.adam_params)

#         n_shots = x_shot.shape[1] # Number of shots for fine-tuning

#         # Initial prediction before fine-tuning
#         y_pred = self.model(x_test)
#         loss = self.loss_fn(y_pred, y_test)

#         # Transform prediction back to original space
#         y_pred = self.scaler_y_test.inverse_transform(y_pred)
#         y_pred = y_pred.detach().cpu().numpy()
#         y_pred = np.exp(y_pred)

#         # Transform test data for comparison
#         y_test_comp = self.scaler_y_test.inverse_transform(y_test)
#         y_test_comp = y_test_comp.detach().cpu().numpy()
#         y_test_comp = np.exp(y_test_comp)

#         # Compute mean absolute percentage error
#         _, err_avg = self._compute_mape(y_pred, y_test_comp)

#         # Inititalise plot
#         plt.cla()
#         plt.title('Epoch: %d, Shots: %d' % (i+1, n_shots))
#         plt.plot(ell_bins, err_avg, label='MAML iter 0', ls='-')

#         plots = int(32/8) # Plot every 8 shots
#         for plot in range(plots):
#             self._finetune(x_shot, y_shot, 8, safe=False) # Fine-tune model on shots
#             y_pred = self.model(x_test)
#             loss = self.loss_fn(y_pred, y_test)

#             # Transform prediction back to original space
#             y_pred = self.scaler_y_test.inverse_transform(y_pred)
#             y_pred = y_pred.detach().cpu().numpy()
#             y_pred = np.exp(y_pred)

#             _, err_avg = self._compute_mape(y_pred, y_test_comp)

#             plt.plot(ell_bins, err_avg, label='MAML iter %d' % ((plot+1)*8), ls='-')

#         plt.legend()
#         plt.xlabel(r'$\ell$')
#         plt.ylabel('Avg. MAPE')
#         plt.title('Loss: %.4f' % loss)
#         plt.pause(0.01)

#         # Reload pre-inner loop weights for the next task
#         self.model.load_state_dict(weights_before)
#         self.adam_params = adam_params_before

#         del weights_before, y_pred, y_test_comp, loss, adam_params_before

#     def _compute_mape(self, y_pred, y_test):

#         err = np.empty((y_pred.shape[0], y_pred.shape[1]))
#         for k in range(y_pred.shape[0]):
#             err[k] = abs(y_pred[k,:] - y_test[k,:])/y_test[k,:] * 100
#         err_avg = np.mean(err, axis=0)

#         return err, err_avg
    
#     # Primary training function for MAML to be called by user
#     def meta_train(self, x_train, y_train,
#                    spt_frac, outer_epochs, inner_epochs,
#                    rec_loss=True, n_shots=32, plot_prog=False, plot_summary=True, ell_bins=None):
        
#         # Check if ell_bins are provided, if not just plot against index
#         if ell_bins is None:
#             ell_bins = np.arange(y_train.shape[2])
        
#         # move data to torch tensors
#         x_train = self._to_torch(x_train)
#         y_train = self._to_torch(y_train)
        
#         # ascertaining the number of tasks and samples
#         n_tasks = y_train.shape[0]
#         n_samples = y_train.shape[1]
#         tasks = np.arange(n_tasks)
#         tasks = self.rng.permutation(tasks) # permute the order tasks are trained

#         if plot_prog:
#             # Isolate a random task for checking the progress of training
#             test_task = self.rng.choice(tasks, size=1)
#             tasks = np.delete(tasks, test_task)
#             n_tasks = len(tasks)
#             rand_inds = self.rng.choice(n_samples, size=n_shots, replace=False)

#             # Split test plot data into support and query sets
#             x_shot, y_shot, x_test, y_test = self._split_shot_test(x_train, y_train, test_task, rand_inds)

#         # Repeat tasks so total meta train epochs is satisfied
#         if n_tasks < (outer_epochs*n_tasks):
#             tasks = np.tile(tasks, int(np.ceil(outer_epochs*n_tasks/n_tasks))) 
        
#         loss_rec = [] # list to record meta loss

#         # Outer loop
#         for i, task in tqdm(enumerate(tasks)):
#             # Create a deepcopy of the model to reset after both inner and outer loops
#             weights_before = copy.deepcopy(self.model.state_dict())

#             # Split data into support and query sets
#             x_spt, y_spt, x_qry, y_qry = self._split_spt_qry(x_train, y_train, task, spt_frac)

#             # Inner loop: Train on support data
#             for j in range(inner_epochs):
#                 innerstep = self.inner_lr * (1 - j * self.inner_decay)
#                 self._train_on_batch(x_spt, y_spt, innerstep)

#             # Check if model has dropout layers and if so enable dropout
#             if hasattr(self.model, 'dropout'):
#                 self.model.train()

#             # Outer loop: Evaluate and update using query data
#             outerstep = self.outer_lr * (1 - i * self.outer_decay)

#             self.model.zero_grad()
#             y_pred = self.model(x_qry)
#             loss = self.loss_fn(y_pred, y_qry)
#             loss.backward() # backpropagate loss
#             loss_rec.append(loss.item()) # Record loss

#             # Reload pre-inner loop weights for the next task
#             self.model.load_state_dict(weights_before)

#             # Update weights using Adam for outer loop
#             self._update_adam_params(outerstep)

#             # Plot progress every n_tasks/10 outer epochs
#             # Progress is measured as how quickly and accurately 
#             # the model can adapt to a new task
#             if (i+1) % int(len(tasks)/10) == 0 and plot_prog:
#                 self._plot_progress(i, x_shot, y_shot, x_test, y_test, ell_bins)

#         # Plot a final summary of the model
#         if plot_prog and plot_summary:
            
#             # Create a deepcopy of the model to reset after fine-tuning
#             weights_before = copy.deepcopy(self.model.state_dict())

#             # Finetune model on all shots
#             self._finetune(x_shot, y_shot, adapt_steps=32, safe=False)
            
#             # Plot summary of model
#             self.plot_summary(x_test, y_test, ell_bins, outer_epochs, loss_rec, n_shots)

#             # Reset model to pre-finetuning weights
#             self.model.load_state_dict(weights_before)

#         # clean up memory
#         del x_train, y_train, x_spt, y_spt, x_qry, y_qry
#         if plot_prog:
#             del x_shot, y_shot, x_test, y_test

#         if rec_loss:
#             return loss_rec
        
#     def cross_validate(self, x_val, y_val, n_shots, adapt_steps):

#         # ascertaining the number of tasks and samples
#         n_tasks = y_val.shape[0]
#         n_samples = y_val.shape[1]

#         # move data to torch tensors
#         x_val = self._to_torch(x_val)
#         y_val = self._to_torch(y_val)

#         # loop through tasks and test the model
#         err_all = []
#         err_avg_all = []
#         predictions = []
#         truths = []
#         for i in range(n_tasks):
#             rand_inds = self.rng.choice(n_samples, size=n_shots, replace=False)

#             # Split test plot data into support and query sets
#             x_shot, y_shot, x_test, y_test = self._split_shot_test(
#                 x_val,
#                 y_val,
#                 test_task=i,
#                 rand_inds=rand_inds
#             )

#             # Fine-tune model on shots
#             if hasattr(self.model, 'dropout'):
#                 n_pred = 100
#             else:
#                 n_pred = None

#             y_pred_all = self._finetune(
#                 x_shot,
#                 y_shot,
#                 adapt_steps,
#                 safe=True,
#                 x_test=x_test,
#                 n_pred=n_pred
#             )
#             y_pred_all = self.scaler_y_test.inverse_transform(y_pred_all)
#             predictions.append(torch.exp(y_pred_all))

#             # Compute mean absolute percentage error
#             y_pred = y_pred_all.mean(0)
#             y_pred = y_pred.detach().cpu().numpy()
#             y_pred = np.exp(y_pred)

#             y_test_comp = self.scaler_y_test.inverse_transform(y_test)
#             y_test_comp = y_test_comp.detach().cpu().numpy()
#             y_test_comp = np.exp(y_test_comp)
#             truths.append(y_test_comp)

#             err, err_avg = self._compute_mape(y_pred, y_test_comp)
#             err_all.append(err)
#             err_avg_all.append(err_avg)

#             del x_shot, y_shot, x_test

#         predictions = torch.stack(predictions).detach().cpu().numpy()   

#         return err_all, err_avg_all, predictions, truths, rand_inds
            
#     def plot_summary(self, x_test, y_test, ell_bins, loss_rec, n_shots):
        
#         # Enable dropout for final prediction to get uncertainty estimates
#         if hasattr(self.model, 'dropout'):
#             y_pred_all = []
#             for i in range(100):
#                 y_pred = self.model(x_test)
#                 y_pred = self.scaler_y_test.inverse_transform(y_pred)
#                 y_pred = y_pred.detach().cpu().numpy()
#                 y_pred = np.exp(y_pred)
#                 y_pred_all.append(y_pred)

#             y_pred_all = np.array(y_pred_all)
#             # Standard deviation of predictions
#             y_pred_std = np.std(y_pred_all, axis=0)
#             y_pred_all = np.mean(y_pred_all, axis=0)
#         else:
#             y_pred = self.model(x_test)
#             y_pred = self.scaler_y_test.inverse_transform(y_pred)
#             y_pred = y_pred.detach().cpu().numpy()
#             y_pred_all = np.exp(y_pred)

#         y_all_comp = self.scaler_y_test.inverse_transform(y_test)
#         y_all_comp = y_all_comp.detach().cpu().numpy()
#         y_all_comp = np.exp(y_all_comp)

#         err, err_avg = self._compute_mape(y_pred_all, y_all_comp)

#         _, axs = plt.subplots(1, 4, figsize=(20, 5))
#         for i in range(len(err)):
#             axs[0].plot(ell_bins, err[i], ls='-', alpha=0.1)
#         axs[0].plot(ell_bins, err_avg, ls='-', c='k', label='Average')
#         axs[0].legend()
#         axs[0].set_xscale('log')
#         axs[0].set_title('Absolute percentage error')

#         # Select worst performing sample for plotting
#         worst_sample = np.argmax(np.mean(err, axis=1))
#         axs[1].plot(ell_bins, y_pred_all[worst_sample], label='Prediction')
#         axs[1].plot(ell_bins, y_all_comp[worst_sample], label='Truth', ls='--')
#         if hasattr(self.model, 'dropout'):
#             axs[1].fill_between(ell_bins, y_pred_all[worst_sample]-y_pred_std[worst_sample],
#                                 y_pred_all[worst_sample]+y_pred_std[worst_sample],
#                                 alpha=0.5, label='Uncertainty')
#         axs[1].legend()
#         axs[1].set_title('Sample %d, Shots: %d' % (worst_sample, n_shots))
#         axs[1].set_xscale('log')
#         axs[1].set_yscale('log')

#         axs[2].plot(ell_bins, y_pred_all[worst_sample]/y_all_comp[worst_sample], 
#                     label='Prediction/Truth')
#         if hasattr(self.model, 'dropout'):
#             axs[2].fill_between(ell_bins, 
#                                 (y_pred_all[worst_sample]-y_pred_std[worst_sample])/y_all_comp[worst_sample],
#                                 (y_pred_all[worst_sample]+y_pred_std[worst_sample])/y_all_comp[worst_sample],
#                                 alpha=0.5, label='Uncertainty')
#         axs[2].set_xscale('log')
#         axs[2].legend()
#         axs[2].set_title('Sample %d, Shots: %d' % (worst_sample, n_shots))
#         axs[3].plot(loss_rec, label='Loss')
#         axs[3].set_yscale('log')
#         axs[3].legend()
#         axs[3].set_title('Outer loop loss')

#         plt.savefig('maml_output_final.pdf', bbox_inches='tight') 