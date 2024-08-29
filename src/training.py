import torch
from torch import autograd as ag
import matplotlib.pyplot as plt
import numpy as np
import copy

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MAML():
    '''
        Class for MAML implemntation of an angular power spectrum emulator.
        SGD is currently hardcoded, need to provide different optimizers.
    '''
    def __init__(self, model, seed=14,
                 ):
        self.model = model
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.rng = np.random.RandomState(seed)

    # convert numpy array to torch tensor
    def to_torch(self, arr):
        return ag.Variable(torch.tensor(arr, dtype=torch.float32).to(self.device))
    
    # send individual data batch to model for training step
    def train_on_batch(self, x ,y, step, loss_fn):
        x = self.to_torch(x)
        y = self.to_torch(y)
        self.model.zero_grad()
        y_pred = self.model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        for param in self.model.parameters():
            param.data -= step * param.grad.data

    # obtain predictions from model
    # Includes built in scaling 
    def predict(self, x):
        x = self.to_torch(x)
        return self.model(x).detach().cpu().numpy()
    
    # meta train the model
    def meta_train(self, x_train, y_train, 
                   inner_lr, outer_lr, loss_fn, 
                   spt_frac, outer_epochs, inner_epochs,
                   inner_decay=1e-4, outer_decay=1e-3, n_shots=5,
                   rec_loss=True, plot_prog=False, scale=True):
        
        # ascertaining the number of tasks and samples
        n_tasks = y_train.shape[0]
        n_samples = y_train.shape[1]
        tasks = np.arange(n_tasks)
        tasks = self.rng.permutation(tasks) # permute the order tasks are trained

        # isolate a random task for plotting the progress of training
        if plot_prog:
            task_plot = self.rng.choice(tasks, size=1)
            print('Plotting task {}'.format(task_plot))
            tasks= np.delete(tasks, task_plot)
            n_tasks = len(tasks)
            rand_inds = self.rng.choice(n_samples, size=n_shots+1, replace=False)

            # print('Training samples:', rand_inds[:-1])
            # print('Test samples:', rand_inds[-1])

            scaler_x_plot = StandardScaler()
            scaler_y_plot = StandardScaler()

            x_shot = x_train[task_plot, rand_inds[:-1]]
            y_shot = y_train[task_plot, rand_inds[:-1]]

            # Fit scaler on support data and transform both support and test data
            x_shot = scaler_x_plot.fit_transform(x_shot)
            x_test = x_train[task_plot, rand_inds[-1]]
            x_test = scaler_x_plot.transform(x_test)

            y_shot = scaler_y_plot.fit_transform(y_shot)
            y_test = y_train[task_plot, rand_inds[-1]]
            y_test = scaler_y_plot.transform(y_test)

        # repeat tasks so total meta train epochs is satisfied
        if n_tasks < (outer_epochs*n_tasks):
            tasks = np.tile(tasks, int(np.ceil(outer_epochs*n_tasks/n_tasks))) 
        
        loss_rec = []
        # Outer loop
        progress = tqdm(range(outer_epochs*n_tasks*inner_epochs*n_samples))
        for i, task in enumerate(tasks):
            # Create a deepcopy of the model to reset after both inner and outer loops
            weights_before = copy.deepcopy(self.model.state_dict())

            # Shuffle indices and split into support and query sets
            spt_size = int(spt_frac * n_samples)
            permu = self.rng.permutation(n_samples)
            spt_inds = permu[:spt_size]
            qry_inds = permu[spt_size:]

            # Select support and query data, then scale appropriately
            x_spt_raw = x_train[task][spt_inds, :]
            y_spt_raw = y_train[task][spt_inds, :]
            x_qry_raw = x_train[task][qry_inds, :]
            y_qry_raw = y_train[task][qry_inds, :]

            # Scaling should be fit only on support set
            if scale:
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()
                x_spt = scaler_x.fit_transform(x_spt_raw)
                y_spt = scaler_y.fit_transform(y_spt_raw)
                x_qry = scaler_x.transform(x_qry_raw)
                y_qry = scaler_y.transform(y_qry_raw)
            else:
                x_spt, y_spt = x_spt_raw, y_spt_raw
                x_qry, y_qry = x_qry_raw, y_qry_raw

            # Inner loop: Train on support data
            for j in range(inner_epochs):
                innerstep = inner_lr * (1 - j * inner_decay)
                self.train_on_batch(x_spt, y_spt, innerstep, loss_fn)
                progress.update(1)

            # Outer loop: Evaluate and update using query data
            outerstep = outer_lr * (1 - i * outer_decay)
            self.model.zero_grad()
            y_pred = self.model(self.to_torch(x_qry))
            loss = loss_fn(y_pred, self.to_torch(y_qry))
            loss.backward()
            loss_rec.append(loss.item())

            # Reload pre-inner loop weights for the next task
            self.model.load_state_dict(weights_before)

            # Update weights
            for param in self.model.parameters():
                param.data -= outerstep * param.grad.data

            # Plot progress every n_tasks/10 outer epochs
            # Progress is measured as how quickly and accurately 
            # the model can adapt to a new task
            if plot_prog and (i+1) % int(len(tasks)/10) == 0:
                plt.cla()
                #plt.ylim([0.8,1.2])
                plt.title('Epoch: %d, Shots: %d' % (i+1, n_shots))
                weights_before = copy.deepcopy(self.model.state_dict())
                y_pred = self.predict(x_test)
                plt.plot(y_pred[0]/y_test[0], label='MAML iter 0', ls='-')
                for inneriter in range(32):
                    innerstep = inner_lr * (1 - inneriter * inner_decay)
                    self.train_on_batch(x_shot, y_shot, innerstep, loss_fn)
                    if (inneriter+1) % 8 == 0:
                        y_pred = self.predict(x_test)
                        plt.plot(y_pred[0]/y_test[0],
                                 label='MAML iter %d' % (inneriter+1),
                                 ls='-'
                                 )
                plt.plot(y_test[0]/y_test[0], label='Truth', ls='--')
                y_pred_final = self.predict(x_test)
                loss = loss_fn(self.to_torch(y_pred_final),
                               self.to_torch(y_test)
                               ).item()
                plt.legend()
                plt.xlabel('Output index')
                plt.ylabel('Predicted/Truth')
                plt.ylim([0.8, 1.2])
                plt.savefig('maml_ratio_final.pdf', bbox_inches='tight')
                plt.pause(0.01)
                self.model.load_state_dict(weights_before)
                print('Loss:',loss)

        y_pred_final = y_pred_final.reshape(1, -1)
        y_pred_final = scaler_y_plot.inverse_transform(y_pred_final)
        y_test = y_test.reshape(1, -1)
        y_test = scaler_y_plot.inverse_transform(y_test)

        plt.figure()
        plt.title('Epoch: %d, Shots: %d' % (i+1, n_shots))
        plt.plot(y_pred_final[0], label='Predicted')
        plt.plot(y_test[0], ls='--', label='Truth')
        #plt.yscale('log')
        plt.legend()
        plt.savefig('maml_output_final.pdf', bbox_inches='tight')

        if rec_loss:
            return loss_rec
        
    def finetune_predict(self, x_train, y_train, x_test, adapt_steps):
        weights_before = copy.deepcopy(self.model.state_dict())
        for i in range(adapt_steps):
            innerstep = self.inner_lr * (1 - i * self.inner_decay)
            self.train_on_batch(x_train, y_train, innerstep)
        y_pred = self.predict(x_test)
        self.model.load_state_dict(weights_before)
        return y_pred
    

class PCA_MAML():
    '''
        MAML training with SGD, utilising PCA to reduce
        dimensionality of output data.
    '''
    def __init__(self, model, seed=14, pca_components=3
                 ):
        self.model = model
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.rng = np.random.RandomState(seed)
        self.pca_components = pca_components

    # convert numpy array to torch tensor
    def to_torch(self, arr):
        return ag.Variable(torch.tensor(arr, dtype=torch.float32).to(self.device))
    
    # send individual data batch to model for training step
    def train_on_batch(self, x ,y, step, loss_fn):
        x = self.to_torch(x)
        y = self.to_torch(y)
        self.model.zero_grad()
        y_pred = self.model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        for param in self.model.parameters():
            param.data -= step * param.grad.data

    # obtain predictions from model
    def predict(self, x):
        x = self.to_torch(x)
        return self.model(x).detach().cpu().numpy()
    
    # meta train the model
    def meta_train(self, x_train, y_train, 
                   inner_lr, outer_lr, loss_fn, 
                   spt_frac, outer_epochs, inner_epochs,
                   inner_decay=1e-4, outer_decay=1e-3, n_shots=5,
                   rec_loss=True, plot_prog=False):
        
        # ascertaining the number of tasks and samples
        n_tasks = y_train.shape[0]
        n_samples = y_train.shape[1]
        tasks = np.arange(n_tasks)
        tasks = self.rng.permutation(tasks) # permute the order tasks are trained

        # isolate a random task for plotting the progress of training
        if plot_prog:
            task_plot = self.rng.choice(tasks, size=1)

            tasks= np.delete(tasks, task_plot)
            n_tasks = len(tasks)
            rand_inds = self.rng.choice(n_samples, size=n_shots+1, replace=False)

            scaler_x_plot = StandardScaler()

            x_shot = x_train[task_plot, rand_inds[:-1]]
            x_shot = scaler_x_plot.fit_transform(x_shot)

            x_test = x_train[task_plot, rand_inds[-1]]
            x_test = scaler_x_plot.transform(x_test)

            # Fit PCA and scale outputs
            pca_plot = PCA(n_components=self.pca_components)
            scaler_y_plot = StandardScaler()

            y_shot = y_train[task_plot, rand_inds[:-1]]
            y_shot = scaler_y_plot.fit_transform(y_shot)
            y_shot = pca_plot.fit_transform(y_shot)

            y_test = y_train[task_plot, rand_inds[-1]]
            y_test = scaler_y_plot.transform(y_test)
            y_test = pca_plot.transform(y_test)
           
        # repeat tasks so total meta train epochs is satisfied
        if n_tasks < (outer_epochs*n_tasks):
            tasks = np.tile(tasks, int(np.ceil(outer_epochs*n_tasks/n_tasks))) 
        
        loss_rec = []
        # Outer loop
        for i, task in enumerate(tasks):
            # Create a deepcopy of the model to reset after both inner and outer loops
            weights_before = copy.deepcopy(self.model.state_dict())

            # Shuffle indices and split into support and query sets
            spt_size = int(spt_frac * n_samples)
            permu = self.rng.permutation(n_samples)
            spt_inds = permu[:spt_size]
            qry_inds = permu[spt_size:]

            # Select support and query data, then scale appropriately
            x_spt = x_train[task, spt_inds]
            y_spt = y_train[task, spt_inds]
            x_qry = x_train[task, qry_inds]
            y_qry = y_train[task, qry_inds]

            # Scaling should be fit only on support set
            scaler_x = StandardScaler()
            
            x_spt = scaler_x.fit_transform(x_spt)
            x_qry = scaler_x.transform(x_qry)

            pca_task = PCA(n_components=self.pca_components)
            scaler_y = StandardScaler()
            
            y_spt = scaler_y.fit_transform(y_spt)
            y_spt = pca_task.fit_transform(y_spt)
            
            y_qry = scaler_y.transform(y_qry)
            y_qry = pca_task.transform(y_qry)
            
            # Inner loop: Train on support data
            for j in range(inner_epochs):
                innerstep = inner_lr * (1 - j * inner_decay)
                self.train_on_batch(x_spt, y_spt, innerstep, loss_fn)

            # Outer loop: Evaluate and update using query data
            outerstep = outer_lr * (1 - i * outer_decay)
            self.model.zero_grad()
            y_pred = self.model(self.to_torch(x_qry))
            loss = loss_fn(y_pred, self.to_torch(y_qry))
            loss.backward()
            loss_rec.append(loss.item())

            # Reload pre-inner loop weights for the next task
            self.model.load_state_dict(weights_before)

            # Update weights
            for param in self.model.parameters():
                param.data -= outerstep * param.grad.data

            # Plot progress every n_tasks/10 outer epochs
            # Progress is measured as how quickly and accurately 
            # the model can adapt to a new task
            if plot_prog and (i+1) % int(len(tasks)/10) == 0:
                plt.cla()
                plt.title('Epoch: %d, Shots: %d' % (i+1, n_shots))
                weights_before = copy.deepcopy(self.model.state_dict())
                y_pred = self.predict(x_test)
                plt.plot(y_pred[0]/y_test[0], label='MAML iter 0', ls='-')
                for inneriter in range(32):
                    innerstep = inner_lr * (1 - inneriter * inner_decay)
                    self.train_on_batch(x_shot, y_shot, innerstep, loss_fn)
                    if (inneriter+1) % 8 == 0:
                        y_pred = self.predict(x_test)
                        plt.plot(y_pred[0]/y_test[0],
                                 label='MAML iter %d' % (inneriter+1),
                                 ls='-'
                                 )
                plt.plot(y_test[0]/y_test[0], label='Truth', ls='--')
                y_pred_final = self.predict(x_test)
                loss = loss_fn(self.to_torch(y_pred_final),
                               self.to_torch(y_test)
                               ).item()
                plt.legend()
                plt.xlabel('Output index')
                plt.ylabel('Predicted/Truth')
                plt.ylim([0.8, 1.2])
                plt.savefig('maml_ratio_final.pdf', bbox_inches='tight')
                plt.pause(0.01)
                self.model.load_state_dict(weights_before)
                print('Loss:',loss)

        y_pred_final = y_pred_final.reshape(1, -1)
        y_pred_final = pca_plot.inverse_transform(y_pred_final)
        y_pred_final = scaler_y_plot.inverse_transform(y_pred_final)

        y_test = y_test.reshape(1, -1)
        y_test = pca_plot.inverse_transform(y_test)
        y_test = scaler_y_plot.inverse_transform(y_test)

        plt.figure()
        plt.title('Epoch: %d, Shots: %d' % (i+1, n_shots))
        plt.plot(y_pred_final[0], label='Predicted')
        plt.plot(y_test[0], ls='--', label='Truth')
        #plt.yscale('log')
        plt.legend()
        plt.savefig('maml_output_final.pdf', bbox_inches='tight')

        if rec_loss:
            return loss_rec
        
    def finetune_predict(self, x_train, y_train, x_test, adapt_steps):
        weights_before = copy.deepcopy(self.model.state_dict())
        for i in range(adapt_steps):
            innerstep = self.inner_lr * (1 - i * self.inner_decay)
            self.train_on_batch(x_train, y_train, innerstep)
        y_pred = self.predict(x_test)
        self.model.load_state_dict(weights_before)
        return y_pred

class Adam_MAML():
    '''
        Class for MAML implementation of an angular power spectrum emulator.
        Uses Adam optimizer instead of SGD.
    '''
    def __init__(self, model, seed=14,
                 ):
        self.model = model
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.rng = np.random.RandomState(seed)
        
        # Initialize Adam optimizer parameters
        self.adam_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.adam_params[name] = {
                    'm': torch.zeros_like(param.data).to(self.device),
                    'v': torch.zeros_like(param.data).to(self.device),
                    't': 0
                }

    # convert numpy array to torch tensor
    def to_torch(self, arr):
        return ag.Variable(torch.tensor(arr, dtype=torch.float32).to(self.device))
    
    # send individual data batch to model for training step
    def train_on_batch(self, x, y, step, loss_fn, beta1=0.9, beta2=0.999, epsilon=1e-8):

        if hasattr(self.model, 'dropout'):
            self.model.train()

        self.model.zero_grad()
        y_pred = self.model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad = param.grad.data
                adam_state = self.adam_params[name]
                
                # Update biased first moment estimate
                adam_state['m'] = beta1 * adam_state['m'] + (1 - beta1) * grad
                # Update biased second raw moment estimate
                adam_state['v'] = beta2 * adam_state['v'] + (1 - beta2) * (grad ** 2)
                adam_state['t'] += 1

                # Compute bias-corrected first moment estimate
                m_hat = adam_state['m'] / (1 - beta1 ** adam_state['t'])
                # Compute bias-corrected second raw moment estimate
                v_hat = adam_state['v'] / (1 - beta2 ** adam_state['t'])

                # Update parameters
                param.data -= step * m_hat / (torch.sqrt(v_hat) + epsilon)

    # obtain predictions from model
    # Includes built-in scaling 
    def predict(self, x):
        return self.model(x)
    
    # meta train the model
    def meta_train(self, x_train, y_train, 
                   inner_lr, outer_lr, loss_fn, 
                   spt_frac, outer_epochs, inner_epochs,
                   inner_decay=1e-4, outer_decay=1e-3, n_shots=5,
                   rec_loss=True, plot_prog=False, ell_bins=None):
        
        if ell_bins is None:
            ell_bins = np.arange(y_train.shape[2])
        
        # move data to torch tensors
        x_train = self.to_torch(x_train)
        y_train = self.to_torch(y_train)
        
        # ascertaining the number of tasks and samples
        n_tasks = y_train.shape[0]
        n_samples = y_train.shape[1]
        tasks = np.arange(n_tasks)
        tasks = self.rng.permutation(tasks) # permute the order tasks are trained

        # isolate a random task for checking the progress of training
        task_plot = self.rng.choice(tasks, size=1)
        tasks = np.delete(tasks, task_plot)
        n_tasks = len(tasks)
        rand_inds = self.rng.choice(n_samples, size=n_shots+1, replace=False)

        scaler_x_plot = TorchStandardScaler()
        scaler_y_plot = TorchStandardScaler()

        x_shot = x_train[task_plot, rand_inds[:-1]]
        y_shot = y_train[task_plot, rand_inds[:-1]]

        # Fit scaler on support data and transform both support and test data
        x_shot = scaler_x_plot.fit_transform(x_shot)
        x_test = x_train[task_plot, rand_inds[-1]]
        x_test = scaler_x_plot.transform(x_test)

        y_shot = scaler_y_plot.fit_transform(y_shot)
        y_test = y_train[task_plot, rand_inds[-1]]
        y_test = scaler_y_plot.transform(y_test)

        x_all = x_train[task_plot]
        x_all = scaler_x_plot.transform(x_all)
        
        y_all = y_train[task_plot]
        y_all = scaler_y_plot.transform(y_all)

        # repeat tasks so total meta train epochs is satisfied
        if n_tasks < (outer_epochs*n_tasks):
            tasks = np.tile(tasks, int(np.ceil(outer_epochs*n_tasks/n_tasks))) 
        
        loss_rec = []
        progress = tqdm(range(outer_epochs*len(tasks)))
        # Outer loop
        for i, task in enumerate(tasks):
            # Create a deepcopy of the model to reset after both inner and outer loops
            weights_before = copy.deepcopy(self.model.state_dict())

            # Shuffle indices and split into support and query sets
            spt_size = int(spt_frac * n_samples)
            permu = self.rng.permutation(n_samples)
            spt_inds = permu[:spt_size]
            qry_inds = permu[spt_size:]

            # Select support and query data, then scale appropriately
            x_spt_raw = x_train[task][spt_inds, :]
            y_spt_raw = y_train[task][spt_inds, :]
            x_qry_raw = x_train[task][qry_inds, :]
            y_qry_raw = y_train[task][qry_inds, :]

            # Scaling should be fit only on support set
            scaler_x = TorchStandardScaler()
            scaler_y = TorchStandardScaler()
            x_spt = scaler_x.fit_transform(x_spt_raw)
            y_spt = y_spt_raw
            y_spt = scaler_y.fit_transform(y_spt_raw)
            x_qry = scaler_x.transform(x_qry_raw)
            y_qry = y_qry_raw
            y_qry = scaler_y.transform(y_qry_raw)

            # Inner loop: Train on support data
            for j in range(inner_epochs):
                innerstep = inner_lr * (1 - j * inner_decay)
                self.train_on_batch(x_spt, y_spt, innerstep, loss_fn)

            # Outer loop: Evaluate and update using query data
            outerstep = outer_lr * (1 - i * outer_decay)

            if hasattr(self.model, 'dropout'):
                self.model.train()

            self.model.zero_grad()
            y_pred = self.model(x_qry)
            loss = loss_fn(y_pred, y_qry)
            loss.backward()
            loss_rec.append(loss.item())

            # Reload pre-inner loop weights for the next task
            self.model.load_state_dict(weights_before)

            progress.update(1)

            # Update weights using Adam for outer loop
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    grad = param.grad.data
                    adam_state = self.adam_params[name]
                    
                    # Update biased first moment estimate
                    adam_state['m'] = 0.9 * adam_state['m'] + 0.1 * grad
                    # Update biased second raw moment estimate
                    adam_state['v'] = 0.999 * adam_state['v'] + 0.001 * (grad ** 2)
                    adam_state['t'] += 1

                    # Compute bias-corrected first moment estimate
                    m_hat = adam_state['m'] / (1 - 0.9 ** adam_state['t'])
                    # Compute bias-corrected second raw moment estimate
                    v_hat = adam_state['v'] / (1 - 0.999 ** adam_state['t'])

                    # Update parameters
                    param.data -= outerstep * m_hat / (torch.sqrt(v_hat) + 1e-8)

            # Plot progress every n_tasks/10 outer epochs
            # Progress is measured as how quickly and accurately 
            # the model can adapt to a new task
            if (i+1) % int(len(tasks)/10) == 0:
                weights_before = copy.deepcopy(self.model.state_dict())

                y_pred = self.predict(x_test)
                y_pred = scaler_y_plot.inverse_transform(y_pred)
                y_pred = y_pred.detach().cpu().numpy()
                y_pred = np.exp(y_pred)

                y_test_comp = scaler_y_plot.inverse_transform(y_test)
                y_test_comp = y_test_comp.detach().cpu().numpy()
                y_test_comp = np.exp(y_test_comp)

                if plot_prog:
                    plt.cla()
                    plt.title('Epoch: %d, Shots: %d' % (i+1, n_shots))
                    plt.plot(ell_bins, y_pred[0]/y_test_comp[0], label='MAML iter 0', ls='-')
                for inneriter in range(32):
                    innerstep = inner_lr * (1 - inneriter * inner_decay)
                    self.train_on_batch(x_shot, y_shot, innerstep, loss_fn)
                    if (inneriter+1) % 8 == 0:
                        y_pred = self.predict(x_test)
                        y_pred = scaler_y_plot.inverse_transform(y_pred)
                        y_pred = y_pred.detach().cpu().numpy()
                        y_pred = np.exp(y_pred)
                        if plot_prog:
                            plt.plot(ell_bins, y_pred[0]/y_test_comp[0],
                                     label='MAML iter %d' % (inneriter+1),
                                     ls='-'
                                     )
                y_pred_final = self.predict(x_test)
                loss = loss_fn(y_pred_final,
                               y_test
                               ).item()

                if plot_prog:
                    plt.plot(ell_bins, y_test_comp[0]/y_test_comp[0], label='Truth', ls='--')
                    plt.legend()
                    plt.xlabel('Output index')
                    plt.ylabel('Predicted/Truth')
                    plt.savefig('maml_ratio_final.pdf', bbox_inches='tight')
                    plt.pause(0.01)
                    print('Loss:',loss)

        # Enable dropout for final prediction to get uncertainty estimates
        if hasattr(self.model, 'dropout'):
            self.model.train()
            y_pred_all = []
            for i in range(1000):
                y_pred = self.model(x_all)
                y_pred = scaler_y_plot.inverse_transform(y_pred)
                y_pred = y_pred.detach().cpu().numpy()
                y_pred = np.exp(y_pred)
                y_pred_all.append(y_pred)
            # Average over all predictions
            y_pred_all = np.array(y_pred_all)
            # Standard deviation of predictions
            y_pred_std = np.std(y_pred_all, axis=0)
            y_pred_all = np.mean(y_pred_all, axis=0)
        else:
            y_pred_all = self.predict(x_all)
            y_pred_all = scaler_y_plot.inverse_transform(y_pred_all)
            y_pred_all = y_pred_all.detach().cpu().numpy()
            y_pred_all = np.exp(y_pred_all)
        
        y_all_comp = scaler_y_plot.inverse_transform(y_all)
        y_all_comp = y_all_comp.detach().cpu().numpy()
        y_all_comp = np.exp(y_all_comp)

        self.model.load_state_dict(weights_before)
                
        progress.close()

        _, axs = plt.subplots(1, 4, figsize=(20, 5))

        err = np.empty((len(y_pred_all[0]), len(ell_bins)))
        for k in range(len(y_pred_all[0])):
            err[k] = abs(y_pred_all[0][k,:] - y_all_comp[0][k,:])/y_all_comp[0][k,:] * 100
            axs[0].plot(ell_bins, err[k],ls='--',alpha=0.2)
        err_avg = np.mean(err, axis=0)
        axs[0].plot(ell_bins, err_avg, ls='-', c='k', label='Average')
        axs[0].legend()
        axs[0].set_xscale('log')
        axs[0].set_title('Absolute percentage error')

        # Select worst performing sample for plotting
        worst_sample = np.argmax(np.mean(err, axis=1))

        axs[1].plot(ell_bins, y_pred_all[0][worst_sample], label='Prediction')
        axs[1].plot(ell_bins, y_all_comp[0][worst_sample], label='Truth', ls='--')
        if hasattr(self.model, 'dropout'):
            axs[1].fill_between(ell_bins, y_pred_all[0][worst_sample]-y_pred_std[0][worst_sample],
                                y_pred_all[0][worst_sample]+y_pred_std[0][worst_sample],
                                alpha=0.5, label='Uncertainty')
        axs[1].legend()
        axs[1].set_title('Sample %d, Shots: %d' % (rand_inds[-1], n_shots))
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')

        axs[2].plot(ell_bins, y_pred_all[0][worst_sample]/y_all_comp[0][worst_sample], 
                    label='Prediction/Truth')
        if hasattr(self.model, 'dropout'):
            axs[2].fill_between(ell_bins, 
                                (y_pred_all[0][worst_sample]-y_pred_std[0][worst_sample])/y_all_comp[0][worst_sample],
                                (y_pred_all[0][worst_sample]+y_pred_std[0][worst_sample])/y_all_comp[0][worst_sample],
                                alpha=0.5, label='Uncertainty')
        axs[2].set_xscale('log')
        axs[2].legend()
        axs[2].set_title('Sample %d, Shots: %d' % (rand_inds[-1], n_shots))
        
        axs[3].plot(loss_rec, label='Loss')
        axs[3].set_yscale('log')
        axs[3].legend()
        axs[3].set_title('Outer loop loss, Epochs: %d' % (outer_epochs*n_tasks))

        plt.savefig('maml_output_final.pdf', bbox_inches='tight')

        if rec_loss:
            return y_pred_all, y_all_comp, y_pred_std, loss_rec
        return y_pred_all, y_all_comp, y_pred_std      
        
    def finetune_predict(self, x_train, y_train, x_test, adapt_steps):
        weights_before = copy.deepcopy(self.model.state_dict())
        for i in range(adapt_steps):
            innerstep = self.inner_lr * (1 - i * self.inner_decay)
            self.train_on_batch(x_train, y_train, innerstep)
        y_pred = self.predict(x_test)
        self.model.load_state_dict(weights_before)
        return y_pred
    
class TorchStandardScaler():

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