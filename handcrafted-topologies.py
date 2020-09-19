#!/usr/bin/env python
# coding: utf-8

# # Handcrafted Meta-Architectures
# 
# In this notebook we implement the :
# - autoencoders, logistic regression and the meta-network 
# - different handcrafted architectures
# - entropy estimators
# 

# ## Libraries

# In[1]:


import sys
import random
import string
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from collections import OrderedDict 
import pandas as pd                 
from sklearn.preprocessing import MinMaxScaler
import sklearn
# from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
import scipy
from pathlib import Path
import json

import os #KP

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Checks
print("PyTorch version : ", torch.__version__)
print("CUDA available? : ", torch.cuda.is_available())

# additional
get_ipython().run_line_magic('matplotlib', 'inline')
# matplotlib.style.use('seaborn-notebook')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Comment out these code if testing locally

from google.colab import drive
drive.mount('/content/drive')
path = Path('drive/My Drive/experiment-results_mnist_3_15jun (handcraft latest github)/') # set your own Google Drive path here


# In[3]:


# Comment out these code if testing on Colab.

#path = Path('/home/hari/Hari/Research and Projects/Self - Organizing - Networks/Misc/test folder') # set your own path here


# In[ ]:


# set folder names here
path_metrics = 'newer-metrics'
path_images = 'newer-images'

# KP added
if not os.path.exists(os.path.join(path,path_metrics)):
    os.mkdir(os.path.join(path,path_metrics))

if not os.path.exists(os.path.join(path,path_images)):
    os.mkdir(os.path.join(path,path_images))


# ## Data preparation

# In[4]:


def create_datasets(batch_size=32, path="./data", num_workers=4):

    # Transformations
    # Crop image 28x28 -> 20x20
    # Always apply ToTensor() after PIL transformations!
    transform = transforms.Compose([transforms.CenterCrop(20),
                                      transforms.ToTensor()])

    # choose the training and test datasets
    train_data = datasets.MNIST(root=path, 
                                train=True,
                                download=True, 
                                transform=transform)

    test_data = datasets.MNIST(root=path,
                               train=False,
                               download=True,
                               transform=transform)
    valid_size = 0.2

    print("Sanity checks : \n")
    print("MNIST training set : ", train_data[0][0].size(), train_data[0][0].min(), train_data[0][0].max())
    print("MNIST testing set : ", test_data[0][0].size(), test_data[0][0].min(), test_data[0][0].max())
    
    
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # load training data in batches
    trainloader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)
    
    # load test data in batches
    testloader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              num_workers=num_workers)
    
    # load validation data in batches
    validloader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)
    
    return trainloader, testloader, validloader


# In[5]:


trainloader, testloader, validloader = create_datasets(batch_size=128)


# ## Autoencoder Unit

# In[6]:


class Autoencoder(nn.Module):
    """Autoencoder class with linear layers"""

    def __init__(self, input_size, hidden_units, dropout=0):
        
        super(Autoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.dropout = dropout
        
        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=self.input_size, out_features=self.hidden_units, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=self.hidden_units, out_features=self.input_size, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# ## Logistic Regression Model

# In[7]:


class LogisticRegressionModel(nn.Module):
    '''Logistic Regressor
    '''

    def __init__(self, in_dim, classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(in_dim, classes)

    def forward(self, x):
        y = self.linear(x) # F.sigmoid(self.linear(x))
        return y


# In[8]:


class LogisticRegressionModel_1(nn.Module):
    """
    Logistic Regressor
    """

    def __init__(self, in_dim, classes, hidden_layers, activation_fn):
        """
        in_dim:         This is the input dimension of the Logistic Classifier
        classes:        The number of classes,i.e. the output dimensions of the classifier
        hidden_layers:  A python list specifying the no. of neurons in each hidden layer. 
                        Length of this list will give the no. of hidden layers.
                        Empty list [] indicates 'no hidden layers'.
        """
        
        super(LogisticRegressionModel_1, self).__init__()
        
        #activation_fn = nn.ReLU(inplace = True)
        n = len(hidden_layers)
        if n > 0:   
            layers = [[nn.Linear(in_dim, hidden_layers[0]), activation_fn]]
            layers.extend([[nn.Linear(hidden_layers[i],hidden_layers[i+1]), activation_fn] 
                 for i in range(n-1)])
        
            # Put everything into a single proper list
            self.linear = [elem for lists in layers for elem in lists]
            self.linear.append(nn.Linear(hidden_layers[n-1], classes))
        
            # Unpack the list arguments into the nn.Sequential container
            self.linear = nn.Sequential(*self.linear)
        else:
            self.linear = nn.Linear(in_dim, classes)   # i.e. no hidden layers
            
        
    def forward(self, x):
        y = self.linear(x) # F.sigmoid(self.linear(x))
        return y


# In[9]:


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# In[10]:


def visualize_earlystop(valid_loss, train_loss):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 1.5) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')


# ## Dynamic subnetwork

# ### Forward Pass

# In[11]:


def forward_pass(name, autoencs, x, train=True, saturate=False):
    
    '''
    The forward pass for each AE unit
    make sure to do things inplace!
    '''

    # define the loss function
    criterion = nn.MSELoss()

    # Clear gradients w.r.t. parameters
    # autoencs[name]['optim'].zero_grad()

    if train or saturate:
        # set to train mode
        autoencs[name]['model'].train()

    else:
        # set to evaluation mode (deactivate dropout, batchnorm etc)
        autoencs[name]['model'].eval()

    # forward pass
    encoded, decoded = autoencs[name]['model'](x)

    # only if training (not saturation)
    if train:
        loss = criterion(decoded, x)
        
        # Store the loss object to be used in backward pass
        autoencs[name]['loss'] = loss
        
        # Store the loss for visualization
        autoencs[name]['train_loss'].append(loss.item())

    return encoded


# ### Backward Pass

# In[12]:


def backward_pass(autoencs):
    
    '''
    Train the subnetwork only once, after the complete secondary pass
    '''    
    for name, attr in autoencs.items():
        
        # Clear gradients w.r.t. parameters
        attr['optim'].zero_grad()

        # backward pass
        attr['loss'].backward()

        # Update the parameters
        attr['optim'].step()


# ### Dynamic AE units

# In[13]:


def create_ae(locs, rfs, patch_sizes, input_dim, hidden_units, dropout, wd, lr, **kwargs):
    
    '''
    Method used for creating Autoencoder units dynamically
    args:
    locs - locations of the encodings in the map
    rfs - locations of the receptive fields
    patch_sizes - [(x, y), ..]
    input_dim - input dimensions for the AE units
    hidden_units - No of hidden units for the AE units
    dropout - if dropout is to be applied
    '''
    
    autoencs = {}

    for i, loc in enumerate(locs):
        
        x, y = patch_sizes[i][0], patch_sizes[i][1]
        model = Autoencoder(input_size=input_dim, 
                            hidden_units=hidden_units,
                            dropout=dropout).cuda()

        optim = torch.optim.Adam(model.parameters(), 
                                    lr=lr,
                                    weight_decay=wd,
                                 )

        # object to store model attributes
        autoencs['ae_'+str(i+1)] = {'model' : model,
                                    'optim' : optim,
                                    'x' : x,
                                    'y' : y,
                                    'rf' : rfs[i],
                                    'loc' : loc,
                                    'train_loss' : [],
                                    'valid_loss' : []}
    return autoencs


# ### Meta-network training loop
# 
# - This is the main loop for the training of the handcrafted architectures
# - All three estimators are included, however we can choose either

# In[14]:


def train(epochs, autoencs, h=20, w=40):

    '''
    Method used for training the Autoencoder
    args:
    epochs - no of epochs
    autoencs - dict containing ae units and their attributes
    h - height of the emap
    w - width of the emap
    '''

    print("===> Training...")
    
    # To store the encoding map -- debugging and viz
    encoding_map = []

    # store all estimated entropies
    entropy = {k : [] for k, v in autoencs.items()}
    
    # Store the mean of the losses for viz
    avg_train_losses = {k : [] for k, v in autoencs.items()}
    
    prev_avg_train_loss_over_all_autoencs = np.inf

    noise_variance = 1e-01

    for epoch in range(epochs):
        
        emap_epoch = []

        for step, (images, labels) in enumerate(trainloader):
            
            # Initialize the encoding map -> (batch_size, width, height)
            map_size = (images.size(0), h, w)
            emap = torch.zeros(map_size, requires_grad=False)

            # Update the map with the images in the mini-batch
            emap[:, :images.size(-1), :images.size(-1)] = images.view(images.size(0), images.size(2), images.size(3))
            
            # Forward passes ---------------------------------------------------
            # For stabilizing the map; not tracking operations for gradients
            with torch.no_grad():
                
                for i in range(len(autoencs.keys())):

                    for name, attr in autoencs.items():
                        
                        # Get the receptive field location i.e. inputs
                        rf = attr['rf']
                        loc = attr['loc']
                        x = attr['x']
                        y = attr['y']

                        # get the input dimensions of AE unit
                        in_dim = autoencs[name]['model'].input_size
                        
                        # get the input batch of images from the map
                        inputs = emap[:, rf[0][0]:rf[0][1], rf[1][0]:rf[1][1]]
                        # get the receptive field size
                        rf_size = inputs.size(1)*inputs.size(2)

                        # Flatten/reshape the inputs
                        inputs = inputs.contiguous().view(inputs.size(0), -1)

                        # condition :- If the receptive field is smaller than the 
                        # input dimensions of the AE unit
                        if rf_size != in_dim:
                            # make the dimensions of the receptive field = input dim of AE
                            # input size expanded by factor = input dim of AE
                            factor = in_dim//rf_size
                            inputs = inputs.repeat(1, factor)

                        encoded = forward_pass(name=name,
                                            autoencs=autoencs,
                                            x=Variable(inputs).cuda(),
                                            train=False,
                                            saturate=True
                                    )
                        
                        # Update the map
                        emap[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] = torch.tensor(encoded.view(images.size(0), x, y).cpu().detach().numpy())

                        # uncomment for debugging --
                        # plt.imshow(emap[0,:,:], cmap='gray');
                        # plt.show();

            # One final forward pass where operations would be tracked
            for name, attr in autoencs.items():
                
                # Get the receptive field location i.e. inputs
                rf = attr['rf']
                loc = attr['loc']
                x = attr['x']
                y = attr['y']
                in_dim = autoencs[name]['model'].input_size
                
                # get the input batch of images from the map
                inputs = emap[:, rf[0][0]:rf[0][1], rf[1][0]:rf[1][1]]
                rf_size = inputs.size(1)*inputs.size(2)

                inputs = inputs.contiguous().view(inputs.size(0), -1)

                # condition :- if the receptive field is smaller than the 
                # input dimensions of the AE unit
                if rf_size != in_dim:
                    # make the dimensions of the receptive field = input dim of AE
                    # input size x n = input dim of AE
                    factor = in_dim//rf_size
                    inputs = inputs.repeat(1, factor)

                encoded = forward_pass(name=name,
                                    autoencs=autoencs,
                                    x=Variable(inputs).cuda(),
                                    train=True,
                                    saturate=False
                            )
                
                # Update the map 
                emap[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] = torch.tensor(encoded.view(images.size(0), x, y).cpu().detach().numpy())

                # uncomment for debugging --
                # print('final pass --')
                # plt.imshow(emap[0,:,:], cmap='gray');
                # plt.show();

            # once the map is stable --
            encoding_map.append(emap)
            emap_epoch.append(emap)
            # Backward passes --------------------------------------------------
            backward_pass(autoencs)

        
        # At the end of each epoch --
        # Estimate entropy using non-parametric methods
        # get the last 128*100 activations (from last 100 minibatches)
        acts_epoch = torch.cat(emap_epoch, dim=0)
        acts = torch.cat(encoding_map[-100:], dim=0)
        
        # Store the train+valid loss for viz at the enc of each epoch
        for idx, (k, v) in enumerate(autoencs.items()):
            
            # Store the mean of all the losses accumulated
            avg_train_losses[k].append(np.mean(autoencs[k]['train_loss']))
            # Clear the previous loss values
            autoencs[k]['train_loss'].clear()

            print(
                "epoch: [{}/{}], unit: {}, train_loss: {:.4f}".format(
                    epoch + 1,
                    epochs,
                    k,
                    avg_train_losses[k][-1]
                )
            )

            loc = v['loc']

            # I. Binned estimator
            x = acts_epoch[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]]
            x = x.reshape(x.size(0), -1)

            h_bin1 = binned_entropy(x)
            h_bin2 = bin_calc_information(x.numpy(), 0.5)
            h_bin3 = calculate_entropy(x).item()
            print('==>Histogram based estimation : [{:.4f},{:.4f},{:.4f}]'.format(h_bin1, h_bin2, h_bin3))

            # for each hidden layer
            x = acts[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]]
            x = x.reshape(x.size(0), -1)

            # II. KNN based entropy estimator
            h_knn = knn_entropy(x.tolist())
            print('==>KNN based estimation : {:.4f}'.format(h_knn.item()))

            # III. KDE
            # Functions to return upper and lower bounds on entropy of layer activity
            h_upper = nats2bits*entropy_estimator_kl(x, noise_variance).item()
            print('==>KDE upper bound : {:.4f}'.format(h_upper))

            h_lower = nats2bits*entropy_estimator_bd(x, noise_variance).item()
            print('==>KDE lower bound : {:.4f}'.format(h_lower.item()))
            
            entropy[k].append({'h_bin1' : h_bin1, 'h_bin2' : h_bin2, 
                               'h_bin3' : h_bin3, 'h_knn' : h_knn, 
                               'h_upper' : h_upper, 'h_lower' : h_lower})
            print('\n')

        print("---------------------------------------------------------------")
        
        
        #Implement Early Stopping for the Autoencs if the difference in re-const. error between previous
        #Epoch and this one is < 0.05
        
        #Contains the training loss of all the autoencs in this epoch
        autoenc_units_train_loss_list = []
        
        for idx, (k, v) in enumerate(autoencs.items()):
            autoenc_units_train_loss_list.append(avg_train_losses[k][-1])
        
        curr_avg_train_loss_over_all_autoencs = np.mean(autoenc_units_train_loss_list)
        print("Average autoencoder reconstruction loss in this epoch, over all the units: {:.4f}".format(curr_avg_train_loss_over_all_autoencs))
        
        if((prev_avg_train_loss_over_all_autoencs - curr_avg_train_loss_over_all_autoencs) < early_stop_ae_diff):
            print("Stopping the autoencoder training early")
            break
        
        else:
            prev_avg_train_loss_over_all_autoencs = curr_avg_train_loss_over_all_autoencs 
            
        print("-----------------------------------------------------------------") 
            
    return avg_train_losses, torch.cat(encoding_map, dim=0), entropy


# ### Logistic Regression Training

# In[47]:


def logit_train(epochs, logit_model, logit_optimizer, logit_criterion, autoencs, h=20, w=40, patience = 0):
    """Method used for training the Logistic Regressor"""

    print("===> Training...")

    avg_loss_values = []
    loss_values = []
    encoding_map = []
    avg_valid_loss_values = []
    valid_loss_values = []
    accuracy = 0.0
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        for step, (images, labels) in enumerate(trainloader):
            
            # Get the labels in the mini batch
            labels = Variable(labels).cuda()
            
            # Initialize the encoding map -> (batch_size, width, height)
            map_size = (images.size(0), h, w)
            emap = torch.zeros(map_size, requires_grad=False)

            # Update the map with the images in the mini-batch
            emap[:, :images.size(-1), :images.size(-1)] = images.view(images.size(0), images.size(2), images.size(3))

            with torch.no_grad():
                # Forward passes ---------------------------------------------------
                for i in range(len(autoencs.keys())):

                    for name, attr in autoencs.items():
                        
                        # Get the receptive field location i.e. inputs
                        rf = attr['rf']
                        loc = attr['loc']
                        x = attr['x']
                        y = attr['y']
                        in_dim = autoencs[name]['model'].input_size
                        
                        # get the input batch of images from the map
                        inputs = emap[:, rf[0][0]:rf[0][1], rf[1][0]:rf[1][1]]
                        rf_size = inputs.size(1)*inputs.size(2)

                        inputs = inputs.contiguous().view(inputs.size(0), -1)

                        # condition :- if the receptive field is smaller than the 
                        # input dimensions of the AE unit
                        if rf_size != in_dim:
                            # make the dimensions of the receptive field = input dim of AE
                            # input size x n = input dim of AE
                            factor = in_dim//rf_size
                            inputs = inputs.repeat(1, factor)

                        encoded = forward_pass(name=name,
                                            autoencs=autoencs,
                                            x=Variable(inputs).cuda(),
                                            train=False,
                                            saturate=False
                                    )

                        # Update the map
                        emap[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] = torch.tensor(encoded.view(images.size(0), x, y).cpu().detach().numpy())
                        
                        # uncomment for debugging --
                        # plt.imshow(emap[0,:,:], cmap='gray');
                        # plt.show();

            # once the map is stable --
            encoding_map.append(emap)
            # Concatenate the encodings
            # To store the encodings each time
            encs = []
            for name, attr in autoencs.items():
                loc = attr['loc']
                inputs = emap[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]]
                encs.append(Variable(inputs.contiguous().view(inputs.size(0), -1)).cuda())

            # concat of all encodings
            enc_cat = torch.cat(tuple(encs), dim=1) # size -> (128, 300)

            # mean of all encodings -- does not increase overhead
            # enc_cat = torch.mean(torch.stack(tuple(encs)), dim=0)

            # ===================forward=====================
            # Using the encodings as input into the logistic regression model
            logit_model.train()
            logit_optimizer.zero_grad() # Clear gradients w.r.t. parameters
            outputs = logit_model(enc_cat)
            loss = logit_criterion(outputs, labels)
            loss_values.append(loss.item())
            
            # ===================backward====================
            loss.backward()
            logit_optimizer.step() # Updating parameters
            
        ######################    
        # validate the model #
        ######################
            
        logit_model.eval() # prep model for evaluation
            
        for step, (images, labels) in enumerate(validloader):
                
            # Get the labels in the mini batch
            labels = Variable(labels).cuda()
            
            # Initialize the encoding map -> (batch_size, width, height)
            map_size = (images.size(0), h, w)
            emap = torch.zeros(map_size, requires_grad=False)
            rf = torch.zeros(map_size, requires_grad=False)

            # Update the map with the images in the mini-batch
            emap[:, :images.size(-1), :images.size(-1)] = images.view(images.size(0), images.size(2), images.size(3))

            with torch.no_grad():
                # Forward passes ---------------------------------------------------
                for i in range(len(autoencs.keys())):

                    for name, attr in autoencs.items():
                        
                        # Get the effective receptive field location i.e. inputs
                        rf = attr['rf']
                        loc = attr['loc']
                        x = attr['x']
                        y = attr['y']
                        in_dim = autoencs[name]['model'].input_size
                        
                        # get the input batch of images from the map
                        inputs = emap[:, rf[0][0]:rf[0][1], rf[1][0]:rf[1][1]]   

                        rf_size = inputs.size(1)*inputs.size(2)

                        inputs = inputs.contiguous().view(inputs.size(0), -1)

                        # condition :- if the effective receptive field is smaller than the 
                        # input dimensions of the AE unit
                        if rf_size != in_dim:
                            # make the dimensions of the effective receptive field = input dim of AE
                            # input size x n = input dim of AE
                            factor = in_dim//rf_size
                            inputs = inputs.repeat(1, factor)

                        encoded = forward_pass(name=name,
                                            autoencs=autoencs,
                                            x=Variable(inputs).cuda(),
                                            train=False,
                                            saturate=False
                                    )

                        # Update the map
                        emap[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] = torch.tensor(encoded.view(images.size(0), x, y).cpu().detach().numpy())
                        
                        # uncomment for debugging --
                        # plt.imshow(emap[0,:,:], cmap='gray');
                        # plt.show();

            # once the map is stable --
            encoding_map.append(emap)
            # Concatenate the encodings
            # To store the encodings each time
            encs = []
            for name, attr in autoencs.items():
                loc = attr['loc']
                inputs = emap[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]]
                encs.append(Variable(inputs.contiguous().view(inputs.size(0), -1)).cuda())

            # concat of all encodings
            enc_cat = torch.cat(tuple(encs), dim=1) # size -> (128, 300)

            outputs = logit_model(enc_cat)
            # calculate the loss
            valid_loss = logit_criterion(outputs, labels)
            # record validation loss
            valid_loss_values.append(valid_loss.item())

        # end of 1 epoch
        accuracy = logit_test(logit_model, autoencs, h, w)
        
        avg_loss_values.append(np.mean(loss_values))
        avg_valid_loss_values.append(np.mean(valid_loss_values))
        
        loss_values.clear()
        valid_loss_values.clear()

        # ===================log========================
        print(
            "epoch [{}/{}], loss:{:.4f}, avg-valid-loss:{:.4f}, accuracy:{:.4f}".format(
                epoch + 1,
                epochs,
                avg_loss_values[-1],
                avg_valid_loss_values[-1],
                accuracy
            )
        )
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_loss_values[-1], logit_model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            visualize_earlystop(avg_valid_loss_values, avg_loss_values)
            break
        
    # load the last checkpoint with the best model
    logit_model.load_state_dict(torch.load('checkpoint.pt'))
    accuracy = logit_test(logit_model, autoencs, h, w)

    # find position of lowest validation loss
    minposs = avg_valid_loss_values.index(min(avg_valid_loss_values))
    avg_valid_loss_values = avg_valid_loss_values[:minposs]
    avg_loss_values = avg_loss_values[:minposs]
    
    return avg_valid_loss_values, avg_loss_values, accuracy # torch.cat(encoding_map, dim=0)
    #return avg_loss_values, accuracy # torch.cat(encoding_map, dim=0)


# ### Logistic Regression Testing

# In[16]:


def logit_test(logit_model, autoencs, h=20, w=40):
    """Method used for training the Logistic Regressor"""

    y_test, y_pred = [], []

    with torch.no_grad(): # Not computing gradients during test (for memory efficiency)
        tp_tn = 0  # true positives and true negatives
        total = 0
        for images, labels in testloader:
            
            # Get the labels in the mini batch
            labels = Variable(labels).cuda()
            
            # Initialize the encoding map -> (batch_size, width, height)
            map_size = (images.size(0), h, w)
            emap = torch.zeros(map_size, requires_grad=False)

            # Update the map with the images in the mini-batch
            emap[:, :images.size(-1), :images.size(-1)] = images.view(images.size(0), images.size(2), images.size(3))

            with torch.no_grad():
                # Forward passes ---------------------------------------------------
                for i in range(len(autoencs.keys())):

                    for name, attr in autoencs.items():
                        
                        # Get the receptive field location i.e. inputs
                        rf = attr['rf']
                        loc = attr['loc']
                        x = attr['x']
                        y = attr['y']
                        in_dim = autoencs[name]['model'].input_size
                        
                        # get the input batch of images from the map
                        inputs = emap[:, rf[0][0]:rf[0][1], rf[1][0]:rf[1][1]]
                        rf_size = inputs.size(1)*inputs.size(2)

                        inputs = inputs.contiguous().view(inputs.size(0), -1)

                        # condition :- if the receptive field is smaller than the 
                        # input dimensions of the AE unit
                        if rf_size != in_dim:
                            # make the dimensions of the receptive field = input dim of AE
                            # input size x n = input dim of AE
                            factor = in_dim//rf_size
                            inputs = inputs.repeat(1, factor)

                        encoded = forward_pass(name=name,
                                            autoencs=autoencs,
                                            x=Variable(inputs).cuda(),
                                            train=False,
                                            saturate=False
                                    )

                        # Update the map
                        emap[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] = torch.tensor(encoded.view(images.size(0), x, y).cpu().detach().numpy())
                        
                        # uncomment for debugging --
                        # plt.imshow(emap[0,:,:], cmap='gray');
                        # plt.show();

            # once the map is stable --
            # Concatenate the encodings
            # To store the encodings each time
            encs = []
            for name, attr in autoencs.items():
                loc = attr['loc']
                inputs = emap[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]]
                encs.append(Variable(inputs.contiguous().view(inputs.size(0), -1)).cuda())
            
            # Concatenate the encodings
            enc_cat = torch.cat(tuple(encs), dim=1) # size -> (128, 300)
            # mean of the encodings
            # enc_cat = torch.mean(torch.stack(tuple(encs)), dim=0)

            logit_model.eval()
            outputs = logit_model(enc_cat)

            # Prediction
            _, predicted = torch.max(outputs.data, 1)
            y_test.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
            total += labels.size(0)
            tp_tn += (predicted == labels).sum()

        accuracy = tp_tn.item() / total * 100
        return accuracy


# ## Intrinsic Metrics

# Normalization of the hidden layer activations :
# 
# $$X_{sc} = \frac{X - X_{min}}{X_{max} - X_{min}}.$$

# In[17]:


def normalize(x):
    # scale values between 0 - 1
    max = torch.max(x)
    min = torch.min(x)
    x = (x - min)/(max - min)
    return x


# ### I. Histogran (binning-based) entropy estimator
# 

# - approach used for the final experiments/evaluation
# - returns bits

# In[18]:


def binned_entropy(x, bins=100):
   
   x = normalize(x)
   hist, _ = np.histogram(x,bins) # discretize into bins and get the pdf
   h = scipy.stats.entropy(pk=hist, base=2)  # or shan_entropy(hist)
   return h


# Alternative :

# In[19]:


def shan_entropy(c):
    # calculating the entropy (just like the formula)
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))
    return H


# Second approach

# In[20]:


# Simplified entropy computation code from https://github.com/ravidziv/IDNNs
import numpy as np

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

def bin_calc_information(layerdata, binsize):
    # This is even further simplified, where we use np.floor instead of digitize
    # returns bits
    def get_h(d):
        digitized = np.floor(d / binsize).astype('int')
        p_ts, _ = get_unique_probs( digitized )
        return -np.sum(p_ts * np.log2(p_ts))

    H_LAYER = get_h(layerdata)
    return H_LAYER


# In[21]:


def calc_information_for_layer(data, bins):

    digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    b2 = np.ascontiguousarray(digitized).view(
        np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts =         np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(sum(unique_counts))
    H = -np.sum(p_ts * np.log2(p_ts))
    return H


# Original approach

# In[22]:


def calculate_entropy(x):
    
    # Normalize the activations i.e. [0,1]
    x = normalize(x)
    
    # calculate entropy per batch
    a, b = torch.unique(x, return_counts=True); # flattens the tensor before unique
    p = b.to(dtype=torch.float) / float(x.size(0)*x.size(1)) # Compute probabilities
    
    # alternatively torch.distributions.Categorical(probs=p).entropy()
    return -torch.where(p > 0, p * p.log2(), p.new([0.0])).sum()
    


# ### II. KNN-based Estimator
# 
# - source : https://github.com/gregversteeg/NPEET
# - `entropy()` from NPEET to estimate continuous entropy
# - returns bits

# In[23]:


#!/usr/bin/env python
# Written by Greg Ver Steeg
# See readme.pdf for documentation
# Or go to http://www.isi.edu/~gregv/npeet.html

import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy as np
import random
import warnings

# CONTINUOUS ESTIMATORS

def knn_entropy(x, k=3, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = ss.cKDTree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * log(2)
    return (const + n_features * np.log(nn).mean()) / log(base)

# UTILITY FUNCTIONS
def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1, p=float('inf'), n_jobs=-1)[0][:, k]


# ### III. Kernel Density Estimator (KDE)
# 
# - source : https://github.com/artemyk/ibsgd/blob/master/kde.py
# - Re-implemented in PyTorch
# - returns nats

# In[24]:


nats2bits = 1.0/np.log(2) # for conversion


# In[25]:


# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """

    # return torch.tensor(sklearn.metrics.pairwise_distances(X, n_jobs=4))
    # return torch.pairwise_distance(X, X, p=2)
    # return torch.pow(X - X, 2).sum(2) 
    return pairwise_distances(X)
    
def get_shape(x):
    dims = x.size(1)
    N = x.size(0)
    return dims, N

def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = get_shape(x)
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
    
    normconst = (dims/2.0)*np.log(2*np.pi*var)
    lprobs = torch.logsumexp(-dists2, dim=1) - np.log(N) - normconst
    h = -torch.mean(lprobs)
    return dims/2 + h

def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    dims, N = get_shape(x)
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.size(1)
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)


# ### Compute entropies

# In[26]:


def get_h_binned(units, arch, run, emap, locs):

    h_bin1, h_bin2, h_bin3 = [], [], []

    # for each ae unit
    for idx, loc in enumerate(locs):

        # for each hidden layer
        x = emap[:, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]]
        x = x.reshape(x.size(0), -1) # reshape (60000x10x10)-> (60000x100)
        
        # I. Binned estimator
        h_bin1.append(binned_entropy(x))
        h_bin2.append(bin_calc_information(x.numpy(), 0.5))
        h_bin3.append(calculate_entropy(x).item())
        print('==>binned entropy estimation complete for unit : {}'.format(idx+1))
        
    results = {'h_bin1' : h_bin1, 'h_bin2' : h_bin2, 'h_bin3' : h_bin3}

    fname = str(units) + '-' + str(arch) + '-' + str(run) + '-' + 'bin' + '.json'
    with open(path/path_metrics/fname, 'w') as f:
        json.dump(results, f)

    print('==>results saved')


# save the results

# In[27]:


def save_entropy(entropy, units, arch, run):
    fname = str(units) + '-' + str(arch) + '-' + str(run) + '-' + 'entropy' + '.json'
    with open(path/path_metrics/fname, 'w') as f:
        json.dump(entropy, f)


# ### Exporting results

# In[28]:


# set folder names here
#path_metrics = 'newer-metrics'
#path_images = 'newer-images'


# In[29]:


def export_results(units, arch, run, ae_loss, logit_loss, accuracy):

    metrics = OrderedDict()

    metrics['units'] = units
    metrics['type'] = arch
    metrics['run'] = run
    metrics['ae_loss'] = ae_loss
    metrics['logit_loss'] = logit_loss
    metrics['accuracy'] = accuracy
    fname = str(units) + '-' + str(arch) + '-' + str(run) + '-' + 'metrics'
    
    json_fname = fname + '.json'

    # save metrics
    with open(path/path_metrics/json_fname, 'w') as f:
        json.dump(metrics, f)

    # tensor_fname = fname + '.pt'
    # with open(path/'metrics'/'3-1-1.json', 'r') as fp:
    #     data = json.load(fp)
    
    # if save_act:
    #     # save the encoding map
    #     torch.save(activities, path/'activations'/tensor_fname)
    #     # torch.load('file.pt')


# ## Plots

# In[30]:


def plot_train_loss(train_loss, units, arch, run):
   plt.figure(figsize=(20,10))
   for k, v in train_loss.items():
       plt.plot(v);
   
   # plt.xlim([1, 5])
   # plt.plot(losses);
   plt.title("Loss curves")
   plt.xlabel("Epoch")
   plt.ylabel("Loss")
   plt.legend([name for name, attr in train_loss.items()]);
   fname = str(units) + '-' + str(arch) + '-' + str(run) + '-' + 'ae_loss' + '.png'
   plt.savefig(path/path_images/fname)
   plt.show()


# In[31]:


def plot_logistic_loss(losses, units, arch, run):
    plt.figure(figsize=(20,10))
    plt.title("Loss curves")
    # plt.xlim([1, 5])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses);
    fname = str(units) + '-' + str(arch) + '-' + str(run) + '-' + 'logit_loss' + '.png'
    plt.savefig(path/path_images/fname)
    plt.show();


# In[32]:


def plot_emap(emap, units, arch, run, from_end=False):

    fig = plt.figure(figsize=(20,8))
    for i in range(6):
        plt.subplot(2,3,i+1)
        # plt.tight_layout()
        if from_end:
            plt.imshow(emap[-1*i, :, :], cmap='gray', interpolation='none')
        else:
            plt.imshow(emap[i, :, :], cmap='gray', interpolation='none')
    # save the image
    fname = 'end' if from_end else 'early'
    fname = str(units) + '-' + str(arch) + '-' + str(run) + '-' + fname + '.png'
    plt.savefig(path/path_images/fname)
    fig.show();


# ## Handcrafted Topologies

# In[48]:


def train_network(topologies, locs, ae_units, in_dim, classes, h, w, patch_sizes, lr, wd, dropout):
    
    for topo in topologies:
        
        print('------------------------------------\n')
        arch = topo['arch']
        
        # for each run 1-3
        for run in range(1,11):

            print('==>arch/run : [{}/{}]'.format(arch, run))
            # receptive fields
            rfs = topo['rfs']

            # Initalizing ae units
            autoencs = create_ae(locs, rfs, patch_sizes=patch_sizes, 
                                 input_dim=topo['input_dim'], hidden_units=100, 
                                 dropout=dropout, wd=wd, lr=lr)

            # train autoencoders
            ae_loss, emap, entropy = train(epochs_ae, autoencs, h=h, w=w)

            # save the non-parametric estimated entropy values
            save_entropy(entropy, ae_units, arch, run)
            
            # calculate binned entropy estimate (for all epochs)
            get_h_binned(ae_units, arch, run, emap, locs)

            # plot emap
            plot_emap(emap, ae_units, arch, run, from_end=False)
            plot_emap(emap, ae_units, arch, run, from_end=True)

            # plot training losses for AE
            plot_train_loss(ae_loss, ae_units, arch, run)

            #logit_model = LogisticRegressionModel(in_dim, classes).cuda()
            logit_model = LogisticRegressionModel_1(
            in_dim, classes, hidden_layers, nn.ReLU(inplace = True)).cuda()
            
            logit_criterion = nn.CrossEntropyLoss()
            logit_optimizer = torch.optim.Adam(logit_model.parameters(), lr=lr, weight_decay=wd)

            valid_losses, logit_loss, accuracy = logit_train(epochs_logit, logit_model, logit_optimizer, logit_criterion, autoencs, h=h, w=w, patience=early_stop_pat)

            plot_logistic_loss(logit_loss, ae_units, arch, run)
            
            # export and save results
            export_results(ae_units, arch, run, ae_loss, logit_loss, accuracy)


# In[43]:


early_stop_ae_diff = 0.005
early_stop_pat = 3
epochs_ae = 5
epochs_logit = 50


# ### Type A topologies
# 
# Hyperparameters :
# 1. No of sub-units = 3
# 2. Input dimensions = 200/400
# 3. Hidden units = 100
# 4. Map size = 20 x 40
# 

# ### Hyperparameters

# In[44]:


ae_units = 3
in_dim = ae_units * 10 * 10
classes = 10
patch_sizes = [(10,10) for i in range(ae_units)]
lr = 1e-02
wd = 1e-04
dropout = 0.2
h = 20
w = 40


test_dim = in_dim // 10
hidden_layers = [test_dim]


# ### Architectures

# In[45]:


# location of encodings on the map
locs = [[(0,10),(20,30)],[(10,20),(20,30)],[(0,10),(30,40)]]

topologies = [
              {'arch' : 1, 
               'name' : 'flat',
               'input_dim' : 200,
               'rfs' : [[(0,20),(0,10)],[(0,10),(0,20)],[(0,20),(10,20)]]
               },
              {'arch' : 2, 
               'name' : 'columnar',
               'input_dim' : 200,
               'rfs' : [[(0,20),(5,15)],[(0,10),(20,30)],[(10,20),(20,30)]]
               },
              {'arch' : 3,
               'name' : 'pyramidal',
               'input_dim' : 200,
               'rfs' : [[(0,20),(0,10)],[(0,20),(10,20)],[(0,20),(20,30)]]
               },
               {'arch' : 4,
               'name' : 'random-1',
               'input_dim' : 200,
               'rfs' : [[(4, 14), (0, 20)], [(0, 10), (16, 36)], [(5, 15), (16, 36)]] 
               },
              {'arch' : 5,
               'name' : 'random-2',
               'input_dim' : 200,
               'rfs' : [[(0, 20), (13, 23)], [(0, 10), (2, 22)], [(0, 20), (13, 23)]] 	
               },
              {'arch' : 6,
               'name' : 'random-3',
               'input_dim' : 200,
               'rfs' : [[(4, 14), (8, 28)], [(0, 10), (16, 36)], [(0, 20), (14, 24)]] 
               }
              ]


# ### Training

# In[49]:



import time
start = time.time()

train_network(topologies, locs, ae_units, in_dim, classes, h, w, patch_sizes, lr, wd, dropout)

time_minutes = (time.time()-start)//60
print("time to run = ", time_minutes//60, " hours and ", time_minutes % 60, " minutes.")


# ### Type B topologies
# 
# Hyperparameters :
# 1. No of sub-units = 5
# 2. Input dimensions = 200
# 3. Hidden units = 100
# 4. Map size = 20 x 50

# ### Hyperparameters

# In[ ]:


ae_units = 5
in_dim = ae_units * 10 * 10
classes = 10
patch_sizes = [(10,10) for i in range(ae_units)]
lr = 1e-02
wd = 1e-04
dropout = 0.2
h = 20
w = 50

test_dim = in_dim // 10
hidden_layers = [test_dim]


# In[ ]:


# location of encodings on the map
locs = [[(0,10),(20,30)],[(10,20),(20,30)],[(0,10),(30,40)],[(10,20),(30,40)],[(0,10),(40,50)]]

topologies = [
              {'arch' : 1,
               'name' : 'flat',
               'input_dim' : 200,
               'rfs' : [[(0,10),(0,20)],[(0,20),(0,10)],[(10,20),(0,20)],[(10,20),(0,20)],[(0,20),(10,20)]]
               },
              {'arch' : 2, 
               'name' : 'x-shaped',
               'input_dim' : 200,
               'rfs' : [[(0,20),(0,10)],[(0,20),(10,20)],[(0,20),(20,30)],[(0,10),(30,40)],[(0,10),(30,40)]]
               },
              {'arch' : 3, 
               'name' : 'columnar',
               'input_dim' : 200,
               'rfs' : [[(0,20),(5,15)],[(0,10),(20,30)],[(10,20),(20,30)],[(0,10),(30,40)],[(10,20),(30,40)]]
               },
              {'arch' : 4, 
               'name' : 'inverted-y',
               'input_dim' : 200,
               'rfs' : [[(0,20),(0,10)],[(0,20),(10,20)],[(0,20),(20,30)],[(0,10),(30,40)],[(10,20),(30,40)]]
               },
              {'arch' : 5,
               'name' : 'asymmetric-pyramidal',
               'input_dim' : 200,
               'rfs' : [[(0,20),(0,10)],[(0,10),(0,20)],[(0,20),(10,20)],[(0,20),(20,30)],[(0,20),(30,40)]]
               },
              {'arch' : 6,
               'name' : 'random-1',
               'input_dim' : 200,
               'rfs' : [[(0,20),(6,16)],[(7,17),(10,30)],[(0,20),(24,34)],[(6,16),(10,30)],[(8,18),(30,50)]]
               },
              {'arch' : 7,
               'name' : 'random-2',
               'input_dim' : 200,
               'rfs' : [[(6,16),(13,33)],[(1,11),(17,37)],[(0,20),(4,14)],[(10,20),(30,50)],[(4,14),(20,40)]]
               },
              {'arch' : 8,
               'name' : 'random-3',
               'input_dim' : 200,
               'rfs' : [[(1,11),(12,32)],[(5,15),(8,28)],[(8,18),(27,47)],[(0,20),(25,35)],[(10,20),(31,41)]]
               }
              ]


# In[ ]:



import time
start = time.time()

train_network(topologies, locs, ae_units, in_dim, classes, h, w, patch_sizes, lr, wd, dropout)

time_minutes = (time.time()-start)//60
print("time to run = ", time_minutes//60, " hours and ", time_minutes % 60, " minutes.")


# ### Type C topologies
# 
# Hyperparameters :
# 1. No of sub-units = 7
# 2. Input dimensions = 200
# 3. Hidden units = 100
# 4. Map size = 20 x 60

# ### Hyperparameters

# In[ ]:


ae_units = 7
in_dim = ae_units * 10 * 10
classes = 10
patch_sizes = [(10,10) for i in range(ae_units)]
lr = 1e-02
wd = 1e-04
dropout = 0.2
h = 20
w = 60

test_dim = in_dim // 10
hidden_layers = [test_dim]


# In[ ]:


# location of encodings on the map
locs = [[(0,10),(20,30)],
        [(10,20),(20,30)],
        [(0,10),(30,40)],
        [(10,20),(30,40)],
        [(0,10),(40,50)],
        [(10,20),(40,50)],
        [(0,10),(50,60)]
        ]

topologies = [
                {'arch' : 1,
               'name' : 'flat',
               'input_dim' : 200,
               'rfs' : [[(0,10),(0,20)],
                        [(0,20),(10,20)],
                        [(10,20),(0,20)],
                        [(0,20),(10,20)],
                        [(0,20),(0,10)],
                        [(10,20),(0,20)],
                        [(0,20),(0,10)]
                        ]
               },
              {'arch' : 2, 
               'name' : 'columnar',
               'input_dim' : 200,
               'rfs' : [[(0,20),(5,15)],
                        [(0,10),(20,30)],
                        [(10,20),(20,30)],
                        [(0,10),(30,40)],
                        [(10,20),(30,40)],
                        [(0,10),(40,50)],
                        [(10,20),(40,50)]
                        ]
               },
              {'arch' : 3, 
               'name' : 'symmetric-pyramidal',
               'input_dim' : 200,
               'rfs' : [[(0,20),(0,10)],
                        [(0,10),(0,20)],
                        [(0,20),(10,20)],
                        [(10,20),(0,20)],
                        [(0,20),(20,30)],
                        [(0,20),(30,40)],
                        [(0,20),(40,50)]
                        ]

               },
              {'arch' : 4,
               'name' : 'asymmetric-pyramidal',
               'input_dim' : 200,
               'rfs' : [[(0,20),(0,10)],
                        [(0,10),(0,20)],
                        [(0,20),(10,20)],
                        [(10,20),(0,20)],
                        [(0,20),(20,30)],
                        [(0,10),(30,50)],
                        [(10,20),(30,50)]
                        ]
               },
              {'arch' : 5,
               'name' : 'semi-x',
               'input_dim' : 200,
               'rfs' : [[(0,20),(0,10)],
                        [(0,20),(10,20)],
                        [(0,20),(20,30)],
                        [(0,10),(30,40)],
                        [(10,20),(30,40)],
                        [(0,10),(40,50)],
                        [(0,10),(40,50)]
                        ]
               },
              {'arch' : 6,
               'name' : 'inverted-y',
               'input_dim' : 200,
               'rfs' : [[(0,20),(0,10)],
                        [(0,20),(10,20)],
                        [(0,20),(20,30)],
                        [(0,10),(30,40)],
                        [(10,20),(30,40)],
                        [(0,10),(40,50)],
                        [(10,20),(40,50)]
                        ]
               },
              {'arch' : 7,
               'name' : 'inverted-tuning-fork',
               'input_dim' : 200,
               'rfs' : [[(0,20),(0,10)],
                        [(0,20),(10,20)],
                        [(0,10),(20,30)],
                        [(10,20),(20,30)],
                        [(0,20),(30,40)],
                        [(0,10),(40,50)],
                        [(10,20),(40,50)]
                        ]
               },
              {'arch' : 8,
               'name' : 'inverted-u',
               'input_dim' : 200,
               'rfs' : [[(0,20),(0,10)],
                        [(0,20),(10,20)],
                        [(0,10),(20,30)],
                        [(10,20),(20,30)],
                        [(0,10),(30,40)],
                        [(10,20),(30,40)],
                        [(0,20),(40,50)]
                        ]
               },
              {'arch' : 9,
               'name' : 'semi-columnar',
               'input_dim' : 200,
               'rfs' : [[(0,20),(0,10)],
                        [(0,20),(10,20)],
                        [(10,20),(0,20)],
                        [(0,10),(20,30)],
                        [(10,20),(20,40)],
                        [(0,10),(30,50)],
                        [(10,20),(40,50)]
                        ]
               },
              {'arch' : 10,
               'name' : 'random-1',
               'input_dim' : 200,
               'rfs' : [[(0,10),(35,55)],
                        [(0,20),(45,55)],
                        [(10,20),(12,32)],
                        [(10,20),(4,24)],
                        [(9,19),(34,54)],
                        [(3,13),(30,50)],
                        [(0,20),(12,22)]
                        ]
               },
              {'arch' : 11,
               'name' : 'random-2',
               'input_dim' : 200,
               'rfs' : [[(2,12),(32,52)],
                        [(7,17),(8,28)],
                        [(9,19),(0,20)],
                        [(6,16),(37,57)],
                        [(0,20),(11,21)],
                        [(3,13),(21,41)],
                        [(10,20),(33,43)]
                        ]
               },
              {'arch' : 12,
               'name' : 'random-3',
               'input_dim' : 200,
               'rfs' : [[(6,16),(26,46)],
                        [(0,20),(34,44)],
                        [(10,20),(30,50)],
                        [(8,18),(37,57)],
                        [(9,19),(23,43)],
                        [(0,10),(11,31)],
                        [(8,18),(38,58)]
                        ]
               }
              ]


# ### Training

# In[ ]:



import time
start = time.time()

train_network(topologies, locs, ae_units, in_dim, classes, h, w, patch_sizes, lr, wd, dropout)

time_minutes = (time.time()-start)//60
print("time to run = ", time_minutes//60, " hours and ", time_minutes % 60, " minutes.")


# ## Note :
# - Only the **binned entropy** estimation is used for final analysis due to its simplicity and significantly small overhead on training. The trend between the three estimators are more or less consistent.
