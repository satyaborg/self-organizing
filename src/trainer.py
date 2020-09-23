# general imports
import os 
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# torch imports
import torch
import torch.nn as nn
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter
from fastprogress import progress_bar
from torchsummary import summary

# custom imports
from src.models import Autoencoder, LogisticRegressionModel
from src.dataset import create_datasets, create_dataloaders
from src.utils import *
from src.estimator import binned_entropy, get_h_binned

logger = logging.getLogger("train_log")

class Trainer(object):
    def __init__(self, **config):
        # general
        self.seed = config.get("seed", 42)
        self.debug = config.get("debug")
        self.dataset = config.get("dataset")
        self.writer = SummaryWriter()
        self.device = torch.device("cuda" if (config.get("device") == "cuda" and torch.cuda.is_available()) else "cpu")
        self.early_stop = config.get("early_stop")
        self.n_classes = config.get("n_classes")
        self.channels = config.get("channels", 1)
        self.training_mode = config.get("training_mode", "handcrafted")
        
        # paths
        self.filename = config["filename"]
        self.paths = config["paths"]
        self.results_path = os.path.join(config["paths"].get("results"), self.filename)

        # AE network
        self.h = 0
        self.w = 0
        self.autoencs = {}
        self.start_epoch = 0
        self.hyperparams = config["hyperparams"].get("ae_network")
        self.epochs = self.hyperparams.get("epochs", 10)
        self.hidden_units = config.get("hidden_units") * self.channels
        self.n_units = config.get("n_units")
        self.patch_size = config.get("patch_size")
        self.runs = config.get("runs")
        self.ae_in_dim = 2 * (self.patch_size**2) * self.channels # effective receptive field size
        self.bins = config.get("bins")
        
        # classifier
        self.classifier_hyperparams = config["hyperparams"].get("classifier")
        self.classifier_epochs = self.classifier_hyperparams.get("epochs")
        self.classifier_in_dim = self.n_units * self.hidden_units
        self.num_layers = self.classifier_hyperparams.get("num_layers", 1) # 1: vanilla

    def prepare_data(self):
        """Prepare the datasets and dataloaders
        
        Returns:
            [type]: [description]
        """
        datasets = create_datasets(self.dataset, self.paths.get("data"), self.channels)
        trainloader, testloader, validloader = create_dataloaders(datasets, 
                                                        batch_size=self.hyperparams.get("batch_size"), 
                                                        valid_pct=self.hyperparams.get("valid_pct"), 
                                                        num_workers=self.hyperparams.get("num_workers")
                                                    )
        return trainloader, testloader, validloader

    def update_emap(self, emap, train=True, saturate=False, criterion=nn.MSELoss):
        """Updates the encoding map after forward passes

        Args:
            emap ([type]): [description]
            train (bool, optional): [description]. Defaults to True.
            saturate (bool, optional): [description]. Defaults to False.
            criterion ([type], optional): [description]. Defaults to nn.MSELoss.

        Returns:
            [type]: [description]
        """
        for name, attr in self.autoencs.items():
            # Get the receptive field location i.e. inputs
            rf, loc, x, y = attr['rf'], attr['loc'], attr['x'], attr['y']
            # get the input dimensions of AE unit
            in_dim = self.autoencs[name]['model'].input_size
            # get the input batch of images from the map
            inputs = emap[:, :, rf[0][0]:rf[0][1], rf[1][0]:rf[1][1]] # torch.Size([128, 3, 16, 64])
            # get the receptive field size
            rf_size = inputs.size(2)*inputs.size(3)*self.channels # review 
            # flatten/reshape the inputs
            inputs = inputs.contiguous().view(inputs.size(0), -1) # review
            # condition :- If the receptive field is smaller than the 
            # input dimensions of the AE unit
            if rf_size != in_dim:
                # make the dimensions of the receptive field = input dim of AE
                # input size expanded by factor = input dim of AE
                factor = in_dim//rf_size
                inputs = inputs.repeat(1, factor)

            encoded = self.forward_pass(name=name,
                                        x=inputs.to(self.device), # Variable
                                        criterion=criterion,
                                        train=train,
                                        saturate=saturate
                                        )
            # Update the emap
            emap[:, :, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] = torch.tensor(encoded.view(emap.size(0), emap.size(1), x, y).cpu().detach().numpy())
            if self.debug:
                plt.imshow(emap[0,0,:,:], cmap='gray')
                plt.show()
        return emap

    def init_emap(self, images):
        """Initializes the encoding map with the input images

        Args:
            images (torch.tensor): Input mini-batch images

        Returns:
            emap (torch.tensor) : Encoding map to store the activations
        """
        # Initialize the encoding map -> (batch_size, channel, height, width)
        map_size = (images.size(0), images.size(1), self.h, self.w)
        emap = torch.zeros(map_size, requires_grad=False)
        # Update the map with the images in the mini-batch
        emap[:, :, :images.size(-1), :images.size(-1)] = images # images.view(images.size(0), images.size(2), images.size(3))
        # emap = torch.squeeze(emap) # to handle both multi-channel and grayscale
        return emap

    def forward_pass(self, name, x, criterion=nn.MSELoss, train=True, saturate=False):
        """Forward pass for each AE unit

        Args:
            name (str): [description]
            x ([type]): [description]
            criterion ([type]): [description]
            train (bool, optional): [description]. Defaults to True.
            saturate (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: hidden layer encoding
        """
        if train or saturate:
            self.autoencs[name]['model'].train() # set to train mode
        else:
            self.autoencs[name]['model'].eval() # set to eval
        encoded, decoded = self.autoencs[name]['model'](x) # forward pass
        if train: # only if training (not saturation)
            # compute loss
            loss = criterion(decoded, x)
            # Store the loss object to be used in backward pass
            self.autoencs[name]['loss'] = loss
            # Store the loss for visualization
            self.autoencs[name]['train_loss'].append(loss.item())
        return encoded
    
    def backward_pass(self):
        """Backward pass for all AE units"""    
        for _, attr in self.autoencs.items():
            attr['optimizer'].zero_grad() # Clear gradients w.r.t. parameters
            attr['loss'].backward()# backward pass
            attr['optimizer'].step() # Update the parameters

    def create_network(self, locs, rfs, **kwargs):
        """
        Method used for creating Autoencoder units dynamically
        args:
        locs - locations of the encodings in the map
        rfs - locations of the receptive fields
        patch_sizes - [(x, y), ..]
        input_dim - input dimensions for the AE units
        hidden_units - No of hidden units for the AE units
        dropout - if dropout is to be applied
        """
        self.autoencs = {} # clear all autoencoders
        for i, loc in enumerate(locs):
            x, y = self.patch_size, self.patch_size # a square patch
            model = Autoencoder(input_size=self.ae_in_dim, 
                                hidden_units=self.hidden_units,
                                dropout=self.hyperparams.get("dropout"))
            model.to(self.device)
            
            optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), 
                                        lr=self.hyperparams.get("lr"),
                                        weight_decay=self.hyperparams.get("wd"),
                                    )
            # optional
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
            #     T_max=self.hyperparams.get("t_max")
            # )
            self.autoencs['ae_'+str(i+1)] = dict(model=model,
                                            optimizer=optimizer,
                                            # scheduler=scheduler, # optional
                                            x=x,
                                            y=y,
                                            rf=rfs[i],
                                            loc=loc,
                                            train_loss=[], # over epoch
                                            valid_loss=[] # over epoch
                                            )

    def feature_extract(self, images):
        """Extract features using trained meta-network

        Args:
            images (torch.tensor): Mini-batch of input images

        Returns:
            enc_cat (torch.tensor) : Concatenated encodings from all AE units 
        """
        with torch.no_grad():
            emap = self.init_emap(images)
            for _ in range(self.n_units):
                emap = self.update_emap(emap, train=False, saturate=False)
            # after the emap is stable: concatenate the encodings
            encs = []
            for _, attr in self.autoencs.items():
                loc = attr['loc']
                inputs = emap[:, :, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]]
                encs.append(inputs.contiguous().view(inputs.size(0), -1).to(self.device))

            # concat of all encodings
            enc_cat = torch.cat(tuple(encs), dim=1) # size -> (128, 300)
        return enc_cat

    # @profile
    def train_classifier(self, trainloader, validloader, testloader, patience=0, **kwargs):
        """Method used for training the Logistic Regressor
        epochs, logit_model, logit_optimizer, logit_criterion, autoencs, h=20, w=40, patience = 0
        """
        logger.info("===> Training classifier...")

        avg_loss_values = []
        loss_values = []
        # encoding_map = []
        avg_valid_loss_values = []
        valid_loss_values = []
        accuracy = 0.0
        filepath = os.path.join(self.paths.get("checkpoint"), self.filename, kwargs.get("prefix") + '-classifier-' + 'checkpoint.pt')
        early_stopping = EarlyStopping(checkpoint_path=filepath, 
                                    patience=patience, verbose=True)
        if self.num_layers < 1:
            raise Exception("The classifier has at least 1 layer, ie. the output layer. Hence, num_layers >=1.")
        else:
            hidden_layers = [self.classifier_in_dim//(10**(i+1)) for i in range(self.num_layers - 1)]
        classifier = LogisticRegressionModel(self.classifier_in_dim,
                                            self.n_classes, 
                                            hidden_layers, 
                                            nn.ReLU(inplace = True)
                                            )
        classifier.to(self.device)
        classifier_criterion = nn.CrossEntropyLoss()
        classifier_optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, classifier.parameters()), 
                                                    lr=self.classifier_hyperparams.get("lr"), 
                                                    weight_decay=self.classifier_hyperparams.get("wd")
                                                    )
        for epoch in range(self.classifier_epochs):
            classifier.train()
            for images, labels in progress_bar(trainloader):
                # feature extraction with trained meta-network
                # Get the labels in the mini batch
                labels = labels.to(self.device)
                enc_cat = self.feature_extract(images)
                # mean of all encodings -- does not increase overhead
                # enc_cat = torch.mean(torch.stack(tuple(encs)), dim=0)
                # ===================forward=====================
                # Using the encodings as input into the classifier
                classifier_optimizer.zero_grad() # Clear gradients w.r.t. parameters
                outputs = classifier(enc_cat)
                loss = classifier_criterion(outputs, labels)
                loss_values.append(loss.item())
                # ===================backward====================
                loss.backward()
                classifier_optimizer.step() # Updating parameters

            ######################    
            # validate the model #
            ######################
            logger.info("==> Validating ...")
            classifier.eval() # evaluation mode
            with torch.no_grad():
                for images, labels in progress_bar(validloader):
                    # Get the labels in the mini batch
                    labels = labels.to(self.device)
                    enc_cat = self.feature_extract(images)
                    # get predictions
                    outputs = classifier(enc_cat)
                    # calculate the loss
                    valid_loss = classifier_criterion(outputs, labels)
                    # record validation loss
                    valid_loss_values.append(valid_loss.item())

            # end of epoch
            logger.info("==> Testing ...")
            accuracy = self.test_classifier(classifier, testloader)
            avg_loss_values.append(np.mean(loss_values))
            avg_valid_loss_values.append(np.mean(valid_loss_values))
            
            loss_values.clear()
            valid_loss_values.clear()
            # ===================log========================
            logger.info(
                "epoch [{}/{}], loss:{:.4f}, avg_valid_loss:{:.4f}, accuracy:{:.4f}".format(
                    epoch + 1,
                    self.classifier_epochs,
                    avg_loss_values[-1],
                    avg_valid_loss_values[-1],
                    accuracy
                )
            )
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(avg_valid_loss_values[-1], classifier)
            if self.early_stop and early_stopping.early_stop:
                logger.info("Early stopping")
                visualize_earlystop(avg_valid_loss_values, avg_loss_values)
                break
        
        # load the last checkpoint with the best model
        # logger.info("==> Loading best classifier model .. ")
        # classifier.load_state_dict(torch.load(filepath))
        # accuracy = self.test_classifier(classifier, testloader)

        # find position of lowest validation loss
        minposs = avg_valid_loss_values.index(min(avg_valid_loss_values))
        avg_valid_loss_values = avg_valid_loss_values[:minposs]
        avg_loss_values = avg_loss_values[:minposs]
        
        return avg_valid_loss_values, avg_loss_values, accuracy # torch.cat(encoding_map, dim=0)
        #return avg_loss_values, accuracy # torch.cat(encoding_map, dim=0)

    def test_classifier(self, classifier, testloader):
        """Method used for training the Logistic Regressor"""
        y_test, y_pred = [], []
        with torch.no_grad(): # Not computing gradients during test (for memory efficiency)
            tp_tn = 0  # true positives and true negatives
            total = 0
            for images, labels in progress_bar(testloader):
                # Get the labels in the mini batch
                labels = labels.to(self.device)
                enc_cat = self.feature_extract(images)
                classifier.eval()
                outputs = classifier(enc_cat)
                # Prediction
                _, predicted = torch.max(outputs.data, 1)
                y_test.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                total += labels.size(0)
                tp_tn += (predicted == labels).sum()

            logger.info("==> accuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
            accuracy = tp_tn.item() / total * 100
            return accuracy

    def train_network(self, trainloader, **kwargs):
        """Method for training the Autoencoders
        TODO: keep everything as torch tensor, check for potential speed-ups
        """
        # To store the encoding map -- debugging and viz
        # encoding_map = []
        # store all estimated entropies
        entropy = {k : [] for k, v in self.autoencs.items()}
        # Store the mean of the losses for viz
        avg_train_losses = {k : [] for k, v in self.autoencs.items()}
        prev_avg_train_loss_over_all_autoencs = np.inf
        criterion = nn.MSELoss() # define the loss function
        for epoch in range(self.start_epoch, self.epochs):
            emap_epoch = []
            for _, (images, _) in enumerate(progress_bar(trainloader)):
                emap = self.init_emap(images)
                # Forward passes: K x K
                # For stabilizing the map; not tracking operations for gradients
                # ===================forward=====================
                with torch.no_grad():
                    for _ in range(self.n_units):
                        emap = self.update_emap(emap, train=False, saturate=True)
                
                # One final forward pass where operations would be tracked
                emap = self.update_emap(emap, train=True, saturate=False, criterion=criterion)
                # once the map is stable
                # emap = torch.squeeze(emap) # get rid of dimensions of 1
                # encoding_map.append(emap)
                emap_epoch.append(emap)
                # ===================backward=====================
                self.backward_pass()

            # At the end of each epoch:
            # Estimate entropy using non-parametric methods
            # get the last 128*100 activations (from last 100 minibatches)
            acts_epoch = torch.cat(emap_epoch, dim=0)
            # acts = torch.cat(encoding_map[-100:], dim=0)
            # Store the train + valid loss for viz at the enc of each epoch
            for _, (k, v) in enumerate(self.autoencs.items()):
                # Store the mean of all the losses accumulated
                avg_train_losses[k].append(np.mean(self.autoencs[k]['train_loss']))
                # Clear the previous loss values
                self.autoencs[k]['train_loss'].clear()
                self.writer.add_scalar('loss/{}'.format(k), avg_train_losses[k][-1], epoch+1)
                loc = v['loc']
                # I. Binned estimator
                x = acts_epoch[:, :, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]] # taking across all emaps; maybe just need the last one?
                x = x.reshape(x.size(0), -1)
                # compute entropy
                h_bin = binned_entropy(x, bins=self.bins)
                self.writer.add_scalar('entropy/{}'.format(k), h_bin, epoch+1)
                # h_bin2 = bin_calc_information(x.numpy(), 0.5)
                # h_bin3 = calculate_entropy(x).item()
                # ===================log========================
                logger.info(
                    "epoch: [{}/{}], unit: {}, train_loss: {:.4f}, entropy: {:.4f}".format(
                        epoch + 1,
                        self.epochs,
                        k,
                        avg_train_losses[k][-1],
                        h_bin
                    )
                )
                # ==============================================
                entropy[k].append({"epoch" : epoch+1, "h_bin" : h_bin})
                # v['scheduler'].step()
            #Implement Early Stopping for the Autoencs if the difference in re-const. error between previous
            #Epoch and this one is < 0.05
            #Contains the training loss of all the autoencs in this epoch
            autoenc_units_train_loss_list = []
            early_stop_ae_diff = 0.05
            for _, (k, v) in enumerate(self.autoencs.items()):
                autoenc_units_train_loss_list.append(avg_train_losses[k][-1])
            curr_avg_train_loss_over_all_autoencs = np.mean(autoenc_units_train_loss_list)
            logger.info("==> average autoencoder reconstruction loss in this epoch, over all the units: {:.4f}".format(curr_avg_train_loss_over_all_autoencs))
            if self.early_stop and ((prev_avg_train_loss_over_all_autoencs - curr_avg_train_loss_over_all_autoencs) < early_stop_ae_diff):
                logger.info("==> stopping the autoencoder training early .. ")
                break
            else:
                prev_avg_train_loss_over_all_autoencs = curr_avg_train_loss_over_all_autoencs 

        # checkpoint AE units
        checkpoint_path = os.path.join(self.paths.get("checkpoint"), self.filename)
        if not os.path.exists(checkpoint_path): os.mkdir(checkpoint_path)
        for name, attr in self.autoencs.items():
            path = os.path.join(checkpoint_path, kwargs.get("prefix") + '-' + name + '-' + 'checkpoint.pt')
            torch.save(attr["model"].state_dict(), path) # save model
        logger.info("==> AE checkpoint saved .. ")

        return avg_train_losses, entropy # torch.cat(encoding_map, dim=0)

    def train_baseline(self):
        """Method to train a baseline Autoencoder with classification head
        """
        pass        

    def save_results(self, **kwargs):
        if kwargs["result_type"] == "ae_network":
            # create results/self.filename folder
            if not os.path.exists(self.results_path): os.mkdir(self.results_path)
            # save the non-parametric estimated entropy values
            # self.paths.get("results")/fname/..
            save_entropy(**kwargs)
            # calculate binned entropy estimate (for all epochs)
            # get_h_binned(**kwargs)
            # plot emap
            # plot_emap(from_end=True, **kwargs)
            # plot_emap(from_end=False, **kwargs)
            # plot training losses for AE
            plot_train_loss(**kwargs)

        elif kwargs["result_type"] == "classifier":
            plot_logistic_loss(**kwargs)
            # export and save results
            export_results(**kwargs)