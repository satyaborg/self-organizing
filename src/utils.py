import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
import os
import gc
import json
from scipy.stats import wilcoxon
from collections import OrderedDict

logger = logging.getLogger("train_log")

colors = [k for k,v in mcolors.TABLEAU_COLORS.items()]
def set_random_seeds(seed: int = 42):
    """Sets all seeds"""
    random.seed(seed) # python random seed 
    np.random.seed(seed) # numpy random seed
    os.environ["PYTHONHASHSEED"] = str(seed) # env seed
    torch.manual_seed(seed) # pytorch seed
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 

def clear_tensors(tensors: list):
    """clear tensors from GPU"""
    for x in tensors:
        del x
    gc.collect()
    torch.cuda.empty_cache()

def create_folders(paths):
    """Creates all folders to store results
    """
    if not os.path.exists(paths["results"]): os.mkdir(paths["results"])
    if not os.path.exists(paths["checkpoint"]): os.mkdir(paths["checkpoint"])
    if not os.path.exists(paths["logs"]): os.mkdir(paths["logs"])

def wilcoxon_test(results_filename: str="results.csv"):
    df = pd.read_csv(results_filename, delimiter=',')

    df['diff'] = df['final'] - df['initial']
    w_if, p_if = wilcoxon(df['diff'], alternative='greater')

    df['diff'] = df['max'] - df['initial']
    w_im, p_im = wilcoxon(df['diff'], alternative='greater')

    logger.info('------------- (Initial - Final)')
    logger.info("w: {}".format(w_if))
    logger.info("p: {}".format(p_if))

    logger.info('------------- (Initial - Max)')
    logger.info("w: {}".format(w_im))
    logger.info("p: {}".format(p_im))


class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
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
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss
    
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

def plot_topology(emap, erfs, units=0, run=0, step=0, export=False):
    """
    Plot the receptive field positions on the map.
    """

    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(emap[-1,:,:], cmap="gray")
    # Create a Rectangle patch
    for i, erf in enumerate(erfs):
        [(x1, x2), (y1, y2)] = erf
        # Rectangle(xy, width, height, angle=0.0, **kwargs)
        rect = matplotlib.patches.Rectangle((y1,x1), y2-y1, x2-x1, linewidth=1, edgecolor=colors[i], linestyle='--', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    if export:
        fname = str(units) + '-' + str(run) + '-' + str(step) + '.png'
        #arch_path = 'Local Rule'     #KP
        plt.savefig(path/arch_path/fname)
    
    plt.show()

def calculate_metrics(entropy, accuracy):
    
    metrics = OrderedDict()

    final_entropies = []
    for k,v in entropy.items(): final_entropies.append(v[-1])
    entropies = [y for [x,y] in final_entropies]
    
    # get the metrics
    metrics['Max Entropy'] = round(np.max(entropies), 2)
    metrics['Min Entropy'] = round(np.min(entropies), 2)
    metrics['Range'] = metrics['Max Entropy'] - metrics['Min Entropy']
    metrics['Avg Entropy'] = round(np.mean(entropies), 2)
    
    quartiles = [round(q, 2) for q in np.percentile(entropies, [25, 50, 75])]
    metrics['Q25'] = quartiles[0]
    metrics['Q50'] = quartiles[1]
    metrics['Q75'] = quartiles[2]

    metrics['Accuracy'] = accuracy
    metrics['Average Ratio'] = np.mean([y/x for [x, y] in final_entropies])

    return pd.DataFrame(metrics, columns=metrics.keys(), index=[0])

def save_entropy(**kwargs):
    fname = os.path.join(kwargs["path"], kwargs["prefix"] + '-' + 'entropy' + '.json')
    with open(fname, 'w') as f:
        json.dump(kwargs["entropy"], f)

def plot_emap(**kwargs):
    fig = plt.figure(figsize=(20,8))
    for i in range(6):
        plt.subplot(2,3,i+1)
        # plt.tight_layout()
        if kwargs["from_end"]:
            plt.imshow(kwargs["emap"][-1*i, :, :], cmap='gray', interpolation='none')
        else:
            plt.imshow(kwargs["emap"][i, :, :], cmap='gray', interpolation='none')
    # save the image
    fname = 'end' if kwargs["from_end"] else 'early'
    fname = os.path.join(kwargs["path"], kwargs["prefix"] + '-' + fname + '.png')
    plt.savefig(fname)
    fig.show()

def plot_train_loss(**kwargs):
   plt.figure(figsize=(20,10))
   for k, v in kwargs["ae_loss"].items():
       plt.plot(v)
   
   # plt.xlim([1, 5])
   # plt.plot(losses);
   plt.title("Loss curves")
   plt.xlabel("Epoch")
   plt.ylabel("Loss")
   plt.legend([name for name, attr in kwargs["ae_loss"].items()])
   fname = os.path.join(kwargs["path"], kwargs["prefix"] + '-' + 'ae_loss' + '.png')
   plt.savefig(fname)
   plt.show()

def plot_logistic_loss(**kwargs):
    plt.figure(figsize=(20,10))
    plt.title("Loss curves")
    # plt.xlim([1, 5])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(kwargs["classifier_loss"])
    fname = os.path.join(kwargs["path"], kwargs["prefix"] + '-' + 'logit_loss' + '.png')
    plt.savefig(fname)
    plt.show()

def export_results(**kwargs):
    metrics = OrderedDict()
    metrics['n_units'] = kwargs["n_units"]
    metrics['type'] = kwargs["arch"]
    metrics['run'] = kwargs["run"]
    metrics['ae_loss'] = kwargs["ae_loss"]
    metrics['classifier_loss'] = kwargs["classifier_loss"]
    metrics['accuracy'] = kwargs["accuracy"]
    fname = os.path.join(kwargs["path"], kwargs["prefix"] + '-' + 'metrics.json')
    # save metrics
    with open(fname, 'w') as f:
        json.dump(metrics, f)