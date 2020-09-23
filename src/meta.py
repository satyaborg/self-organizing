import random
import logging
from src.trainer import Trainer
import numpy as np

PROB = {
        'standard': lambda e, e_, k_, T=1 : 1 if (e_-e)>=0 else np.exp((e_-e)/(k_*T))
        }
SCHEDULE = {
            'linear': lambda T, T_step : T - T_step,
            'exp' : lambda T, T_step : T * T_step
            }
logger = logging.getLogger("train_log")

class Meta(Trainer):
    """Meta class (inherits from Trainer)

    Args:
        Trainer ([type]): [description]
    """
    def __init__(self, **config):
        """Hill-climber (self.heuristic) :-
        fchc : first choice hill climbing
        rw   : random walk
        shc  : stochastic hill climbing
        sa   : simulated annealing
        """
        super(Meta, self).__init__(**config)
        # self, heuristic, k, h, w, erf_size, locs
        # general

        # meta-learning hyperparameters
        self.meta_hyperparams = config.get("meta_hyperparams")
        self.erf_size = self.meta_hyperparams.get("erf_size") # [10,20]
        self.heuristic = self.meta_hyperparams.get("heuristic")   # the hill-climbing algorithm
        self.meta_steps = self.meta_hyperparams.get("meta_steps")
        self.max_ptb = self.meta_hyperparams.get("max_ptb")
        # self.h = h
        # self.w = w
        # self.locs = locs

        self.erfs = []           # a list of list of tuples; stores the old erf
        self.erfs_new = []
        self.alpha_new = 0       # alpha for current step
        self.alpha = []          # history of all accepted alphas
        self.beta = []
        self.erf_hist = []      # history of all accepted rf's
        self.max_entropy = 0
        self.shift = 1
        
        # self.T = 
        # self.T_step = 
        # self.k_ = 
        
        # self.accept_prob_hist = []
        # self.P = 
        # self.schedule = 
        
        print('Meta-heuristic chosen : {}'.format(self.heuristic))
        
    def __str__(self):
        pass

    def init_sa(self):
        self.T = self.meta_hyperparams.get("T_initial")
        self.T_step = self.meta_hyperparams.get("T_step")
        self.k_ = self.meta_hyperparams.get("k")
        self.schedule = SCHEDULE[self.meta_hyperparams.get("schedule")]
        self.P = PROB[self.meta_hyperparams.get("prob_function")]

    def alter_erf(self):
        # alter the receptive field
        for i in range(self.k):
            self.autoencs['ae_'+str(i+1)]['erf'] = self.erfs_new[i]

    def alter_erf_local(self, i):
        # alter the receptive field of the k th autoencoder
        self.autoencs['ae_'+str(i)]['erf'] = self.erfs_new[i-1]
        print('Sanity Check - Altered effective receptive fields : ', self.autoencs['ae_'+str(i)]['erf'])

    def init_erf(self):
        """
        Args -
        k : The number of sub-units
        h : height of the encoding map
        w : Width of the encoding map
        returns -
        erfs : Randomly initalized effective receptive fields
        """
        self.erfs_new = [] # empty the erfs
        
        for i in range(self.k):
            r1 = np.random.choice(self.erf_size, replace=False)
            x = random.randint(0, self.h-r1)

            r2 = self.erf_size[0] if r1 == self.erf_size[1] else self.erf_size[1] # make it better
            y = random.randint(0, self.w-r2)

            self.erfs_new.append([(x, x+r1), (y, y+r2)])
        
        self.validate_erf(self.erfs_new)
        print('Initialized effective receptive fields : ', self.erfs_new)
        
    def init_topo(self, erfs_topo):
        """
        Args -
        k : The number of sub-units
        h : height of the encoding map
        w : Width of the encoding map
        returns -
        erfs : Randomly initalized effective receptive fields
        """
        self.erfs_new = erfs_topo
        self.validate_erf(self.erfs_new)
        print('Initialized effective receptive fields : ', self.erfs_new)

    def validate_erf(self, erfs):
        """
        Args -
        erfs : Randomly initalized  effective receptive fields
        h : Height of the encoding map
        w : Width of the encoding map
        returns -
        Boolean
        """
        for erf in erfs:
            [(x1, x2), (y1, y2)] = erf
            if not (x1 <= self.h and x2 <= self.h and y1 <= self.w and y2 <= self.w):
              print('Invalid effective receptive field!')
              return False
                # if not (x2 - x1)*(y2 - y1) == 200:
                #     print('Invalid effective receptive field!')
                #     return False

    def set_accept(self):
        self.erfs = self.erfs_new
        self.erf_hist.append(self.erfs)
        self.alpha.append(self.alpha_new)

    def set_reject(self):
        self.alpha.append(self.alpha[-1])

    def meta_obj(self):
        """
        Compute if the optimization target (e.g. entropy), stored in alpha, should be accepted.
        If yes, accept the erf (and store alpha and erf history)

        Args - uses self.alpha, which should have been updated outside of this method
        alpha : A tuple of values of alpha @t and t-1
        beta : A tuple of values of beta @t and t-1 (not used anymore)
        returns - True if the last step is an improvement
        """

        # if there are no alpha, cannot evaluate
        if len(self.alpha) == 0:
            return False

        alpha_prev = self.alpha[-1]

        if self.heuristic == 'sa':
            accept_prob = self.P(alpha_prev, self.alpha_new, self.k_, self.T)
            self.accept_prob_hist.append(accept_prob)

            logging.info("Probability of accepting the step: {}".format(accept_prob))
            logging.info("\t(e_delta, k*T) = ({}, {})".format(self.alpha_new-alpha_prev, self.k_*self.T))

            accept = accept_prob > random.uniform(0,1)
        elif self.heuristic == 'shc':
            p = 0.0 # probability of selecting the uphill move
            steepness = alpha_new - alpha_prev
            # improvement or same
            if steepness >= 0:
                if steepness >= 0.5: # accept
                    # if steepness is more than 0.5 accept with p=1
                    # else the p is smaller and smaller
                    # accept the new receptive field locations and terminate loop
                    accept = False
                elif steepness <= 0.25: # reject
                    accept = True
            else:
                accept = True
        # if either rw or fchc or rrhc
        else:
            accept = (a2 < a1)
        return accept

    def rand_shift(self):
        """
        Add a small random shift to the receptive fields
        Args -
        erfs : The current effective receptive field locations of the sub-units
        returns -
        erfs : New effective receptive field locations
        """
        
        # approach 1 : Add random shift to both x and y
        # find the maximum amount of shift that can be applied for either
        # axes
        # select a random int between 0 and max
        
        self.erfs_new = [] #.clear() # empty the erfs

        for idx, erf in enumerate(self.erfs):
            
            # get current erf location
            [(x1, x2), (y1, y2)] = erf
            
            # rw : random walk
            # hc : hill climbing
            # fchc : first-choice hill climbing
            # shc : stochastic hill climbing
            # rrhw : random-restart hill climbing
            # approach 1
            if self.heuristic == 'rw':
                # x : can shift by either -x1 or (h-x2+x1+1)
                # y : can shift by either -y1 or (w-y1)
                # all of search space
                x_shift = np.random.randint(-1*x1,(h-x2+1)) # +1 to make it inclusive
                y_shift = np.random.randint(-1*y1,(w-y2+1)) # +1 to make it inclusive

                # replace the new receptive fields
                self.erfs_new.append([(x1+x_shift, x2+x_shift), 
                                        (y1+y_shift, y2+y_shift)])
                
            # approach 2 : Add a small random shift to both x and y
            # to find a state in the neighbourhood of current state
            else: 
                x_shift, y_shift = 0, 0

                fail_count = 0
                for i in range(100):    # try max of 100 times
                    if x1 == 0 and x2 == h: x_shift = 0 # no shift
                    elif x1 == 0 : x_shift = 1 # +1 shift only
                    elif x2 == h : x_shift = -1 # -1 shift only
                    else: x_shift = np.random.choice(step_list, replace=True)

                    if y1 == 0 and y2 == w: y_shift = 0 # no shift
                    elif y1 == 0 : y_shift = 1 # +1 shift only
                    elif y2 == w : y_shift = -1 # -1 shift only
                    else: y_shift = np.random.choice(step_list, replace=True)

                    # try again if NEITHER of x,y have a shift
                    if x_shift == 0 and y_shift == 0:
                        fail_count += 1
                        logging.info("random shift is (0,0), so try again.")
                    else:
                        break

                # replace the new receptive fields
                self.erfs_new.append([(x1+x_shift, x2+x_shift), 
                                        (y1+y_shift, y2+y_shift)])

        # validate that the new receptive fields are valid
        self.validate_erf(self.erfs_new)

        logging.info('old effective receptive fields : {}'.format(self.erfs))
        logging.info('new effective receptive fields : {}'.format(self.erfs_new))

        # replace the old erfs with the new ones
        # self.alter_erf()
        
    def rand_shift_local_no_size_change(self, erf_idx):
        """
        Add a small random shift to the receptive field number idx
        Args -
        erfs : The current effective receptive field locations of the sub-units
        returns -
        erfs : New effective receptive field locations, with the e.r.f number idx having been peturbed
        """

        # approach 1 : Add random shift to both x and y
        # find the maximum amount of shift that can be applied for either
        # axes
        # select a random int between 0 and max
        
        # self.erfs_new = self.erfs  #Copy over the previous erfs, before peturbing one of them

        # #aaaah, wait. From what I can see, from the output is that the above assignment seems to be a shallow copy :3
        # #And hence any change is reflected across both. I need to ensure that I can do a deep copy.

        self.erfs_new = []
        self.erfs_new = copy.deepcopy(self.erfs)
        
        # get current erf location
        [(x1, x2), (y1, y2)] = self.erfs[erf_idx]
        
        # rw : random walk
        # hc : hill climbing
        # fchc : first-choice hill climbing
        # shc : stochastic hill climbing
        # rrhw : random-restart hill climbing
        # approach 1
        if self.heuristic == 'rw':
            # x : can shift by either -x1 or (h-x2+x1+1)
            # y : can shift by either -y1 or (w-y1)
            # all of search space
            x_shift = np.random.randint(-1*x1,(h-x2+1)) # +1 to make it inclusive
            y_shift = np.random.randint(-1*y1,(w-y2+1)) # +1 to make it inclusive

            # replace the new receptive fields
            self.erfs_new[erf_idx] = ([(x1+x_shift, x2+x_shift), 
                                    (y1+y_shift, y2+y_shift)])
            
        # approach 2 : Add a small random shift to both x and y
        # to find a state in the neighbourhood of current state
        else: 
            x_shift, y_shift = 0, 0

            if x1 == 0 and x2 == h: x_shift = 0 # no shift
            elif x1 == 0 : x_shift = 1 # +1 shift only
            elif x2 == h : x_shift = -1 # -1 shift only
            else : x_shift = np.random.choice(step_list, replace=True); # either +1 or -1

            if y1 == 0 and y2 == w: y_shift = 0 # no shift
            elif y1 == 0 : y_shift = 1 # +1 shift only
            elif y2 == w : y_shift = -1 # -1 shift only
            else : y_shift = np.random.choice(step_list, replace=True); # either +1 or -1

            # replace the new receptive fields
            self.erfs_new[erf_idx] = ([(x1+x_shift, x2+x_shift), 
                                    (y1+y_shift, y2+y_shift)])

        # validate that the new receptive fields are valid
        self.validate_erf(self.erfs_new)

        logging.info('old effective receptive fields : {}'.format(self.erfs))
        logging.info('new effective receptive fields : {}'.format(self.erfs_new))

        # replace the old erfs with the new ones
        # self.alter_erf()
      
    def rand_shift_local(self, erf_idx):
        """
        Add a small random shift to the receptive field number idx
        Args -
        erfs : The current effective receptive field locations of the sub-units
        returns -
        erfs : New effective receptive field locations, with the e.r.f number idx having been peturbed
        """

        # approach 1 : Add random shift to both x and y
        # find the maximum amount of shift that can be applied for either
        # axes
        # select a random int between 0 and max
        
        # self.erfs_new = self.erfs  #Copy over the previous erfs, before peturbing one of them

        # #aaaah, wait. From what I can see, from the output is that the above assignment seems to be a shallow copy :3
        # #And hence any change is reflected across both. I need to ensure that I can do a deep copy.

        self.erfs_new = []
        self.erfs_new = copy.deepcopy(self.erfs)

        # get current erf location
        [(x1, x2), (y1, y2)] = self.erfs[erf_idx]
        
        # rw : random walk
        # hc : hill climbing
        # fchc : first-choice hill climbing
        # shc : stochastic hill climbing
        # rrhw : random-restart hill climbing
        # approach 1
        if self.heuristic == 'rw':
            # x : can shift by either -x1 or (h-x2+x1+1)
            # y : can shift by either -y1 or (w-y1)
            # all of search space
            x_shift = np.random.randint(-1*x1,(h-x2+1)) # +1 to make it inclusive
            y_shift = np.random.randint(-1*y1,(w-y2+1)) # +1 to make it inclusive

            # replace the new receptive fields
            self.erfs_new[erf_idx] = ([(x1+x_shift, x2+x_shift), 
                                    (y1+y_shift, y2+y_shift)])
            
        # approach 2 : Add a small random shift to both x and y
        # to find a state in the neighbourhood of current state
        else: 
            x_shift, y_shift = 0, 0

            if x1 == 0 and x2 == h: x_shift = 0 # no shift
            elif x1 == 0 : x_shift = 1 # +1 shift only
            elif x2 == h : x_shift = -1 # -1 shift only
            else : x_shift = np.random.choice(step_list, replace=True)  # either +1 or -1

            if y1 == 0 and y2 == w: y_shift = 0 # no shift
            elif y1 == 0 : y_shift = 1 # +1 shift only
            elif y2 == w : y_shift = -1 # -1 shift only
            else : y_shift = np.random.choice(step_list, replace=True)  # either +1 or -1

            ch = np.random.randint(0, 1000)
            ch = ch%4

            if ch == 0:
              self.erfs_new[erf_idx] = ([(x1+x_shift, x2), (y1, y2)])
            elif ch == 1:
              self.erfs_new[erf_idx] = ([(x1, x2+x_shift), (y1, y2)])
            elif ch == 2: 
              self.erfs_new[erf_idx] = ([(x1, x2), (y1+y_shift, y2)])
            else:
              self.erfs_new[erf_idx] = ([(x1, x2), (y1, y2+y_shift)])

            # replace the new receptive fields
            # self.erfs_new[erf_idx] = ([(x1+x_shift, x2+x_shift), 
            #                         (y1+y_shift, y2+y_shift)])

        # validate that the new receptive fields are valid
        self.validate_erf(self.erfs_new)

        logging.info('old effective receptive fields : {}'.format(self.erfs))
        logging.info('new effective receptive fields : {}'.format(self.erfs_new))
        
        # replace the old erfs with the new ones
        # self.alter_erf()

    def compute_metrics(self, h_bin, init=False):
        """
        entropy : A list of k entropies
        """

        # correlated metrics --
        alpha = round(np.median(h_bin), 4)
        self.max_entropy = max(alpha, self.max_entropy) 

        # add new alpha to the list
        self.alpha_new = alpha
        
    def meta_heuristic(self, meta, init=False, run=0, meta_step=0, init_topo=False, arch_count=0):

        """
        One step of the hill climbing algorithm
        Note that it is called from meta_loop, which also has important functionality for each 'step'.
        
        Initialise, perturb rf's, train ae's, calc entropy

        Returns TRUE if the step was accepted
        """

        accuracy_k, losses_k = None, None
        if init:
            if init_topo:
                self.init_topo(erfs_topo)     # set topology
            else:
                self.init_erf()               # random toppology
        else:
            meta.rand_shift()                 # perturb receptive field positions
            if cleared_weights == 0:   # ie. weights preserved.  
                meta.alter_erf()

        if init or (cleared_weights == 1):
            meta.create_ae(patch_sizes=patch_sizes, input_dim=input_dim, 
                        hidden_units=hidden_units, dropout=dropout, wd=wd, lr=lr)

        # training with new AE units with random weights
        loss, encoding_map = train(epochs=epochs_ae, autoencs=meta.autoencs, h=h, w=w)

        # estimate the entropy
        h_bin = get_h(encoding_map, meta.locs) # h_knn, h_upper, h_lower
        # entropy = [v[-1] for k,v in entropy.items()]

        # compute and record alpha (into self.alpha_new)
        meta.compute_metrics(h_bin, init)

        # evaluate --
        if init:
            logging.info("--> step INITIAL")
            meta.set_accept()
            meta.accept_prob_hist.append(1.0)

            # note: looks like it overrides, so only 1 for each meta_step
            if logging_topology_maps:
                plot_topology(encoding_map, meta.erfs, units=meta.k, run=run, 
                            step=meta_step, export=True)
            ret = True
        elif meta.meta_obj():
            logging.info("--> step ACCEPTED")
            meta.set_accept()

            # plot the accepted receptive field positions
            # note: looks like it overrides, so only 1 for each meta_step
            if logging_topology_maps:
                plot_topology(encoding_map, meta.erfs, units=meta.k, run=run, 
                            step=meta_step, export=True)
            ret = True
        else:
            logging.info("--> step REJECTED")
            meta.set_reject()
            # plot the rejected receptive field positions
            if logging_topology_maps:
                plot_topology(encoding_map, meta.erfs_new)
            ret = False

        if logging_topology_graphs:
            call_visualise(meta.locs, meta.autoencs, units=meta.k, run=run, step=meta_step)

        del encoding_map
        gc.collect()    

        return ret

    def is_calc_accuracy(self, accepted, init, m_step, rand_ptb_step):

        if not calc_accuracy:
            return False

        once_per_temp = True
        if once_per_temp:
            if init or rand_ptb_step == (max_ptb-1):   # initial arch or last ptb of meta-step (temp)
                return True
        elif accepted:
            return True
        return False

    def test_arch_k(locs, autoencs):
        print('Testing arch')
        # in_dim = k * 10 * 10
        #logit_model = LogisticRegressionModel(k*x*y, classes).cuda()

        logit_model = LogisticRegressionModel_1(
            k*x*y, classes, hidden_layers, nn.ReLU(inplace=True)).to(device)

        logit_criterion = nn.CrossEntropyLoss()
        logit_optimizer = torch.optim.Adam(logit_model.parameters(), lr=lr, 
                                            weight_decay=wd)
        # plot_train_loss(train_loss)

        # training logistic regression model
        valid_losses, losses, accuracy = logit_train(epochs_logit, logit_model, 
                                                    logit_optimizer, 
                                                    logit_criterion, autoencs, 
                                                    h=h, w=w, 
                                                    patience=early_stop_pat)
        # plot_logistic_loss(losses=losses) 

        return accuracy, losses