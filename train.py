# #%%
# %load_ext autoreload
# %autoreload 2
import os
import gc
import sys
import torch
import logging
import warnings
from src import config
from time import strftime
from src.trainer import Trainer
from src.utils import clear_tensors, create_folders
from src.handcrafted import Handcrafted
# from src.meta import Meta

warnings.filterwarnings("ignore") # Ignore warnings
create_folders(config.get("paths")) # create all folders
filename = strftime("%b%d_%H-%M-%S") # tensorboard format
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', 
                    filename=os.path.join("logs", filename + ".log"),
                    filemode="w", 
                    level=logging.INFO
                    ) # logs to file
logger = logging.getLogger("train_log")
handler = logging.StreamHandler(sys.stdout) # logs to console
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

if __name__ == "__main__":
    logger.info("Run details:")
    logger.info("Filename : {}".format(filename))
    logger.info("CUDA is available : {}".format(torch.cuda.is_available()))
    config["filename"] = filename
    ae_units = config.get("ae_units")
    for n_units in ae_units:
        config["n_units"] = n_units
        trainer = Trainer(**config)
        trainloader, testloader, validloader = trainer.prepare_data() # prepare dataloaders
        handcrafted = Handcrafted(dataset=trainer.dataset, n_units=trainer.n_units) # get the handcrafted architectures
        locs, topology, trainer.h, trainer.w = handcrafted.get_topology()
        if trainer.training_mode == "handcrafted": # handcrafted mode
            for topo in topology:
                logger.info("---------------------------------------------------------------")
                arch = topo['arch']
                name = topo['name']
                for run in range(1, trainer.runs+1): # for each run
                    logger.info('==> [units/arch/run/name] : [{}/{}/{}/{}]'.format(trainer.n_units, arch, run, name))
                    rfs = topo['rfs'] # receptive fields
                    logger.info("==> AE units initialized ..")
                    trainer.create_network(locs, rfs) # create/initialize network
                    logger.info('==> Training started ..')
                    prefix = str(trainer.n_units) + '-' + str(arch) + '-' + str(run)
                    ae_loss, entropy = trainer.train_network(trainloader, prefix=prefix) # train autoencoders
                    results = dict(result_type="meta_network",
                                path=trainer.results_path,
                                prefix=prefix,
                                ae_loss=ae_loss,
                                n_units=trainer.n_units,
                                # emap=emap,
                                entropy=entropy,
                                arch=arch,
                                run=run,
                                locs=locs
                                )
                    # clear_tensors([emap])
                    trainer.save_results(**results) # save results
                    valid_losses, classifier_loss, accuracy = trainer.train_classifier(trainloader, validloader, testloader, patience=0, units=trainer.n_units, arch=arch, run=run, prefix=prefix)
                    results["result_type"], results["accuracy"], results["classifier_loss"] = "classifier", accuracy, classifier_loss
                    trainer.save_results(**results) # save results
                    # clear_tensors([trainer.autoencs]) # clear all tensors and reclaim memory
                    # del trainer.autoencs
                    # gc.collect()
                    # torch.cuda.empty_cache()  
                logger.info("***************************************************************")
        
        elif trainer.training_mode == "meta_learning": # meta-learning mode
            meta = Meta(**config) # hill_climber, k, h, w, erf_size, locs
            for run in range(1, trainer.runs+1): # for each run
                logger.info('==> [units/run] : [{}/{}]'.format(trainer.n_units, run))
                accuracy, losses = [], []
                if meta.heuristic == "sa": meta.init_sa()
                # main loop - meta steps (1 termperature per step)
                accepted_per_temp = []
                temps = []
                for i in range(meta.meta_steps): # meta-learning loop
                    print("---------------------------- meta-step: {}/{}".format(i, meta.meta_steps-1))
                    logging.info("temp, shift: {}, {}".format(meta.T, meta.shift))
                    init = (i == 0)
                    # inner loop - ptb (perturbations per meta step)
                    accepted_count = 0
                    for rand_ptb in range(meta.max_ptb):
                        print('----- ptb-step: {}/{}'.format(rand_ptb, meta.max_ptb-1))

                        if meta.local_entropy_enabled:
                            accepted = meta.meta_heuristic_local(meta, init=init, run=run, meta_step=i, init_topo=init_topo, arch_count=accepted_count, locs=locs)
                        else:
                            accepted = meta.meta_heuristic(meta, init=init, run=run, meta_step=i, init_topo=init_topo, arch_count=accepted_count)
                        
                        if accepted:
                            accepted_count += 1

                        if is_calc_accuracy(accepted, init, i, rand_ptb):
                            accuracy_k, losses_k = test_arch_k(meta.locs, meta.autoencs)

                            logging.info("accuracy_k: {}, losses: {}".format(accuracy_k, losses_k))

                            accuracy.append(accuracy_k)
                            losses.append(losses_k)
                            #print("List of Accuracies and Losses so far: \n Accuracies: {} \n Losses: {}".format(accuracy, losses))

                        init = False    # init should not be True after first run

                    temps.append(meta.T)
                    accepted_per_temp.append(accepted_count)
                    meta.T = meta.schedule(meta.T, meta.T_step)
                    # meta.shift = (meta.shift * meta.T) / T_initial

                print("(initial, final, max) entropy: {}, {}, {}".format(meta.alpha[0], meta.alpha[-1], meta.max_entropy))

                # return all receptive fields (including good intermediate ones)
                # and evalute accuracy

                torch.cuda.empty_cache()

                if meta.heuristic == "sa":
                    sa_plots(temps, accepted_per_temp, meta.accept_prob_hist, meta.alpha, run, k)

                if calc_accuracy:
                    logging.info('==> Visualizing the accuracies\n')
                    meta_viz(accuracy, losses, run, k, meta)

                del meta
                gc.collect()
                torch.cuda.empty_cache()
