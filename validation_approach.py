from model2 import densenet121, densenet121Container
from sklearn import metrics
import torch
from datasets import CheXpert_dataset
from torch.utils import data
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger("validation")

if __name__ == "__main__":
    args_datadir = "./data/CheXpert-v1.0-small/"
    logger.info("Begin loading the weights")
    """ This needs to be adapted !!! """
    approach = "_not_layerwise"
    """ This needs to be adapted !!! """
    logger.info("Approach: {}".format(approach))

    fedma_weights_no_comm_help = load_weights("FedMA_no_comm" + str(approach))
    assignments = load_weights("FedMA_no_comm_assignments" + str(approach))

    fedma_weights_no_comm =[[] for i in range(16)]
    for i in range(16):
        """ This needs to be adapted !!! """
        fedma_weights_no_comm[i] = match_global_to_local_weights(fedma_weights_no_comm_help, assignments, i, not_layerwise = True) 
        """ This needs to be adapted !!! """
        
    fedma_weights_comm = load_weights("Pre_FedMA_comm" + str(approach))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Create a dataloader for the validation dataset")
    validation_set = CheXpert_dataset(args_datadir, valid=True, transform=True)
    # batch size needs to be increased
    validation_dl = data.DataLoader(dataset=validation_set, batch_size=50 , shuffle=False, num_workers=16)

    logger.info("Create a densenet121 model for FedMA without communication for each layer")
    global_model_FedMA_no_comm = [densenet121() for i in range (16)]

    # continuing with FedMA weights without communication
    for worker_index in range(16):
        new_state_dict = {}
        for param_idx, (key_name, param) in enumerate(global_model_FedMA_no_comm[worker_index].state_dict().items()):
            if "num_batches_tracked" in key_name:
                continue
            if "conv" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(fedma_weights_no_comm[worker_index][param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(fedma_weights_no_comm[worker_index][param_idx])}
            elif "norm" in key_name:
                temp_dict = {key_name: torch.from_numpy(fedma_weights_no_comm[worker_index][param_idx].reshape(param.size()))}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(fedma_weights_no_comm[worker_index][param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(fedma_weights_no_comm[worker_index][param_idx])}

            new_state_dict.update(temp_dict)
        global_model_FedMA_no_comm[worker_index].load_state_dict(new_state_dict)

    # target = [[] for i in range(16)]
    out = [[] for i in range(16)]

    observation_classes = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices"]
    
    for worker_index in range(16):
        target = []
        for batch_idx, (x, target_b) in enumerate(validation_dl):
                # target_b = handle_uncertainty_labels(target_b)
                # target_b = handle_NaN_values(target_b)
                # target_b = torch.tensor(target_b)
                x, target_b = x.to(device), target_b.to(device)
                out_b = global_model_FedMA_no_comm[worker_index](x)
                #logger.info(out_b)
                # target[worker_index] = np.append(target[worker_index], target_b.tolist())
                target = np.append(target, target_b.tolist())
                out[worker_index] = np.append(out[worker_index], out_b.tolist())

        # target[worker_index] = np.reshape(target[worker_index], (-1, 14))
        target = np.reshape(target, (-1, 14))
        out[worker_index] = np.reshape(out[worker_index], (-1, 14))
        logger.info("target")
        logger.info(target)
        logger.info("output")
        logger.info(out[worker_index])
        auroc = metrics.roc_auc_score(target, out[worker_index])
        logger.info("The local model {} with weights from FedMA without communication has a total AUROC of: {}".format(worker_index, auroc))
        logger.info("-------------------------------------------------------------------------------------------------")
        roc_auc = [[] for i in range(len(observation_classes))]
        for i in range(len(observation_classes)):
            roc_auc[i] = metrics.roc_auc_score(target[worker_index][:, i], out[worker_index][:, i])
            logger.info("This is the AUROC-score of model {} for observation class '{}': {}".format(worker_index, observation_classes[i], roc_auc[i]))
        logger.info("================================================================================================")

    logger.info("Continuing with the weights resulting from communication")
    
    logger.info("Create a densenet121 architecture for FedMA with communication for each layer")
    global_model_FedMA_comm = [densenet121() for i in range (16)]

    # continuing with FedMA weights with communication
    for worker_index in range(16):
        new_state_dict = {}
        for param_idx, (key_name, param) in enumerate(global_model_FedMA_comm[worker_index].state_dict().items()):
            if "num_batches_tracked" in key_name:
                continue
            if "conv" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(fedma_weights_comm[worker_index][param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(fedma_weights_comm[worker_index][param_idx])}
            elif "norm" in key_name:
                temp_dict = {key_name: torch.from_numpy(fedma_weights_comm[worker_index][param_idx].reshape(param.size()))}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(fedma_weights_comm[worker_index][param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(fedma_weights_comm[worker_index][param_idx])}

            new_state_dict.update(temp_dict)
        global_model_FedMA_comm[worker_index].load_state_dict(new_state_dict)

    target = [[] for i in range(16)]
    out = [[] for i in range(16)]
    
    for worker_index in range(16):
        for batch_idx, (x, target_b) in enumerate(validation_dl):
            # target_b = handle_uncertainty_labels(target_b)
            # target_b = handle_NaN_values(target_b)
            # target_b = torch.tensor(target_b)
            x, target_b = x.to(device), target_b.to(device)
            out_b = global_model_FedMA_comm[worker_index](x)
            #logger.info(out_b)
            target[worker_index] = np.append(target[worker_index], target_b.tolist())
            out[worker_index] = np.append(out[worker_index], out_b.tolist())

        target[worker_index] = np.reshape(target[worker_index], (-1, 14))
        out[worker_index] = np.reshape(out[worker_index], (-1, 14))

        auroc = metrics.roc_auc_score(target[worker_index], out[worker_index])
        logger.info("The local model {} with weights from FedMA with communication has a total AUROC of: {}".format(worker_index, auroc))
        logger.info("-------------------------------------------------------------------------------------------------")
        
        roc_auc = [[] for i in range(len(observation_classes))]
        for i in range(len(observation_classes)):
            roc_auc[i] = metrics.roc_auc_score(target[worker_index][:, i], out[worker_index][:, i])
            logger.info("This is the AUROC-score of model {} for observation class '{}': {}".format(worker_index, observation_classes[i], roc_auc[i]))
        logger.info("================================================================================================")