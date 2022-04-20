""" die in dieser Datei befindlichen Methoden und Klassen im Zusammenhang mit dem Datensatz dienen als Referenz und müsse noch angepasst werden """

import math 
import os
import numpy as np
import torch
import torch.nn as nn
import logging
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, roc_auc_score
import io
import time

# we've changed to a faster solver
#from scipy.optimize import linear_sum_assignment
import logging

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from datasets import MNIST_truncated, CIFAR10_truncated, ImageFolderTruncated, CIFAR10ColorGrayScaleTruncated, CheXpert_dataset
from combine_nets import prepare_uniform_weights, prepare_sanity_weights, prepare_weight_matrix, normalize_weights, get_weighted_average_pred

from vgg import *
from model import *
from model2 import *



logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger("utils")

""" New Method """
def densenet_index(layer_index):
    if layer_index <= 2:
        this_index = layer_index -1
    elif layer_index == 3:
        this_index = 6
    else:
        if layer_index % 2 == 0:
            even_number = True
        else:
            even_number = False
        if even_number:
            this_index = 11 + (layer_index - 4) * 3 
        else:
            this_index = 11 + (layer_index - 5) * 3 + 1
    return this_index

def densenet_index_2(layer_index):
    if layer_index <= 1:
        this_index = layer_index -1
    else:
        this_index = 6 + (layer_index - 2) * 6
    
    return this_index

def densenet_index_3(layer_index):
    if layer_index <= 1:
        this_index = layer_index -1
    else:
        this_index = 11 + (layer_index - 2) * 6
    
    return this_index

def is_batchnorm(layer_index):
    if layer_index == 2:
        return True
    elif (layer_index >= 3) and (layer_index % 2 == 1):
        return True
    else:
        return False

def densenet_index_to_layer(index):
    #notice layer index starts here with 0 and above at 1
    if index <= 1:
        layer_index = index
    elif index <=5:
        layer_index = 1
    elif index <= 10:
        layer_index = 2
    elif (index+1) % 6 == 0:
        layer_index = (index-11) / 3 + 3
    elif index == 726:
        layer_index = 241
    else:
        help = (index+1) % 6
        layer_index = (index - help - 11) / 3 + 4
    return int(layer_index)

def densenet_index_to_layer_2(index):
    #notice layer index starts here with 0 and above at 1
    if index <= 5:
        layer_index = 0
    elif (index) % 6 == 0:
        layer_index = (index-6) / 6 + 1
    elif index == 726:
        layer_index = 120
    else:
        help = (index) % 6
        layer_index = (index - help - 6) / 6 + 1
    return int(layer_index)

def densenet_index_to_layer_3(index):
    #notice layer index starts here with 0 and above at 1
    if index <= 10:
        layer_index = 0
    elif (index+1) % 6 == 0:
        layer_index = (index-11) / 6 + 1
    elif index == 726:
        layer_index = 120
    else:
        help = (index+1) % 6
        layer_index = (index - help - 11) / 6 + 1
    return int(layer_index)

def match_global_to_local_weights(hungarian_weights, assignments, client_index, not_layerwise = False):
    
    dummy_model = densenet121()
    new_weights_list = []
    help_index = -1
    counter = 0
    for param_idx, (key_name, param) in enumerate(dummy_model.state_dict().items()):
        logger.info("Parameter {}: {}".format(param_idx, key_name))
        layer_index = densenet_index_to_layer(param_idx)
        logger.info("Layer {}".format(layer_index))
        if layer_index != help_index:
            help_index = layer_index
            counter = 0
        else:
            counter += 1
        if "classifier.bias" in key_name:
            layer_index += 1
        if "num_batches_tracked" in key_name:
            new_weights_list.append(np.array([]))
            continue
        if "classifier" in key_name:
            if not not_layerwise:
                new_weights = hungarian_weights[layer_index]
            else:
                new_weights = hungarian_weights[layer_index][client_index]
            logger.info("Shape of the new weights: {}".format(np.array(new_weights).shape))
            new_weights_list.append(np.array(new_weights))
        else:
            #logger.info("Those are the assignments")
            logger.info(assignments[layer_index][client_index])
            #logger.info("Safety check")
            new_weights = np.array((param))
            if "conv" in key_name:
                new_weights = np.zeros((len(assignments[layer_index][client_index]), np.array(param).shape[1] * np.array(param).shape[2] *np.array(param).shape[3]))
            for j in range(len(assignments[layer_index][client_index])):
                #if (len(assignments[layer_index][client_index]) == np.array(param).shape[0]):
                    #logger.info("passed")
                #else:
                    #logger.info("failed")
                if not not_layerwise:
                    # if "conv" in key_name:
                    #     logger.info("Weights shape {}".format(np.array(hungarian_weights[layer_index]).shape))
                    #     new_weights[j] = hungarian_weights[layer_index][assignments[layer_index][client_index][j]]
                    #else:
                    # new_weights = np.zeros((len(assignments[layer_index][client_index]), np.array(hungarian_weights[layer_index][counter]).shape[1]))
                    # logger.info("Weights shape {}".format(np.array(hungarian_weights[layer_index][counter]).shape))
                    new_weights[j] = hungarian_weights[layer_index][counter][assignments[layer_index][client_index][j]]
                else:
                    # if "conv" in key_name:
                    #     logger.info("Weights shape {}".format(np.array(hungarian_weights[layer_index][client_index]).shape))
                    #     new_weights[j] = hungarian_weights[layer_index][client_index][assignments[layer_index][client_index][j]]
                    # else:
                    # new_weights = np.zeros((len(assignments[layer_index][client_index]), np.array(hungarian_weights[layer_index][client_index][counter]).shape[1]))
                    # logger.info("Weights shape {}".format(np.array(hungarian_weights[layer_index][client_index][counter]).shape))
                    new_weights[j] = hungarian_weights[layer_index][client_index][counter][assignments[layer_index][client_index][j]]
            logger.info("Shape of the new weights: {}".format(np.array(new_weights).shape))
            new_weights_list.append(np.array(new_weights))
    return new_weights_list

def match_global_to_local_weights_2(hungarian_weights, assignments, client_index, not_layerwise = False):
    
    dummy_model = densenet121()
    new_weights_list = []
    help_index = -1
    counter = 0
    for param_idx, (key_name, param) in enumerate(dummy_model.state_dict().items()):
        logger.info("Parameter {}: {}".format(param_idx, key_name))
        layer_index = densenet_index_to_layer_2(param_idx)
        logger.info("Layer {}".format(layer_index))
        if layer_index != help_index:
            help_index = layer_index
            counter = 0
        else:
            counter += 1
        if "classifier.bias" in key_name:
            layer_index += 1
        if "num_batches_tracked" in key_name:
            new_weights_list.append(np.array([]))
            continue
        if "classifier" in key_name:
            if not not_layerwise:
                new_weights = hungarian_weights[layer_index]
            else:
                new_weights = hungarian_weights[layer_index][client_index]
            logger.info("Shape of the new weights: {}".format(np.array(new_weights).shape))
            new_weights_list.append(np.array(new_weights))
        else:
            #logger.info("Those are the assignments")
            logger.info(assignments[layer_index][client_index])
            #logger.info("Safety check")
            new_weights = np.array((param))
            if "conv" in key_name:
                new_weights = np.zeros((len(assignments[layer_index][client_index]), np.array(param).shape[1] * np.array(param).shape[2] *np.array(param).shape[3]))
            for j in range(len(assignments[layer_index][client_index])):
                #if (len(assignments[layer_index][client_index]) == np.array(param).shape[0]):
                    #logger.info("passed")
                #else:
                    #logger.info("failed")
                if not not_layerwise:
                    # if "conv" in key_name:
                    #     logger.info("Weights shape {}".format(np.array(hungarian_weights[layer_index]).shape))
                    #     new_weights[j] = hungarian_weights[layer_index][assignments[layer_index][client_index][j]]
                    #else:
                    # new_weights = np.zeros((len(assignments[layer_index][client_index]), np.array(hungarian_weights[layer_index][counter]).shape[1]))
                    # logger.info("Weights shape {}".format(np.array(hungarian_weights[layer_index][counter]).shape))
                    new_weights[j] = hungarian_weights[layer_index][counter][assignments[layer_index][client_index][j]]
                else:
                    # if "conv" in key_name:
                    #     logger.info("Weights shape {}".format(np.array(hungarian_weights[layer_index][client_index]).shape))
                    #     new_weights[j] = hungarian_weights[layer_index][client_index][assignments[layer_index][client_index][j]]
                    # else:
                    # new_weights = np.zeros((len(assignments[layer_index][client_index]), np.array(hungarian_weights[layer_index][client_index][counter]).shape[1]))
                    # logger.info("Weights shape {}".format(np.array(hungarian_weights[layer_index][client_index][counter]).shape))
                    new_weights[j] = hungarian_weights[layer_index][client_index][counter][assignments[layer_index][client_index][j]]
            logger.info("Shape of the new weights: {}".format(np.array(new_weights).shape))
            new_weights_list.append(np.array(new_weights))
    return new_weights_list

def match_global_to_local_weights_3(hungarian_weights, assignments, client_index, not_layerwise = False):
    
    dummy_model = densenet121()
    new_weights_list = []
    help_index = -1
    counter = 0
    for param_idx, (key_name, param) in enumerate(dummy_model.state_dict().items()):
        logger.info("Parameter {}: {}".format(param_idx, key_name))
        layer_index = densenet_index_to_layer_3(param_idx)
        logger.info("Layer {}".format(layer_index))
        if layer_index != help_index:
            help_index = layer_index
            counter = 0
        else:
            counter += 1
        if "classifier.bias" in key_name:
            layer_index += 1
        if "num_batches_tracked" in key_name:
            new_weights_list.append(np.array([]))
        elif "classifier" in key_name:
            if not not_layerwise:
                new_weights = hungarian_weights[layer_index]
            else:
                new_weights = hungarian_weights[layer_index][client_index]
            logger.info("Shape of the new weights: {}".format(np.array(new_weights).shape))
            new_weights_list.append(np.array(new_weights))
        elif "norm" in key_name:
            if not not_layerwise:
                new_weights = hungarian_weights[layer_index][counter]
            else:
                new_weights = hungarian_weights[layer_index][client_index][counter]
            logger.info("Shape of the new weights: {}".format(np.array(new_weights).shape))
            new_weights_list.append(np.array(new_weights))
        else:
            #logger.info("Those are the assignments")
            logger.info(assignments[layer_index][client_index])
            #logger.info("Safety check")
            new_weights = np.array((param))
            if "conv" in key_name:
                new_weights = np.zeros((len(assignments[layer_index][client_index]), np.array(param).shape[1] * np.array(param).shape[2] *np.array(param).shape[3]))
            for j in range(len(assignments[layer_index][client_index])):
                #if (len(assignments[layer_index][client_index]) == np.array(param).shape[0]):
                    #logger.info("passed")
                #else:
                    #logger.info("failed")
                if not not_layerwise:
                    # if "conv" in key_name:
                    #     logger.info("Weights shape {}".format(np.array(hungarian_weights[layer_index]).shape))
                    #     new_weights[j] = hungarian_weights[layer_index][assignments[layer_index][client_index][j]]
                    #else:
                    # new_weights = np.zeros((len(assignments[layer_index][client_index]), np.array(hungarian_weights[layer_index][counter]).shape[1]))
                    # logger.info("Weights shape {}".format(np.array(hungarian_weights[layer_index][counter]).shape))
                    new_weights[j] = hungarian_weights[layer_index][counter][assignments[layer_index][client_index][j]]
                else:
                    # if "conv" in key_name:
                    #     logger.info("Weights shape {}".format(np.array(hungarian_weights[layer_index][client_index]).shape))
                    #     new_weights[j] = hungarian_weights[layer_index][client_index][assignments[layer_index][client_index][j]]
                    # else:
                    # new_weights = np.zeros((len(assignments[layer_index][client_index]), np.array(hungarian_weights[layer_index][client_index][counter]).shape[1]))
                    # logger.info("Weights shape {}".format(np.array(hungarian_weights[layer_index][client_index][counter]).shape))
                    new_weights[j] = hungarian_weights[layer_index][client_index][counter][assignments[layer_index][client_index][j]]
            logger.info("Shape of the new weights: {}".format(np.array(new_weights).shape))
            new_weights_list.append(np.array(new_weights))
    return new_weights_list

""" Original methods from the FedMA implementation """
def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def parse_class_dist(net_class_config):

    cls_net_map = {}

    for net_idx, net_classes in enumerate(net_class_config):
        for net_cls in net_classes:
            if net_cls not in cls_net_map:
                cls_net_map[net_cls] = []
            cls_net_map[net_cls].append(net_idx)

    return cls_net_map

def record_net_data_stats(y_train, net_dataidx_map, logdir, dataset="chexpert"):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        tmp_list = {}
        if dataset == "chexpert":
            for i in y_train.columns:
                
                class_count = 0
                for x in dataidx:
                    if y_train.iloc[x][i] == 1.0:
                        class_count += 1
                tmp = {i: class_count}
                tmp_list.update(tmp)
            net_cls_counts[net_i] = tmp_list
        else:
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_nets, alpha, args):

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'cinic10':
        _train_dir = './data/cinic10/cinic-10-trainlarge/train'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        trainset = ImageFolderTruncated(_train_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), 
                                                                                            requires_grad=False),
                                                                                            (4,4,4,4),mode='reflect').data.squeeze()),
                                                            transforms.ToPILImage(),
                                                            transforms.RandomCrop(32),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=cinic_mean,std=cinic_std),
                                                            ]))
        y_train = trainset.get_train_labels
        n_train = y_train.shape[0]
    elif dataset == 'chexpert':
        logger.info("Load CheXpert data")
        X_train, y_train, X_test, y_test = load_chexpert_data(datadir) # evtl hier schon validation set einführen
        n_train = y_train.shape[0]
        logger.info("Total number of training data: %s" % n_train)

    # Note for the CheXpert dataset only partition "homo" is applicable
    if partition == "homo":
        if dataset == 'chexpert':
            logger.info("Assigning patients to local clients")
            if args.retrain:
                patients = list(set(X_train))
                patients = np.random.permutation(patients)
                batch_patients = np.array_split(patients, n_nets)

                help_list = [[0] for x in range(n_nets)]

                for i in range(n_train):
                    found_it = False
                    for n in range(n_nets):
                        if found_it:
                            break
                        for patient in batch_patients[n]:
                            if patient == X_train[i]:
                                help_list[n].append(i)
                                found_it = True
                                break
                        
                for n in range(n_nets):
                    help_list[n].pop(0)
                
                net_dataidx_map = {i: help_list[i] for i in range(n_nets)}
                save_datamap(net_dataidx_map)
            else:
                folder = "./saved_weights/Durchlauf6/"
                net_dataidx_map = load_datamap_from_folder(folder)
            
            logger.info("Assignment of patients done")

            for i in range(n_nets):
                dataidxs = net_dataidx_map[i]
                logger.info("Network %s. n_training: %d" % (str(i), len(dataidxs)))
        else:
            idxs = np.random.permutation(n_train)
            batch_idxs = np.array_split(idxs, n_nets)
            net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir" and dataset != 'chexpert':
        min_size = 0
        K = 10
        
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fbs" and dataset != 'chexpert':
        # in this part we conduct a experimental study on exploring the effect of increasing the number of batches
        # but the number of data points are approximately fixed for each batch
        # the arguments we need to use here are: `args.partition_step_size`, `args.local_points`, `args.partition_step`(step can be {0, 1, ..., args.partition_step_size - 1}).
        # Note that `args.partition` need to be fixed as "hetero-fbs" where fbs means fixed batch size
        net_dataidx_map = {}

        # stage 1st: homo partition
        idxs = np.random.permutation(n_train)
        total_num_batches = int(n_train/args.local_points) # e.g. currently we have 180k, we want each local batch has 5k data points the `total_num_batches` becomes 36
        step_batch_idxs = np.array_split(idxs, args.partition_step_size)
        
        sub_partition_size = int(total_num_batches / args.partition_step_size) # e.g. for `total_num_batches` at 36 and `args.partition_step_size` at 6, we have `sub_partition_size` at 6

        # stage 2nd: hetero partition
        n_batches = (args.partition_step + 1) * sub_partition_size
        min_size = 0
        K = 10
        #N = len(step_batch_idxs[args.step])
        baseline_indices = np.concatenate([step_batch_idxs[i] for i in range(args.partition_step + 1)])
        y_train = y_train[baseline_indices]
        N = y_train.shape[0]

        while min_size < 10:
            idx_batch = [[] for _ in range(n_batches)]
            # for each class in the dataset
            for k in range(K):
            
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_batches))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_batches) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        # we leave this to the end
        for j in range(n_batches):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
        return y_train, net_dataidx_map, traindata_cls_counts, baseline_indices

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)

    #return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)
    return y_train, net_dataidx_map, traindata_cls_counts

def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def load_chexpert_data(datadir): 

    chexpert_train_ds = CheXpert_dataset(datadir, train=True, transform=True)
    chexpert_test_ds = CheXpert_dataset(datadir, train=False, transform=True)

    buffer_train = io.StringIO()
    logger.info("Training dataset info:")
    chexpert_train_ds.dataframe.info(verbose= False, memory_usage="deep", buf = buffer_train)
    train_info = buffer_train.getvalue()
    buffer_train.close()
    logger.info(train_info)
    buffer_test = io.StringIO()
    logger.info("Testing dataset info:")
    chexpert_test_ds.dataframe.info(verbose= False, memory_usage="deep", buf = buffer_test)
    test_info = buffer_test.getvalue()
    logger.info(test_info)
    buffer_test.close()

    # training_set = data.DataLoader(chexpert_train_ds, batch_size = chexpert_train_ds.__len__(), shuffle=True)
    # test_set = data.DataLoader(chexpert_test_ds, batch_size = chexpert_test_ds.__len__(), shuffle=True)
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

    X_train = []
    for x in chexpert_train_ds.dataframe["Path"]:
        X_train.append(int(x[33:38]))
    y_train = chexpert_train_ds.dataframe[observation_classes]
    X_test = chexpert_test_ds.dataframe["Path"]
    y_test = chexpert_test_ds.dataframe[observation_classes]
    logger.info("CheXpert data loaded")

    return (X_train, y_train, X_test, y_test)

def handle_uncertainty_labels(labels):
    # apply the U-Ones approach for the class Atelectasis, for the other classes apply the U-Zeroes approach
    
    for x in labels:
        for i in range(14):
            if (x[i] == -1):
                x[i] = 0
            elif (i == 8 and x[8] == 0):
                x[8] = 1
            
    return labels

def handle_single_uncertainty_label(label):
    # apply the U-Ones approach for the class Atelectasis, U-Multiclass approach (for Cardiomegaly) is the default approach
    
    if (label[8] == 0):
        label[8] = 1
    return label   

def handle_NaN_values(labels):

    for x in labels:
        for i in range(14):
            if math.isnan(x[i]):
                x[i] = 0  
    return labels  

def compute_auroc(model, dataloader, device="cpu", dataset=None):

    logger.info("Calculate auroc score")
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    target =  np.array([])
    out =  np.array([])
    for batch_idx, (x, target_b) in enumerate(dataloader):
            target_b = handle_uncertainty_labels(target_b)
            target_b = handle_NaN_values(target_b)
            # target_b = torch.tensor(target_b)
            x, target_b = x.to(device), target_b.to(device)
            out_b = model(x)
            #logger.info(out_b)
            target = np.append(target, target_b.tolist())
            out = np.append(out, out_b.tolist())
    
    target = np.reshape(target, (-1, 14))
    out = np.reshape(out, (-1, 14))
    logger.info("Exemplary target values for one entry")
    logger.info(target[0])
    logger.info("Exemplary output values for one entry")
    logger.info(out[0])
    #NaN values zu 0 umwandeln für target
    # roc_auc_score nicht für multilabel multiclass problems --> auf multilabel problem reduziert
    
    auroc = roc_auc_score(target, out)

    if was_training:
        model.train()
    logger.info("auroc score calculated")
    return auroc

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", dataset=None):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            target = handle_uncertainty_labels(target)
               
            x, target = x.to(device), target.to(device)
            out = model(x)
        
            _, pred_label = torch.max(out.data, 1)
            if dataset == "chexpert":
                pred_label = []
                for pred in  out:
                    pred_row = []
                    for i in pred:
                        if i >= 0.5:
                            pred_row.append(1)
                        elif i <= -0.5:
                            pred_row.append(-1)
                        elif (i < 0.5 and i > -0.5):
                            pred_row.append(0)
                        else:
                            pred_row.append(None)
                    pred_label.append(pred_row)

            for i in range(len(target)):
                for l in range(len(target[i])):
                    if not math.isnan(target[i][l].item()):
                        total += 1
                        if (pred_label[i][l] == target[i][l].item()):
                            correct += 1
            # total += x.data.size()[0]
            # correct += (pred_label == target.data).sum().item() # hier evtl aufpassen
            pred_label = torch.tensor(pred_label)
            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())               

    if get_confusion_matrix:
        logger.info("Calculate confusion matrix")
        if dataset == "chexpert":
            conf_matrix = multilabel_confusion_matrix(true_labels_list, pred_labels_list)
        else:
            conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)


def init_cnns(net_configs, n_nets):
    '''
    Initialize the local CNNs
    Please note that this part is hard coded right now
    '''

    input_size = (16 * 5 * 5) # hard coded, defined by the SimpleCNN useds
    output_size = net_configs[-1] #
    hidden_sizes = [120, 84]

    cnns = {net_i: None for net_i in range(n_nets)}

    # we add this book keeping to store meta data of model weights
    model_meta_data = []
    layer_type = []

    for cnn_i in range(n_nets):
        cnn = SimpleCNN(input_size, hidden_sizes, output_size)

        cnns[cnn_i] = cnn

    for (k, v) in cnns[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
        #logger.info("Layer name: {}, layer shape: {}".format(k, v.shape))
    return cnns, model_meta_data, layer_type


def init_models(net_configs, n_nets, args):
    '''
    Initialize the local LeNets
    Please note that this part is hard coded right now
    '''
    from model2 import densenet121

    cnns = {net_i: None for net_i in range(n_nets)}

    # we add this book keeping to store meta data of model weights
    model_meta_data = []
    layer_type = []

    for cnn_i in range(n_nets):
        if args.model == "lenet":
            cnn = LeNet()
        elif args.model == "vgg":
            cnn = vgg11()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10"):
                cnn = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == "mnist":
                cnn = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
        elif args.model == "moderate-cnn":
            if args.dataset == "mnist":
                cnn = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10"):
                cnn = ModerateCNN()
        elif args.model =="densenet121":
            cnn = densenet121()

        cnns[cnn_i] = cnn

    for (k, v) in cnns[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
        #logger.info("{} ::: Layer name: {}, layer shape: {}".format(args.model, k, v.shape))
    return cnns, model_meta_data, layer_type


def save_model(model, model_index):
    logger.info("saving local model-{}".format(model_index))
    with open("trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def save_datamap(datamap):
    logger.info("saving datamap")
    with open("dataidx_map", "wb") as f_:
        torch.save(datamap, f_)
    return

def save_weights(weights, alias):
    logger.info("saving weights-{}".format(alias))
    with open("./saved_weights/trained_weights"+str(alias), "wb") as f_:
        torch.save(weights, f_)
    return

def load_weights(alias):
    logger.info("loading weights-{}".format(alias))
    with open("./saved_weights/trained_weights"+str(alias), "rb") as f_:
        weights = torch.load(f_)
    return weights

def load_weights_from_folder(alias, folder = "./saved_weights/"):
    logger.info("loading weights-{}".format(alias))
    with open(str(folder) + str(alias), "rb") as f_:
        weights = torch.load(f_)
    return weights

def load_datamap_from_folder(folder):
    with open(folder+"dataidx_map", "rb") as f_:
        dataidx_map = torch.load(f_)
    return dataidx_map

def load_model(model, model_index, rank=0, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model

def load_model_from_folder(folder, model, model_index, rank=0, device="cpu"):
    with open(folder+"trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):

    if dataset in ('mnist', 'cifar10'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])


        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    elif dataset == 'cinic10':
        # statistic for normalizing the dataset
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]

        cinic_directory = './data/cinic10'

        training_set = ImageFolderTruncated(cinic_directory + '/cinic-10-trainlarge/train', 
                                                                        dataidxs=dataidxs,
                                                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                                                     transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), requires_grad=False),
                                                                                                     (4,4,4,4),mode='reflect').data.squeeze()),
                                                                                                     transforms.ToPILImage(),
                                                                                                     transforms.RandomCrop(32),
                                                                                                     transforms.RandomHorizontalFlip(),
                                                                                                     transforms.ToTensor(),
                                                                                                     transforms.Normalize(mean=cinic_mean,std=cinic_std),
                                                                                                     ]))
        train_dl = torch.utils.data.DataLoader(training_set, batch_size=train_bs, shuffle=True)
        logger.info("Len of training set: {}, len of imgs in training set: {}, len of train dl: {}".format(len(training_set), len(training_set.imgs), len(train_dl)))

        test_dl = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(cinic_directory + '/cinic-10-trainlarge/test',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)])), batch_size=test_bs, shuffle=False)

    elif dataset == 'chexpert':
        if dataidxs != None:
            local = True
        else:
            local = False
        training_set = CheXpert_dataset(datadir, train=True, transform=True, dataidxs=dataidxs)
        test_set = CheXpert_dataset(datadir, train=False, transform=True, local = local)

        logger.info("Training data after indexing: {}".format(training_set.__len__()))

        train_dl = data.DataLoader(dataset=training_set, batch_size=train_bs, shuffle=True, num_workers=16)
        test_dl = data.DataLoader(dataset=test_set, batch_size=test_bs, shuffle=False, num_workers=16)

    return train_dl, test_dl


def pdm_prepare_full_weights_cnn(nets, device="cpu"):
    """
    we extract all weights of the conv nets out here:
    """
    weights = []
    for net_i, net in enumerate(nets):
        net_weights = []
        statedict = net.state_dict()

        for param_id, (k, v) in enumerate(statedict.items()):
            if device == "cpu":
                if 'fc' in k or 'classifier' in k:
                    if 'weight' in k:
                        net_weights.append(v.numpy().T)
                    else:
                        net_weights.append(v.numpy())
                elif 'conv' in k or 'features' in k:
                    if 'weight' in k:
                        _weight_shape = v.size()
                        if len(_weight_shape) == 4:
                            net_weights.append(v.numpy().reshape(_weight_shape[0], _weight_shape[1]*_weight_shape[2]*_weight_shape[3]))
                        else:
                            net_weights.append(v.numpy())
                    else:
                        net_weights.append(v.numpy())
            else:
                if 'fc' in k or 'classifier' in k:
                    if 'weight' in k:
                        net_weights.append(v.cpu().numpy().T)
                    else:
                        net_weights.append(v.cpu().numpy())
                elif 'conv' in k or 'features' in k:
                    if 'weight' in k:
                        _weight_shape = v.size()
                        if len(_weight_shape) == 4:
                            net_weights.append(v.cpu().numpy().reshape(_weight_shape[0], _weight_shape[1]*_weight_shape[2]*_weight_shape[3]))
                        else:
                            net_weights.append(v.cpu().numpy())
                    else:
                        net_weights.append(v.cpu().numpy())
        weights.append(net_weights)
    return weights

def pdm_prepare_freq(cls_freqs, n_classes):
    freqs = []

    for net_i in range(len(cls_freqs)):
        net_freqs = [0] * n_classes
        current_class = 0
        for cls_i in cls_freqs[net_i].keys():
            net_freqs[current_class] = cls_freqs[net_i][cls_i]
            current_class += 1

        freqs.append(np.array(net_freqs))

    return freqs

def compute_ensemble_auroc(models: list, dataloader, n_classes, train_cls_counts=None, uniform_weights=False, sanity_weights=False, device="cpu", dataset = None):

    was_training = [False]*len(models)
    for i, model in enumerate(models):
        if model.training:
            was_training[i] = True
            model.eval()

    if uniform_weights is True:
        weights_list = prepare_uniform_weights(n_classes, len(models))
    elif sanity_weights is True:
        weights_list = prepare_sanity_weights(n_classes, len(models))
    else:
        weights_list = prepare_weight_matrix(n_classes, train_cls_counts)

    weights_norm = normalize_weights(weights_list)
    
    target =  np.array([])
    out =  np.array([])
    with torch.no_grad():
        for batch_idx, (x, target_b) in enumerate(dataloader):
                target_b = handle_uncertainty_labels(target_b)
                target_b = handle_NaN_values(target_b)
                x, target_b = x.to(device), target_b.to(device)
                # target_b = target_b.long()
                out_b = get_weighted_average_pred(models, weights_norm, x, device=device)

                target.append(target_b.tolist())
                out.append(out_b.tolist())

    target = np.reshape(target, (-1, 14))
    out = np.reshape(out, (-1, 14))

    auroc = roc_auc_score(target, out)

    if was_training:
        model.train()
    logger.info("auroc score calculated")
    return auroc

def compute_ensemble_accuracy(models: list, dataloader, n_classes, train_cls_counts=None, uniform_weights=False, sanity_weights=False, device="cpu", dataset = None):

    correct, total = 0, 0
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    was_training = [False]*len(models)
    for i, model in enumerate(models):
        if model.training:
            was_training[i] = True
            model.eval()

    if uniform_weights is True:
        weights_list = prepare_uniform_weights(n_classes, len(models))
    elif sanity_weights is True:
        weights_list = prepare_sanity_weights(n_classes, len(models))
    else:
        weights_list = prepare_weight_matrix(n_classes, train_cls_counts)

    weights_norm = normalize_weights(weights_list)

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            target = handle_uncertainty_labels(target)
    
            x, target = x.to(device), target.to(device)
            target = target.long()
            out = get_weighted_average_pred(models, weights_norm, x, device=device)

            _, pred_label = torch.max(out, 1)
            if dataset == "chexpert":
                pred_label = []
                for i in  out:
                    if i >= 0.5:
                        pred_label.append(1)
                    elif i <= -0.5:
                        pred_label.append(-1)
                    elif (i < 0.5 and i > -0.5):
                        pred_label.append(0)
                    else:
                        pred_label.append(None)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    #logger.info(correct, total)

    if dataset == "chexpert":
            conf_matrix = multilabel_confusion_matrix(true_labels_list, pred_labels_list)
    else:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    for i, model in enumerate(models):
        if was_training[i]:
            model.train()

    return correct / float(total), conf_matrix


class ModerateCNNContainerConvBlocks(nn.Module):
    def __init__(self, num_filters, output_dim=10):
        super(ModerateCNNContainerConvBlocks, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=num_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x