""" die in dieser Datei befindlichen Methoden und Klassen im Zusammenhang mit dem Datensatz dienen als Referenz und müsse noch angepasst werden """

import os
import numpy as np
import torch
import torch.nn as nn
import logging
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.metrics import confusion_matrix

# we've changed to a faster solver
#from scipy.optimize import linear_sum_assignment
import logging

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from datasets import MNIST_truncated, CIFAR10_truncated, ImageFolderTruncated, CIFAR10ColorGrayScaleTruncated
from combine_nets import prepare_uniform_weights, prepare_sanity_weights, prepare_weight_matrix, normalize_weights, get_weighted_average_pred

from vgg import *
from model import *
from model2 import *

from typing import Any
from types import FunctionType

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

""" These methods have been added additionally for the DenseNet
taken from https://github.com/pytorch/vision/tree/main/torchvision """
def _log_api_usage_once(obj: Any) -> None:
    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")

try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401

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

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
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

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir":
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

    elif partition == "hetero-fbs":
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

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())               

    if get_confusion_matrix:
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

def load_model(model, model_index, rank=0, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
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
                            pass
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
                            pass
                    else:
                        net_weights.append(v.cpu().numpy())
        weights.append(net_weights)
    return weights

def pdm_prepare_freq(cls_freqs, n_classes):
    freqs = []

    for net_i in sorted(cls_freqs.keys()):
        net_freqs = [0] * n_classes

        for cls_i in cls_freqs[net_i]:
            net_freqs[cls_i] = cls_freqs[net_i][cls_i]

        freqs.append(np.array(net_freqs))

    return freqs


def compute_ensemble_accuracy(models: list, dataloader, n_classes, train_cls_counts=None, uniform_weights=False, sanity_weights=False, device="cpu"):

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
            x, target = x.to(device), target.to(device)
            target = target.long()
            out = get_weighted_average_pred(models, weights_norm, x, device=device)

            _, pred_label = torch.max(out, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    #logger.info(correct, total)

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