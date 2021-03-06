import torch
import torch.nn.functional as F
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger("combine_nets")



def prepare_weight_matrix(n_classes, weights: dict):
    """
    Creates a weight matrix from a dictionary

    : weights: parameters to be formatted into a matrix
    """
    weights_list = {}

    for net_i, cls_cnts in weights.items():
        cls = np.array(list(cls_cnts.keys()))
        cnts = np.array(list(cls_cnts.values()))
        weights_list[net_i] = np.array([0] * n_classes, dtype=np.float32)
        weights_list[net_i][cls] = cnts
        weights_list[net_i] = torch.from_numpy(weights_list[net_i]).view(1, -1)

    return weights_list


def prepare_uniform_weights(n_classes, net_cnt, fill_val=1):
    """
    Creates a weight matrix containing only one value

    : n_classes: number of classes - determines columns
    : net_cnt: number of clients - determines rows
    fill_val: value to be filled in the matrix
    """

    weights_list = {}

    for net_i in range(net_cnt):
        temp = np.array([fill_val] * n_classes, dtype=np.float32)
        weights_list[net_i] = torch.from_numpy(temp).view(1, -1)

    return weights_list


def prepare_sanity_weights(n_classes, net_cnt):
    """
    Creates a weight matrix containing only 0 values

    : n_classes: number of classes - determines columns
    : net_cnt: number of clients - determines rows
    """
    return prepare_uniform_weights(n_classes, net_cnt, fill_val=0)


def normalize_weights(weights):
    """
    normalizes the weights
    """

    Z = np.array([])
    eps = 1e-6
    weights_norm = {}

    for _, weight in weights.items():
        if len(Z) == 0:
            Z = weight.data.numpy()
        else:
            Z = Z + weight.data.numpy()

    for mi, weight in weights.items():
        weights_norm[mi] = weight / torch.from_numpy(Z + eps)

    return weights_norm


def get_weighted_average_pred(models: list, weights: dict, x, device="cpu"):
    """
    Returns the weighted output of a list of models on some data batch x

    : models: list of models
    : weights: weights to apply on the output
    : x: batch
    : device: device on which the computation takes place (cpu or gpu)
    """
    out_weighted = None

    # Compute the predictions
    for model_i, model in enumerate(models):
        #logger.info("Model: {}".format(next(model.parameters()).device))
        #logger.info("data device: {}".format(x.device))
        out = F.softmax(model(x), dim=-1)  # (N, C)

        weight = weights[model_i].to(device)

        if out_weighted is None:
            weight = weight.to(device)
            out_weighted = (out * weight)
        else:
            out_weighted += (out * weight)

    return out_weighted

