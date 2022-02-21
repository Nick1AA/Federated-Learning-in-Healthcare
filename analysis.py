""" In this file the analysis takes place"""
from model2 import densenet121
from sklearn import metrics
import torch
from datasets import CheXpert_dataset
from torch.utils import data
from utils import handle_uncertainty_labels
import matplotlib.pyplot as plt
import numpy as np
import logging

args_datadir = "./data/CheXpert-v1.0-small/"

def analyze(models_1, models_2, global_weights_1, global_weights_2, args):
    global_model = densenet121()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    validation_set = CheXpert_dataset(args_datadir, valid=True, transform=True)
    validation_dl = data.DataLoader(dataset=validation_set, batch_size=validation_set.__len__(), shuffle=True)

    new_state_dict = {}
    model_counter = 0
    # starting with fedavg weights which are stored in variable global_weights_1
    for param_idx, (key_name, param) in enumerate(global_model.state_dict().items()):
        if "conv" in key_name or "features" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(global_weights_1[param_idx].reshape(param.size()))}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(global_weights_1[param_idx])}
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(global_weights_1[param_idx].T)}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(global_weights_1[param_idx])}

        new_state_dict.update(temp_dict)
    global_model.load_state_dict(new_state_dict)

    target, out, data = []
    # there should be only one batch containing all of the data
    for batch_idx, (x, target_b) in enumerate(validation_dl):
            handle_uncertainty_labels(target)
            if args.dataset == "chexpert":
                target_new = []
                for z in range(len(target_b)):
                    target_new.append(target_b[z][1:15])
                target_b = target_new  
            x, target_b = x.to(device), target.to(device)
            out_b = global_model(x)

            data.append(x)
            target.append(target_b)
            out.append(out_b)

    target = np.reshape(target, (-1, 14))
    out = np.reshape(out, (-1, 14))

    auroc = metrics.roc_auc_score(target, out)
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(target, out)
    logging.info("The global model with weights from FedAvg has a total AUROC of: " + auroc)

    # Plot ROC curve and AUROC for the whole input
    plt.figure()
    lw = 2
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % auroc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic total (FedAvg)")
    plt.legend(loc="lower right")
    plt.show()

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

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(observation_classes)):
        fpr[i], tpr[i], _ = metrics.roc_curve(target[:, i], out[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Plot of a ROC curve for each class
    for i in range (len(observation_classes)):
        plt.figure()
        lw = 2
        plt.plot(
            fpr[i],
            tpr[i],
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc[i],
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic " + observation_classes[i] + " (FedAvg)")
        plt.legend(loc="lower right")
        plt.show()

    # continuing with FedMA weights which are stored in variable global_weights_2
    for param_idx, (key_name, param) in enumerate(global_model.state_dict().items()):
        if "conv" in key_name or "features" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(global_weights_2[param_idx].reshape(param.size()))}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(global_weights_2[param_idx])}
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(global_weights_2[param_idx].T)}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(global_weights_2[param_idx])}

        new_state_dict.update(temp_dict)
    global_model.load_state_dict(new_state_dict)

    out = []
    for x in data:
        out_b = global_model(x)
        out.append(out_b)

    out = np.reshape(out, (-1, 14))

    auroc = metrics.roc_auc_score(target, out)
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(target, out)
    logging.info("The global model with weights from FedMA has a total AUROC of: " + auroc)

    # Plot ROC curve and AUROC for the whole input
    plt.figure()
    lw = 2
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[2],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic total (FedMA)")
    plt.legend(loc="lower right")
    plt.show()

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(observation_classes)):
        fpr[i], tpr[i], _ = metrics.roc_curve(target[:, i], out[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Plot of a ROC curve for each class
    for i in range (len(observation_classes)):
        plt.figure()
        lw = 2
        plt.plot(
            fpr[i],
            tpr[i],
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc[i],
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic " + observation_classes[i] + " (FedMA)")
        plt.legend(loc="lower right")
        plt.show()

    # Now we switch from a global level to the local models to check their auroc
    # The local models using FedAvg
    counter = 1
    for model in models_1:
        out = []
        for x in data:
            out_b = model(x)
            out.append(out_b)

        out = np.reshape(out, (-1, 14))

        auroc = metrics.roc_auc_score(target, out)
        false_positive_rate, true_positive_rate, _ = metrics.roc_curve(target, out)
        logging.info("The local model " + counter + " with weights from FedAvg has a total AUROC of: " + auroc)

        # Plot ROC curve and AUROC for the whole input
        plt.figure()
        lw = 2
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc[2],
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic total (FedAvg)")
        plt.legend(loc="lower right")
        plt.show()

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(observation_classes)):
            fpr[i], tpr[i], _ = metrics.roc_curve(target[:, i], out[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # logging.info ROC area for each class
        for i in range (len(observation_classes)):
            logging.info("The local model " + counter + " with weights from FedAvg has an AUROC for class " + observation_classes[i] + 
            " of: " + roc_auc[i])
        counter += 1

    # The local models using FedMA
    counter = 1
    for model in models_2:
        out = []
        for x in data:
            out_b = model(x)
            out.append(out_b)

        out = np.reshape(out, (-1, 14))

        auroc = metrics.roc_auc_score(target, out)
        false_positive_rate, true_positive_rate, _ = metrics.roc_curve(target, out)
        logging.info("The local model " + counter + " with weights from FedMA has a total AUROC of: " + auroc)

        # Plot ROC curve and AUROC for the whole input
        plt.figure()
        lw = 2
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc[2],
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic total (FedMA)")
        plt.legend(loc="lower right")
        plt.show()

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(observation_classes)):
            fpr[i], tpr[i], _ = metrics.roc_curve(target[:, i], out[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # logging.info ROC area for each class
        for i in range (len(observation_classes)):
            logging.info("The local model " + counter + " with weights from FedMA has an AUROC for class " + observation_classes[i] + 
            " of: " + roc_auc[i])
        counter += 1
