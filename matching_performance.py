import logging
from model import *
from model2 import densenet121Container, densenet121
from utils import *

from vgg import *
from vgg import matched_vgg11

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger("matching_performance")

def compute_model_averaging_accuracy(models, weights, train_dl, test_dl, n_classes, args, device = "cpu"):
    """An variant of fedaveraging"""
    if args.model == "lenet":
        avg_cnn = LeNet()
    elif args.model == "vgg":
        avg_cnn = vgg11()
    elif args.model == "simple-cnn":
        if args.dataset in ("cifar10", "cinic10"):
            avg_cnn = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
        elif args.dataset == "mnist":
            avg_cnn = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
    elif args.model == "moderate-cnn":
        if args.dataset in ("cifar10", "cinic10"):
            avg_cnn = ModerateCNN()
        elif args.dataset == "mnist":
            avg_cnn = ModerateCNNMNIST()
    elif args.model == "densenet121":
        avg_cnn = densenet121()
    
    new_state_dict = {}
    model_counter = 0

    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(avg_cnn.state_dict().items()):
        if "num_batches_tracked" in key_name:
            continue
        if "conv" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        elif "norm" in key_name:
            temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}

        new_state_dict.update(temp_dict)

    avg_cnn.load_state_dict(new_state_dict)

    # switch to eval mode:
    avg_cnn.eval()
    ##    
    
    if not (args.dataset == "chexpert"):
        correct, total = 0, 0
        for batch_idx, (x, target) in enumerate(test_dl):
            target = handle_uncertainty_labels(target)
            
            out_k = avg_cnn(x)
            _, pred_label = torch.max(out_k, 1)
            if args.dataset == "chexpert":
                    pred_label = []
                    for i in  out_k:
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
            
        logger.info("Accuracy for Fed Averaging correct: {}, total: {}".format(correct, total))
    else:
        auroc = compute_accuracy(avg_cnn, test_dl, device, dataset = args.dataset)
        logger.info("AUROC score for Fed Averaging: {}".format(auroc))

def compute_local_model_auroc(models, hungarian_weights, assignments, net_dataidx_map, args_datadir, n_classes, args, device = "cpu"):
    
    
    new_weights = []
    for i in range(args.n_nets):
        logger.info("Analyzing AUROC score of local model {}".format(i))
        dataidxs = net_dataidx_map[i]
        train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)
        # train_auroc = compute_auroc(models[i], train_dl_local, device, dataset = args.dataset)
        # test_auroc = compute_auroc(models[i], test_dl_local, device, dataset = args.dataset)
        # logger.info("auroc score with local weights on local training dataset: {}".format(train_auroc))
        # logger.info("auroc score with local weights on local test dataset: {}".format(test_auroc))

        if len(hungarian_weights[0]) == args.n_nets:
            new_local_weights = match_global_to_local_weights(hungarian_weights, assignments, i, not_layerwise = True)
        else:
            new_local_weights = match_global_to_local_weights(hungarian_weights, assignments, i)
        new_weights.append(new_local_weights)
    
    logger.info("===================================================================")
    matched_cnn = densenet121()
    train_auroc_list = []
    test_auroc_list = []

    for i in range(args.n_nets):
        logger.info("Analyzing AUROC score of local model {}".format(i))
        new_state_dict = {}
        for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
            if "num_batches_tracked" in key_name:
                continue
            if "conv" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx])}
            elif "norm" in key_name:
                temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].reshape(param.size()))}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx])}

            new_state_dict.update(temp_dict)

        matched_cnn.load_state_dict(new_state_dict)
        matched_cnn.eval()
        dataidxs = net_dataidx_map[i]
        train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)
        train_auroc = compute_auroc(matched_cnn, train_dl_local, device, dataset = args.dataset)
        test_auroc = compute_auroc(matched_cnn, test_dl_local, device, dataset = args.dataset)
        logger.info("AUROC score with matched weights on local training dataset: {}".format(train_auroc))
        logger.info("AUROC score with matched weights on local test dataset: {}".format(test_auroc))
        train_auroc_list.append(train_auroc)
        test_auroc_list.append(test_auroc)
    
    return train_auroc_list, test_auroc_list

def compute_local_model_auroc_global_dataset(models, hungarian_weights, assignments, test_data, args_datadir, n_classes, args, device = "cpu"):
    """An variant of fedaveraging"""
    old_weights = pdm_prepare_full_weights_cnn(models, device)
    new_weights = []
    for i in range(args.n_nets):
        logger.info("Analyzing AUROC score of local model {}".format(i))
        test_auroc = compute_auroc(models[i], test_data, device, dataset = args.dataset)
        logger.info("AUROC score with local weights on global test dataset: {}".format(test_auroc))

        if len(hungarian_weights[0]) == args.n_nets:
            new_local_weights = match_global_to_local_weights(hungarian_weights, assignments, i, not_layerwise = True)
        else:
            new_local_weights = match_global_to_local_weights(hungarian_weights, assignments, i)
        new_weights.append(new_local_weights)
    logger.info("===================================================================")
    matched_cnn = densenet121()
    test_auroc_list = []

    for i in range(args.n_nets):
        logger.info("Analyzing AUROC score of local model {}".format(i))
        new_state_dict = {}
        for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
            if "num_batches_tracked" in key_name:
                continue
            if "conv" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx])}
            elif "norm" in key_name:
                temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].reshape(param.size()))}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx])}

            new_state_dict.update(temp_dict)

        matched_cnn.load_state_dict(new_state_dict)
        matched_cnn.eval()

        test_auroc = compute_auroc(matched_cnn, test_data, device, dataset = args.dataset)
        logger.info("AUROC score with matched weights on global test dataset: {}".format(test_auroc))
        test_auroc_list.append(test_auroc)
    
    return test_auroc_list

def compute_local_model_auroc_2(models, hungarian_weights, assignments, net_dataidx_map, args_datadir, n_classes, args, device = "cpu"):
    """An variant of fedaveraging"""
    
    new_weights = []
    for i in range(args.n_nets):
        logger.info("Analyzing AUROC score of local model {}".format(i))
        dataidxs = net_dataidx_map[i]
        train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)
        # train_auroc = compute_auroc(models[i], train_dl_local, device, dataset = args.dataset)
        # test_auroc = compute_auroc(models[i], test_dl_local, device, dataset = args.dataset)
        # logger.info("auroc score with local weights on local training dataset: {}".format(train_auroc))
        # logger.info("auroc score with local weights on local test dataset: {}".format(test_auroc))

        if len(hungarian_weights[0]) == args.n_nets:
            new_local_weights = match_global_to_local_weights_2(hungarian_weights, assignments, i, not_layerwise = True)
        else:
            new_local_weights = match_global_to_local_weights_2(hungarian_weights, assignments, i)
        new_weights.append(new_local_weights)
    
    logger.info("===================================================================")
    matched_cnn = densenet121()
    train_auroc_list = []
    test_auroc_list = []

    for i in range(args.n_nets):
        logger.info("Analyzing AUROC score of local model {}".format(i))
        new_state_dict = {}
        for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
            if "num_batches_tracked" in key_name:
                continue
            if "conv" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx])}
            elif "norm" in key_name:
                temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].reshape(param.size()))}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx])}

            new_state_dict.update(temp_dict)

        matched_cnn.load_state_dict(new_state_dict)
        matched_cnn.eval()
        dataidxs = net_dataidx_map[i]
        train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)
        train_auroc = compute_auroc(matched_cnn, train_dl_local, device, dataset = args.dataset)
        test_auroc = compute_auroc(matched_cnn, test_dl_local, device, dataset = args.dataset)
        logger.info("AUROC score with matched weights on local training dataset: {}".format(train_auroc))
        logger.info("AUROC score with matched weights on local test dataset: {}".format(test_auroc))
        train_auroc_list.append(train_auroc)
        test_auroc_list.append(test_auroc)
    
    return train_auroc_list, test_auroc_list

def compute_local_model_auroc_global_dataset_2(models, hungarian_weights, assignments, test_data, args_datadir, n_classes, args, device = "cpu"):
    """An variant of fedaveraging"""
    old_weights = pdm_prepare_full_weights_cnn(models, device)
    new_weights = []
    for i in range(args.n_nets):
        logger.info("Analyzing AUROC score of local model {}".format(i))
        test_auroc = compute_auroc(models[i], test_data, device, dataset = args.dataset)
        logger.info("AUROC score with local weights on global test dataset: {}".format(test_auroc))

        if len(hungarian_weights[0]) == args.n_nets:
            new_local_weights = match_global_to_local_weights_2(hungarian_weights, assignments, i, not_layerwise = True)
        else:
            new_local_weights = match_global_to_local_weights_2(hungarian_weights, assignments, i)
        new_weights.append(new_local_weights)
    logger.info("===================================================================")
    matched_cnn = densenet121()
    test_auroc_list = []

    for i in range(args.n_nets):
        logger.info("Analyzing AUROC score of local model {}".format(i))
        new_state_dict = {}
        for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
            if "num_batches_tracked" in key_name:
                continue
            if "conv" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx])}
            elif "norm" in key_name:
                temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].reshape(param.size()))}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx])}

            new_state_dict.update(temp_dict)

        matched_cnn.load_state_dict(new_state_dict)
        matched_cnn.eval()

        test_auroc = compute_auroc(matched_cnn, test_data, device, dataset = args.dataset)
        logger.info("AUROC score with matched weights on global test dataset: {}".format(test_auroc))
        test_auroc_list.append(test_auroc)
    
    return test_auroc_list

def compute_local_model_auroc_3(models, hungarian_weights, assignments, net_dataidx_map, args_datadir, n_classes, args, device = "cpu"):
    """An variant of fedaveraging"""
    
    new_weights = []
    for i in range(args.n_nets):
        logger.info("Analyzing AUROC score of local model {}".format(i))
        dataidxs = net_dataidx_map[i]
        train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)
        # train_auroc = compute_auroc(models[i], train_dl_local, device, dataset = args.dataset)
        # test_auroc = compute_auroc(models[i], test_dl_local, device, dataset = args.dataset)
        # logger.info("auroc score with local weights on local training dataset: {}".format(train_auroc))
        # logger.info("auroc score with local weights on local test dataset: {}".format(test_auroc))

        if len(hungarian_weights[0]) == args.n_nets:
            new_local_weights = match_global_to_local_weights_3(hungarian_weights, assignments, i, not_layerwise = True)
        else:
            new_local_weights = match_global_to_local_weights_3(hungarian_weights, assignments, i)
        new_weights.append(new_local_weights)
    
    logger.info("===================================================================")
    matched_cnn = densenet121()
    train_auroc_list = []
    test_auroc_list = []

    for i in range(args.n_nets):
        logger.info("Analyzing AUROC score of local model {}".format(i))
        new_state_dict = {}
        for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
            if "num_batches_tracked" in key_name:
                continue
            if "conv" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx])}
            elif "norm" in key_name:
                temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].reshape(param.size()))}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx])}

            new_state_dict.update(temp_dict)

        matched_cnn.load_state_dict(new_state_dict)
        matched_cnn.eval()
        dataidxs = net_dataidx_map[i]
        train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)
        train_auroc = compute_auroc(matched_cnn, train_dl_local, device, dataset = args.dataset)
        test_auroc = compute_auroc(matched_cnn, test_dl_local, device, dataset = args.dataset)
        logger.info("AUROC score with matched weights on local training dataset: {}".format(train_auroc))
        logger.info("AUROC score with matched weights on local test dataset: {}".format(test_auroc))
        train_auroc_list.append(train_auroc)
        test_auroc_list.append(test_auroc)
    
    return train_auroc_list, test_auroc_list

def compute_local_model_auroc_global_dataset_3(models, hungarian_weights, assignments, test_data, args_datadir, n_classes, args, device = "cpu"):
    """An variant of fedaveraging"""
    old_weights = pdm_prepare_full_weights_cnn(models, device)
    new_weights = []
    for i in range(args.n_nets):
        logger.info("Analyzing AUROC score of local model {}".format(i))
        test_auroc = compute_auroc(models[i], test_data, device, dataset = args.dataset)
        logger.info("AUROC score with local weights on global test dataset: {}".format(test_auroc))

        if len(hungarian_weights[0]) == args.n_nets:
            new_local_weights = match_global_to_local_weights_3(hungarian_weights, assignments, i, not_layerwise = True)
        else:
            new_local_weights = match_global_to_local_weights_3(hungarian_weights, assignments, i)
        new_weights.append(new_local_weights)
    logger.info("===================================================================")
    matched_cnn = densenet121()
    test_auroc_list = []

    for i in range(args.n_nets):
        logger.info("Analyzing AUROC score of local model {}".format(i))
        new_state_dict = {}
        for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
            if "num_batches_tracked" in key_name:
                continue
            if "conv" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx])}
            elif "norm" in key_name:
                temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].reshape(param.size()))}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(new_weights[i][param_idx])}

            new_state_dict.update(temp_dict)

        matched_cnn.load_state_dict(new_state_dict)
        matched_cnn.eval()

        test_auroc = compute_auroc(matched_cnn, test_data, device, dataset = args.dataset)
        logger.info("AUROC score with matched weights on global test dataset: {}".format(test_auroc))
        test_auroc_list.append(test_auroc)
    
    return test_auroc_list

def compute_full_cnn_accuracy(models, weights, train_dl, test_dl, n_classes, device, args):
    """Note that we only handle the FC weights for now"""
    # we need to figure out the FC dims first

    #LeNetContainer
    # def __init__(self, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10)

    # this should be safe to be hard-coded since most of the modern image classification dataset are in RGB format
    #args_n_nets = len(models)

    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]

        output_dim = weights[-1].shape[0]

        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset == "mnist":
            input_channel = 1
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=10)
    elif args.model == "densenet121":
        num_filters = []
        logging.info("These are the hungarian weights")
        logging.info(weights)
        # densenet121 hat insgesamt 242 conv oder norm Schichten inkl. eine classifier Schicht
        for i in range (len(weights)):
            num_filters.append(np.array(weights[densenet_index(i)]).shape[0])
        matched_cnn = densenet121Container(num_filters = num_filters, layer_index = 300)
    elif args.model == "moderate-cnn":
        #[(35, 27), (35,), (68, 315), (68,), (132, 612), (132,), (132, 1188), (132,), 
        #(260, 1188), (260,), (260, 2340), (260,), 
        #(4160, 1025), (1025,), (1025, 515), (515,), (515, 10), (10,)]
        num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
        input_dim = weights[12].shape[0]
        hidden_dims = [weights[12].shape[1], weights[14].shape[1]]
        if args.dataset in ("cifar10", "cinic10"):
            matched_cnn = ModerateCNNContainer(3,
                                                num_filters, 
                                                kernel_size=3, 
                                                input_dim=input_dim, 
                                                hidden_dims=hidden_dims, 
                                                output_dim=10)
        elif args.dataset == "mnist":
            matched_cnn = ModerateCNNContainer(1,
                                                num_filters, 
                                                kernel_size=3, 
                                                input_dim=input_dim, 
                                                hidden_dims=hidden_dims, 
                                                output_dim=10)

    #logger.info("Keys of layers of convblock ...")
    new_state_dict = {}
    model_counter = 0
    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        #print("&"*30)
        #print("Key: {}, Weight Shape: {}, Matched weight shape: {}".format(key_name, param.size(), weights[param_idx].shape))
        #print("&"*30)
        if "conv" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        elif "norm" in key_name:
            temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}

        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    matched_cnn.to(device)
    matched_cnn.eval()

    ##    
    if not (args.dataset == "chexpert"):
        correct, total = 0, 0
        for batch_idx, (x, target) in enumerate(test_dl):
            target = handle_uncertainty_labels(target)
            
            x, target = x.to(device), target.to(device)
            out_k = matched_cnn(x)
            _, pred_label = torch.max(out_k, 1)
            if args.dataset == "chexpert":
                    pred_label = []
                    for i in  out_k:
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
            
        logger.info("Accuracy for Neural Matching correct: {}, total: {}".format(correct, total))
    else:
        auroc = compute_accuracy(matched_cnn, test_dl, device, dataset = args.dataset)
        logger.info("auroc score for Neural Matching: {}".format(auroc))