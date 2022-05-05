# Federated-Learning-in-Healthcare
This repository compares the performance of FedAvg and FedMA on the CheXpert dataset.
Note that the FedMA algorithm in its raw format can not be applied on the densenet121 architecture. Therefore adaptions were made that affect the BBP_MAP step of the algorithm.
The resulting simplified version can be run to test the following scenarios:
* Matching batch normalization layers on their own wherby
    * All parameters of the batch normalization layer are considered
    * Only the weight and bias of the batch normalization layer are considered
* Matching with batch normalization layer combined with the convolutional layer wherby
    * All parameters of the batch normalization layer are considered
    * Only the weight and bias of the batch normalization layer are considered
* Matching without considering batch normalization layer
    * Using the local parameters for the batch normalization layer
    * Using averaged batch normalization parameters

## Requirements:
* Download the CheXpert dataset by registering at https://stanfordmlgroup.github.io/competitions/chexpert/ and upload it to the _data_ folder
* Create the _FedLE_ environment using the environment.yml file
    * Install anaconda
    * Open the anaconda prompt and navigate to the folder where you placed this project
    * Type _conda env create -f environment.yml_

## Run the code:
Depending on which scenario you want to run small adjustments in _main.py_ need to be made.
* Matching batch normalization layers on their own wherby
    * All parameters of the batchnorm layer are considered:
        *Make sure that the method *BBP_MAP_not_layerwise* is called at line 3344 and line 3177
        *Make sure that the method *layer_wise_group_descent_all_param* is called at line 2788
        *Make sure that the method *compute_local_model_auroc* is called at line 3384 and line 3183
        *Make sure that the method *compute_local_model_auroc_global_dataset* is called at line 3405 and line 3204
        *Make sure that the method *match_global_to_local_weights* is called at line 3171
        *Make sure that the method *local_retrain* is called at line 3172 
    * Only the weight and bias of the batchnorm layer are considered:
        *Make sure that the method *BBP_MAP_not_layerwise* is called at line 3344 and line 3177
        *Make sure that the method *layer_wise_group_descent* is called at line 2788
        *Make sure that the method *compute_local_model_auroc* is called at line 3384 and line 3183
        *Make sure that the method *compute_local_model_auroc_global_dataset* is called at line 3405 and line 3204
        *Make sure that the method *match_global_to_local_weights* is called at line 3171
        *Make sure that the method *local_retrain* is called at line 3172
* Matching with batchnorm combined with the convolutional layer wherby
    * All parameters of the batchnorm layer are considered
        *Make sure that the method *BBP_MAP_not_layerwise_2* is called at line 3344 and line 3177
        *Make sure that the method *layer_wise_group_descent_2_all_param* is called at line 2882
        *Make sure that the method *compute_local_model_auroc_2* is called at line 3384 and line 3183
        *Make sure that the method *compute_local_model_auroc_global_dataset_2* is called at line 3405 and line 3204
        *Make sure that the method *match_global_to_local_weights_2* is called at line 3171
        *Make sure that the method *local_retrain_2* is called at line 3172
    * Only the weight and bias of the batchnorm layer are considered
        *Make sure that the method *BBP_MAP_not_layerwise_2* is called at line 3344 and line 3177
        *Make sure that the method *layer_wise_group_descent_2* is called at line 2882
        *Make sure that the method *compute_local_model_auroc_2* is called at line 3384 and line 3183
        *Make sure that the method *compute_local_model_auroc_global_dataset_2* is called at line 3405 and line 3204
        *Make sure that the method *match_global_to_local_weights_2* is called at line 3171
        *Make sure that the method *local_retrain_2* is called at line 3172
* Matching without considering batchnorm layer
    * Using the local parameters for the batchnorm layer:
        *Make sure that the method *BBP_MAP_not_layerwise_2* is called at line 3344 and line 3177
        *Make sure that the method *layer_wise_group_descent_3* is called at line 2882
        *Make sure that the method *compute_local_model_auroc* is called at line 3384 and line 3183
        *Make sure that the method *compute_local_model_auroc_global_dataset* is called at line 3405 and line 3204
        *Make sure that the method *match_global_to_local_weights_2* is called at line 3171
        *Make sure that the method *local_retrain_2* is called at line 3172
        *Make sure to comment lines 2935,2936 and uncomment 2931-2934
        *In _matching/pfnm.py_ make sure to uncomment and comment according to the notes in the code
    * Using averaged batchnorm parameters:
        *Make sure that the method *BBP_MAP_not_layerwise_2* is called at line 3344 and line 3177
        *Make sure that the method *layer_wise_group_descent_3* is called at line 2882
        *Make sure that the method *compute_local_model_auroc* is called at line 3384 and line 3183
        *Make sure that the method *compute_local_model_auroc_global_dataset* is called at line 3405 and line 3204
        *Make sure that the method *match_global_to_local_weights_2* is called at line 3171
        *Make sure that the method *local_retrain_2* is called at line 3172
        *Make sure to uncomment lines 2935,2936 and comment 2931-2934
        *In _matching/pfnm.py_ make sure to uncomment and comment according to the notes in the code




* Activate the _FedLE_ environment

Necessary libraries can be extracted from environment.yml. Additionally scikit-learn, accimage, opencv, matplotlib, pillow-simp and lapsolver have been installed
