# Federated-Learning-in-Healthcare
This repository compares the performance of FedAvg and FedMA on the CheXpert dataset. It makes use of the project https://github.com/IBM/FedMA by Wang et al.
Note that the FedMA algorithm in its raw format can not be applied on the DenseNet121 architecture. Therefore adaptions were made that affect the BBP_MAP step of the algorithm.
The resulting simplified version can be run to test the following scenarios:
* Matching batch normalization layers on their own wherby
    * All parameters of the batch normalization layer are considered
    * Only the weight and bias of the batch normalization layer are considered
* Matching with batch normalization layer combined with the convolutional layer (we discovered that this approach does not work on the DenseNet121) wherby
    * All parameters of the batch normalization layer are considered
    * Only the weight and bias of the batch normalization layer are considered
* Matching without considering batch normalization layer
    * Using the local parameters for the batch normalization layer
    * Using averaged batch normalization parameters

A diagram summarizing the structure and dependencies between the files can be found in _structure.uxf_. Note that those files that are not included in this diagram behave similar to a file existing in the diagram. We summarize these files in the following:
   * main_all_param.py, main2.py, main2_all_param.py, main3.py, main3_averaged.py are similar to main.py and depict slightly adapted copies of main.py that allow us to run the different approaches in parallel
   * run_all_param.sh, run2.sh, run2_all_param.sh, run3.sh, run3_averaged.sh, run_FedAvg.sh behave similar to run.sh and describe the shell scripts that allow us to run the different approaches of FedMA and FedAvg
   * model.py is the original file written by Wang et al. and contains analogously to model2.py classes to build models of different NN architectures
   * matching/utils.py and matching/utils2.py have the same relationship as model.py and model2.py
   * validation.py, validation_approach.py, run_validation.sh serve to analyze our trained models on the validation dataset 

## Requirements:
* Download the CheXpert dataset by registering at https://stanfordmlgroup.github.io/competitions/chexpert/ and upload it to the _data_ folder
* Create the _FedMA-CheXpert_ environment using the environment.yml file
    * Install anaconda
    * Open the anaconda prompt and navigate to the folder where you placed this project
    * Type _conda env create -f environment.yml_
* Activate the _FedMA-CheXpert_ environment

## Run the code:
Depending on which scenario you want to run a different shell script needs to be executed. If you want to load trained weights or train the models from scratch minor adaptions to the _main.py_ file and the shell script of the corresponding approach might be necessary.
* Matching batch normalization layers on their own wherby
    * All parameters of the batchnorm layer are considered: execute *run.sh* 
    * Only the weight and bias of the batchnorm layer are considered: execute *run_all_param.sh* 
* Matching with batchnorm combined with the convolutional layer wherby
    * All parameters of the batchnorm layer are considered: execute *run2.sh* 
    * Only the weight and bias of the batchnorm layer are considered: execute *run2_all_param.sh* 
* Matching without considering batchnorm layer
    * Using the local parameters for the batchnorm layer: execute *run3.sh* 
    * Using averaged batchnorm parameters: execute *run3_averaged.sh* 
* FedAvg: execute *run_FedAvg.sh* 
