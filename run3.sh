#!/bin/sh
# cd $TMP;
# tar xf /pfs/work7/workspace/scratch/sq8430-data_workspace/move_data.tar;
python main3.py --model=densenet121 \
--dataset=chexpert \
--lr=0.0001 \
--retrain_lr=0.0001 \
--batch-size=16 \
--epochs=3 \
--retrain_epochs=2 \
--n_nets=16 \
--partition=homo \
--comm_type=fedma \
--comm_round=2 \
--oneshot_matching= \
--retrain=;
# rsync -zvh trained_weightsPre_FedMA_comm_not_layerwise_3 /pfs/work7/workspace/scratch/sq8430-data_workspace/;
# rsync -zvh matched_weights_not_layerwise_3 /pfs/work7/workspace/scratch/sq8430-data_workspace/;
# rsync -zvh trained_weightsFedMA_comm_not_layerwise_3 /pfs/work7/workspace/scratch/sq8430-data_workspace/;