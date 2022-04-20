#!/bin/sh
python main2_all_param.py --model=densenet121 \
--dataset=chexpert \
--lr=0.0001 \
--retrain_lr=0.0001 \
--batch-size=16 \
--epochs=3 \
--retrain_epochs=2 \
--n_nets=16 \
--partition=homo \
--comm_type=fedma_fedavg \
--comm_round=3 \
--oneshot_matching= \
--retrain=