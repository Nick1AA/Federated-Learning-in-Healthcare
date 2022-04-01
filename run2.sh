#!/bin/sh
python main.py --model=densenet121 \
--dataset=chexpert \
--lr=0.0001 \
--retrain_lr=0.0001 \
--batch-size=16 \
--epochs=3 \
--retrain_epochs=3 \
--n_nets=16 \
--partition=homo \
--comm_type=fedma_fedavg \
--comm_round=10 \
--oneshot_matching= \
--retrain= 