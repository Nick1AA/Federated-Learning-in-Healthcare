#!/bin/sh
python main.py --model=densenet121 \
--dataset=chexpert \
--lr=0.0001 \
--retrain_lr=0.0001 \
--batch-size=16 \
--epochs=3 \
--retrain_epochs=2 \
--n_nets=16 \
--partition=homo \
--comm_type=fedavg \
--comm_round=1 \
--oneshot_matching= \
--retrain=;