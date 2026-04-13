# FedBCI: Reliable Clustered Federated Learning by
Bridging Intra- and Inter-Cluster Inconsistencies

## Dataset Partitioning

We adopt the data partitioning implementation from PFLlib and provide the partitioning results for CIFAR-100-exdir0.1 as a zip file.


## Run the code

CUDA_VISIBLE_DEVICES=0 python -u main.py -t 1 -jr 1 -nc 50 -nb 100 -data Cifar100-exdir0.1 -m cnn -algo FedBCI -did 0 -lam 4

## Acknowledgement

Some parts of our code and implementation has been adapted from PFLlib repository.



