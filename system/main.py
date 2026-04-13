import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
from sklearn.cluster import KMeans
from flcore.servers.serverbci import FedBCI
from flcore.trainmodel.models import *
from utils.mem_utils import MemReporter
from contextlib import redirect_stdout, redirect_stderr
import random


warnings.simplefilter("ignore")
#torch.manual_seed(0)

# hyper-params for AG News
vocab_size = 98635
max_len = 200
hidden_dim = 32


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "cnn":
            if args.dataset[:5] == "mnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim1=512).to(args.device)
            elif args.dataset[:5] == "emnis":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim1=512).to(args.device)
            elif args.dataset[:5] == "Cifar":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim1=512).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim1=512).to(args.device)


        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        else:
            raise NotImplementedError

        print(args.model)

        if args.algorithm == "FedBCI":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedBCI(args, i)
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

        reporter.report()

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("All done!")


if __name__ == "__main__":
    total_start = time.time()
    set_seed(0)
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="cifar10")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedBCI")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=50,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-lam', "--lamda", type=float, default=4.0)
    parser.add_argument('-nclust', "--num_clusters", type=int, default=4,
                        help="Number of clusters for clustered federated learning")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not available.\n")
        args.device = "cpu"


    log_file_path = 'training_log.txt'

    with open(log_file_path, 'w') as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            run(args)