import os
from visualise_network_sparsity import visualise_network
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", metavar='DIR', required = True, help = "directory with sparsity stats of all networks to be visualised")

    args = parser.parse_args()
    for sparsity_data in os.listdir(args.data):
        print("Visualing network:", sparsity_data)
        visualise_network(os.path.join(args.data, sparsity_data))
