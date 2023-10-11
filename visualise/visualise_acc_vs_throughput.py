import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", default = None, type = str,
                        help = "Path to .csv file with accuracy and throughput")
    parser.add_argument("--output", default = None, type = str,
                        help = "Path to .png file to save to")

    args = parser.parse_args()

    data = pd.read_csv(args.filepath)

    fig, ax1 = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(16)
    ax2 = ax1.twinx()

    ax1.plot(data["ReLU_Threshold"], data["Top5_Accuracy"], label="Accuracy")
    ax1.axhline(data["Top5_Accuracy"].max() - 1, color='k', linestyle='dashed', linewidth=1, label="Accuracy Loss = 1%")
    ax2.plot(data["ReLU_Threshold"], data["Throughput"]/(data["Throughput"].min()), label="Throughput", color = 'r')


    fig.suptitle('Overview of relu thresholding for resnet18')
    ax1.set(xlabel = "Relu Threshold", ylabel = "Accuracy")
    ax2.set(xlabel = "Relu Threshold", ylabel = "Normalised Throughput")
    ax1.legend(loc = "best")
    ax2.legend(loc = "best")

    fig.savefig(args.output)