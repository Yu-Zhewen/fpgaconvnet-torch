import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_avg_channel_sparsity_distribution(data, ax):
    ax.hist(data)
    ax.axvline(data.mean(), color='k', linestyle='dashed', linewidth=1, label="mean = "+str(data.mean()))
    ax.set_title("Histogram of avg sparsity of channels")
    ax.set(xlabel = "Sparsity", ylabel = "Number of Channels")
    ax.legend(loc = "upper right")

def plot_channel_sparsity_avg_histograms(data, ax):
    avg_hist = data.mean(axis = 0)
    ax.bar(list(range(len(avg_hist))), avg_hist)

    hist_sum = 0
    for i in range(len(avg_hist)):
        hist_sum += avg_hist[i]*i
    hist_sum /= sum(avg_hist)

    ax.axvline(hist_sum, color='k', linestyle='dashed', linewidth=1, label="mean "+str(hist_sum))
    ax.set_title("Histogram of avg number of zeros per window")
    ax.set(xlabel = "Number of zeros in window", ylabel = "Number of windows")
    ax.legend(loc = "upper right")

def plot_channel_sparsity_correlation_heatmap(data, ax):
    sns.heatmap(data[:32, :32], annot = True)
    ax.set_title("Heatmap of correlation between channel (reduced to 32 channels)")
    ax.set(xlabel = "Channels", ylabel = "Channels")
    ax.legend(loc = "upper right")

def plot_channel_sparsity_correlation_histogram(data, ax):
    flattened_corr_data = data.flatten()
    flattened_corr_data = flattened_corr_data[flattened_corr_data != 1]/2

    ax.hist(flattened_corr_data, bins=20)
    ax.axvline(flattened_corr_data.mean(), color='k', linestyle='dashed', linewidth=1, label="mean = "+str(flattened_corr_data.mean()))
    ax.set_title("Histogram of correlation of channels")
    ax.set(xlabel = "Correlation", ylabel = "Number of Channels Pairs")
    ax.legend(loc = "upper right")

def visualise_layer(corr_data, hist_data, output_path):
    corr_data[np.abs(corr_data) == np.Inf] = 1
    corr_data[np.abs(corr_data) == np.NaN] = 0
    corr_data[np.abs(corr_data) == np.nan] = 0
    corr_data[np.abs(corr_data) == np.NAN] = 0
    fig, ax = plt.subplots(3)
    fig.set_figheight(20)
    fig.set_figwidth(20)
    fig.suptitle('Sparsity statistics for ' + output_path.split("/")[-1][:-4], fontsize = 16)
    try:
        plot_channel_sparsity_avg_histograms(hist_data, ax[0])
        plot_channel_sparsity_correlation_histogram(corr_data, ax[1])
        plot_channel_sparsity_correlation_heatmap(corr_data, ax[2])
        print("Saving in output path", output_path)
        fig.savefig(output_path)
    except:
        pass

if __name__ == "__main__":
    layer_dir = "/home/ka720/sparseCNN/runlog/resnet18_sparsity_run_50K_relu_0_2023_04_05_13_22_35_310476"
    corr_data = np.load(os.path.join(layer_dir, "resnet18_layer2.0.conv2.1_correlation.npy"))
    hist_data = np.load(os.path.join(layer_dir, "resnet18_layer2.0.conv2.1_histograms.npy"))
    # print(corr_data)
    # corr_data[np.abs(corr_data) == np.Inf] = 1

    # print(corr_data)
    # output = "Generic.png"
    # visualise_layer(corr_data, hist_data, output)
