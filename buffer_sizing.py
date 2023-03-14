import numpy as np

def moving_average(a, n=3):
    assert len(a.shape) == 1
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def detrend_moving_average(a, n=3):
    return moving_average(a-np.average(a), n=n)

def max_distance_metric(streams, n=4):
    assert len(streams.shape) == 2
    ma = []
    for i in range(streams.shape[1]):
        # ma.append(detrend_moving_average(streams[:,i], n=n))
        ma.append(moving_average(streams[:,i], n=n))
    ma = np.vstack(ma)
    avg = np.average(streams, axis=0)
    avg_max = np.amax(avg)
    avg_min = np.amin(avg)
    avg_diff = avg_max-avg_min
    # get the max and min across streams dimension
    ma_max = np.amax(ma, axis=0)
    ma_min = np.amin(ma, axis=0)
    ma_diff = np.absolute(ma_max - ma_min)
    return np.average(ma_diff) - avg_diff

# load test data
# test_data = np.load("runlog/resnet18_sparsity_run_50k_2023_03_14_18_15_43_840440/resnet18_layer4.1.conv1.1_sparsity.npy")
# test_data = np.load("runlog/resnet18_sparsity_run_50k_2023_03_14_18_15_43_840440/resnet18_layer1.0.conv1.1_sparsity.npy")
test_data = np.load("runlog/resnet18_sparsity_run_50k_2023_03_14_18_15_43_840440/resnet18_layer2.1.conv1.1_sparsity.npy")

print(test_data.shape)

# print(moving_average(test_data[:,0], n=10).shape)
print(max_distance_metric(test_data, n=10000))
