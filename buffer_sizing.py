import numpy as np
import itertools

STREAMS=2

def moving_average(a, n=3):
    assert len(a.shape) == 1
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def detrend_moving_average(a, n=3):
    return moving_average(a, n=n)-np.average(a)

def max_distance_metric(streams, n=4):
    assert len(streams.shape) == 2
    avg = np.average(streams, axis=0)
    avg_max = np.amax(avg)
    avg_min = np.amin(avg)
    avg_diff = avg_max-avg_min
    ma = []
    for i in range(streams.shape[1]):
        ma.append(detrend_moving_average(streams[:,i], n=n))
        # ma.append(moving_average(streams[:,i], n=n))
    ma = np.vstack(ma)
    # get the max and min across streams dimension
    ma_max = np.amax(ma, axis=0)
    ma_min = np.amin(ma, axis=0)
    ma_diff = np.absolute(ma_max - ma_min)
    # return np.average(ma_diff) - avg_diff
    return np.average(ma_diff)

# load test data
# test_data = np.load("runlog/resnet18_sparsity_run_50k_2023_03_14_18_15_43_840440/resnet18_layer4.1.conv1.1_sparsity.npy")
# test_data = np.load("runlog/resnet18_sparsity_run_50k_2023_03_14_18_15_43_840440/resnet18_layer1.0.conv1.1_sparsity.npy")
test_data = np.load("runlog/resnet18_sparsity_run_50k_2023_03_14_18_15_43_840440/resnet18_layer2.1.conv1.1_sparsity.npy")

# with open(f"input.dat", 'w') as f:
#     f.write("\n".join([ str(i) for i in test_data.reshape(-1).tolist() ]))

# load balancing
streams = [ test_data[:,i] for i in range(test_data.shape[1]) ]
streams_avg = [ np.average(s) for s in streams ]

indices = np.argsort(streams_avg)
indices = range(len(streams))
test_data_intr = np.empty((streams[0].shape[0]*(test_data.shape[1]//STREAMS), STREAMS))
intr_factor = len(indices)//STREAMS
intr_stream_avg = [0]*STREAMS
for i in range(STREAMS):
    for j in range(intr_factor):
        idx = indices[j*STREAMS+i]
        intr_stream_avg[i] += streams_avg[idx]
        test_data_intr[j::intr_factor, i] = streams[idx]
    intr_stream_avg[i] /= intr_factor

# print(moving_average(test_data[:,0], n=10).shape)
"""
print(max_distance_metric(test_data_intr, n=1))
print(max_distance_metric(test_data_intr, n=2))
print(max_distance_metric(test_data_intr, n=5))
print(max_distance_metric(test_data_intr, n=10))
print(max_distance_metric(test_data_intr, n=20))
print(max_distance_metric(test_data_intr, n=50))
print(max_distance_metric(test_data_intr, n=100))
print(max_distance_metric(test_data_intr, n=1000))
print(max_distance_metric(test_data_intr, n=10000))
"""

#print(max_distance_metric(test_data_intr, n=2))
for i in range(2,40,1):
    print(max_distance_metric(test_data_intr, n=i))
"""
for i in range(5,50,5):
    print(max_distance_metric(test_data_intr, n=i))
for i in range(50,1501,50):
    print(max_distance_metric(test_data_intr, n=i))
"""
