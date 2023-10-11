## relu_main.py 
The relu_main script is used to run and log sparsity collection and optimiser for different relu tuning policies

### Notes
Assumes the existence of a the following directories:
- ./runlog
- ./runlog/&lt;arch-name&lt; : Directory to store sparsity information
- ./relu_thresholds
- ./relu_thresholds/&lt;arch-name&lt; : Directory to store json files containing relu threshold information. Annotated to onnx model by onnx_sparsity_attribute.py
- ./onnx_models
- ./onnx_models/&lt;arch-name&lt; : Directory to store annotared onnx models
- ../../fpgaconvnet-optimiser/fpgaconvnet/optimiser/&lt;arch-name&lt; : Directory to stor eoptimiser outputs

Uses the krish-skipping branches of fpgaconvnet-optimiser and fpgaconvnet-model


### Usage
'''
python relu_main.py
'''

### Flags
- **arch**: model_name
- **relu-policy**: relu policy choice between slowest_node and uniform 
- **fixed-hardware**: Uses fixed-hardware and does not run optimiser. Must provide "platform_path", and "optimised_config_path" flags to load fixed hardware
- **normalise-hardware**: Runs optimiser on same DSPs for dense and no skipping windows
- **accuracy_path**: "model_path". For fixed hardware
- **model_path**: Path to sparse .onnx model. 
- **platform_path**: Path to platform specs (.toml). For fixed hardware
- **gpu**
- **enable-wandb**

### Parameters you may want to vary
- **THRESHOLD_INC in relu_main.py**: Amount you want to increase ReLU by for each iteration
- **--gain flag in fpgaconvnet-optimiser cli**: Minimum gain to push fine

### Example Usage:
#### **Uniform Increase with changing hardware for resnet18**:
'''
python relu_main.py -a resnet18 --relu_policy uniform
'''

#### **Uniform Increase with fixed hardware for resnet50**:
'''
python relu_main.py -a resnet50 --fixed-hardware --relu_policy uniform
'''

#### **Slowest node Increase with changing hardware  for vgg11**:
'''
python relu_main.py -a vgg11 --relu_policy slowest_node
'''

#### **Slowest node Increase with changing hardware compared to normalised sparse and dense for vgg11**:
'''
python relu_main.py -a vgg11 --normalise-hardware --relu_policy slowest_node
'''

#### **Slowest node Increase with slowest hardware for resnet18**:
'''
python relu_main.py -a resnet18 --fixed-hardware --relu_policy uniform
'''


<!-- ## Collecting sparsity for a specific relu  threshold configuration -->



