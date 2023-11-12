# fpgaconvnet-torch
PyTorch frontend for fpgaConvNet, providing emulated accuracy results for features such as quantization and sparsity.

## Code Structure
* **models/**   general interfaces for model creation, inference and onnx export.
* **quantization/**    emulation for fixed point, and block floating point representations.
* **sparsity/**    post-activation sparsity, and also tunable threshold relu.
* **optimiser_interface/**    python interface to launch fpgaconvnet optimiser and collect prediction results.

## Examples
```
python quantization_example.py
python activation_sparsity_example.py
python threshold_relu_example.py
```

## Model Zoo

* `imagenet`: `resnet18`, `resnet50`, `mobilenet_v2`, `repvgg_a0`
 

## Quantization Results

### imagenet (top-1 val acc)
| Model        | Source      | Float32 | Fixed16 | Fixed8 | BFP8 (Layer) | BFP8 (Channel) |
|--------------|-------------|---------|---------|--------|--------------|----------------|
| resnet18     | torchvision | 69.76   | 69.76   | 1.03   | 68.48        | 69.26          |
| resnet50     | torchvision | 76.13   | 76.10   | 0.36   | 74.38        | 75.75          |
| mobilenet_v2 | torchvision | 71.87   | 71.76   | 0.10   | 53.68        | 69.51          |
| repvgg_a0    | timm        | 72.41   | 72.40   | 0.21   | 0.21         | 66.08          |

## Links to other repos
* Optimizer: https://github.com/AlexMontgomerie/fpgaconvnet-optimiser; https://github.com/AlexMontgomerie/samo
* Model: https://github.com/AlexMontgomerie/fpgaconvnet-model
* HLS: https://github.com/AlexMontgomerie/fpgaconvnet-hls
* Tutorial: https://github.com/AlexMontgomerie/fpgaconvnet-tutorial   
