# fpgaconvnet-torch
PyTorch frontend for fpgaConvNet, providing emulated accuracy results for features such as quantization and sparsity.

## Code Structure
* **models/**   general interfaces for model creation, inference and onnx export.
* **quantization/**    emulation for fixed point, and block floating point representations.
* **sparsity/**    post-activation sparsity, and also tunable threshold relu.
* **optimiser_interface/**    python interface to launch fpgaconvnet optimiser and collect prediction results.

## Examples
#### Qunaitzation for ImageNet Classification 
```
python imagenet_quantization.py
```

#### Qunaitzation for COCO Object Detection 
```
python coco_quantization.py
```

#### Sparisty for ImageNet Classification 
```
python imagenet_activation_sparsity.py
```

#### Hardware-aware Sparisty for ImageNet Classification 
```
python imagenet_activation_sparsity.py
```

## Links to other repos
* Optimizer: https://github.com/AlexMontgomerie/fpgaconvnet-optimiser; https://github.com/AlexMontgomerie/samo
* Model: https://github.com/AlexMontgomerie/fpgaconvnet-model
* HLS: https://github.com/AlexMontgomerie/fpgaconvnet-hls
* Tutorial: https://github.com/AlexMontgomerie/fpgaconvnet-tutorial   
