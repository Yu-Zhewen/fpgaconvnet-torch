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
* `coco`: `yolov8n`
* `camvid`: `unet`
* `cityscapes`: `unet`

## Quantization Results
@ commit ec09e56
```
bash scripts/run_quantization.sh
```

### imagenet (val, top-1 acc)
| Model        | Source                                                      | Float32 | Fixed16 | Fixed8 | BFP8 (Layer) | BFP8 (Channel) |
|--------------|-------------------------------------------------------------|---------|---------|--------|--------------|----------------|
| resnet18     | [torchvision](https://github.com/pytorch/vision)            | 69.76   | 69.76   | 1.03   | 68.48        | 69.26          |
| resnet50     | [torchvision](https://github.com/pytorch/vision)            | 76.13   | 76.10   | 0.36   | 74.38        | 75.75          |
| mobilenet_v2 | [torchvision](https://github.com/pytorch/vision)            | 71.87   | 71.76   | 0.10   | 53.68        | 69.51          |
| repvgg_a0    | [timm](https://github.com/huggingface/pytorch-image-models) | 72.41   | 72.40   | 0.21   | 0.21         | 66.08          |

### coco (val, mAP50-95)
| Model   | Source                                                    | Float32 | Fixed16 | Fixed8 | BFP8 (Layer) | BFP8 (Channel) |
|---------|-----------------------------------------------------------|---------|---------|--------|--------------|----------------|
| yolov8n | [ultralytics](https://github.com/ultralytics/ultralytics) | 37.1    | 37.1    | 0.0    | 0.0          | 35.1           |

### camvid (val, mIOU)
| Model         | Source                                          | Float32 | Fixed16 | Fixed8 | BFP8 (Layer) | BFP8 (Channel) |
|---------------|-------------------------------------------------|---------|---------|--------|--------------|----------------|
| unet          | [nncf](https://github.com/openvinotoolkit/nncf) | 71.95   | 71.95   | 61.02  | 71.60        | 71.85          |
| unet-bilinear | [nncf](https://github.com/openvinotoolkit/nncf) | 71.67   | 71.67   | 60.62  | 71.40        | 71.75          |

### cityscapes (val, mIOU)
| Model | Source                                                         | Float32 | Fixed16 | Fixed8 | BFP8 (Layer) | BFP8 (Channel) |
|-------|----------------------------------------------------------------|---------|---------|--------|--------------|----------------|
| unet  | [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) | 69.10   | 69.10   | 1.98   | 61.74        | 68.43          |

## Links to other repos
* Optimizer: https://github.com/AlexMontgomerie/fpgaconvnet-optimiser; https://github.com/AlexMontgomerie/samo
* Model: https://github.com/AlexMontgomerie/fpgaconvnet-model
* HLS: https://github.com/AlexMontgomerie/fpgaconvnet-hls
* Tutorial: https://github.com/AlexMontgomerie/fpgaconvnet-tutorial
