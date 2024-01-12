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
python encoding_example.py
```

## Model Zoo

* `imagenet`: `resnet18`, `resnet50`, `mobilenet_v2`, `repvgg_a0`
* `coco`: `yolov8n`
* `camvid`: `unet`
* `cityscapes`: `unet`
* `llgmri`: `unet`
* `ucf101`: `x3d_s`, `x3d_m`
* `brats2020`: `unet3d`

## Quantization Results

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
| yolov8n | [ultralytics](https://github.com/ultralytics/ultralytics) | 37.1    | 37.1    | 0.0    | 29.6         | 35.1           |
| yolov8s | [ultralytics](https://github.com/ultralytics/ultralytics) | 39.2    | 39.1    | 0.0    | 38.7         | 36.8           |

### camvid (val, mIOU)
| Model         | Source                                          | Float32 | Fixed16 | Fixed8 | BFP8 (Layer) | BFP8 (Channel) |
|---------------|-------------------------------------------------|---------|---------|--------|--------------|----------------|
| unet          | [nncf](https://github.com/openvinotoolkit/nncf) | 71.95   | 71.95   | 61.02  | 71.60        | 71.85          |
| unet-bilinear | [nncf](https://github.com/openvinotoolkit/nncf) | 71.67   | 71.67   | 60.62  | 71.40        | 71.75          |

### cityscapes (val, mIOU)
| Model | Source                                                         | Float32 | Fixed16 | Fixed8 | BFP8 (Layer) | BFP8 (Channel) |
|-------|----------------------------------------------------------------|---------|---------|--------|--------------|----------------|
| unet  | [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) | 69.10   | 69.10   | 1.98   | 61.74        | 68.43          |

### llgmri (val, Dice coefficient)
| Model         | Source                                          | Float32 | Fixed16 | Fixed8 | BFP8 (Layer) | BFP8 (Channel) |
|---------------|-------------------------------------------------|---------|---------|--------|--------------|----------------|
| unet          | [brain-segmentation-pytorch](https://github.com/mateuszbuda/brain-segmentation-pytorch) | 90.89   | 90.88   | 80.98  | 90.95        | 90.85          |
| unet-bilinear | [brain-segmentation-pytorch](https://github.com/mateuszbuda/brain-segmentation-pytorch) | 91.05   | 91.05   | 77.51  | 91.04        | 91.03          |

### ucf101 (val-split1, top-1 acc)
| Model | Source                                                         | Float32 | Fixed16 | Fixed8 | BFP8 (Layer) | BFP8 (Channel) |
|-------|----------------------------------------------------------------|---------|---------|--------|--------------|----------------|
| x3d_s  | [mmaction2](https://github.com/open-mmlab/mmaction2) | 93.68  | 93.57   |  1.13   | 90.21  | 93.57   |
| x3d_m  | [mmaction2](https://github.com/open-mmlab/mmaction2) | 96.40  | 96.40   |  0.81   | 95.24  | 96.29   |

### brats2020 (val, Dice coefficient)
| Model | Source                                                         | Float32 | Fixed16 | Fixed8 | BFP8 (Layer) | BFP8 (Channel) |
|-------|----------------------------------------------------------------|---------|---------|--------|--------------|----------------|
| unet3d  | [BraTS20_3dUnet_3dAutoEncoder](https://www.kaggle.com/code/polomarco/brats20-3dunet-3dautoencoder) | 85.34  |  85.23  |  1.15   |  85.14  |  85.34   |

## Sparsity Results
* Q - Fixed16 Quantization
* AS - Activation Sparsity
* WS - Weight Sparsity (applying global pruning threshold)
* Post-training, without fine-tuning

| Model    | Experiment     | Accuracy | Sparsity |
|----------|----------------|----------|----------|
| resnet18 | Q+AS           | 69.74    | 50.75    |
| resnet18 | Q+AS+WS(0.005) | 69.42    | 56.33    |
| resnet18 | Q+AS+WS(0.010) | 67.36    | 61.47    |
| resnet18 | Q+AS+WS(0.015) | 58.38    | 65.91    |
| resnet18 | Q+AS+WS(0.020) | 27.91    | 69.63    |

## Encoding Results
* BFP8 (Channel) Quantization
* RLE-8, run-length encoding, use 8 bits for encoding (max length 2^8)
* Compression Ratio, average over all weights and activations

| Dataset    | Model                | Experiment | Avg Compression Ratio |
|------------|----------------------|------------|-------------------|
| coco       | yolov8n ([onnx](https://drive.google.com/file/d/1zN1K4b2SU6Mpu4lMsLg2drupGyjphSxg/view?usp=sharing))       | RLE-8      | 1.753       |
| camvid     | unet-bilinear ([onnx](https://drive.google.com/file/d/1C_Q58_NKMVfpbqg3ZbQ1IzyMSgoopex7/view?usp=sharing)) | RLE-8      | 1.175       |
| cityscapes | unet (onnx)                                                                                                | RLE-8      | GPU TIMEOUT |
| ucf101     | x3d_s ([onnx](https://drive.google.com/file/d/1gY5HGMWacbTQ5cK8MWdQgQ1lQM5VWRFb/view?usp=sharing))         | RLE-8      | 1.737       |
| ucf101     | x3d_m ([onnx](https://drive.google.com/file/d/1WaLjJYE0l_AiIrZw559Ile3xQza_wnWJ/view?usp=sharing))         | RLE-8      | 1.721       |
| brats2020  | unet3d ([onnx](https://drive.google.com/file/d/1yRPN4FRfp91pq25N5bXUTQQyezUndIaM/view?usp=sharing))        | RLE-8      | 0.821       |
| coco       | yolov8n ([onnx](https://drive.google.com/file/d/1Qns_9Mp-6K42t9cWhTi1lOHNMIQwFYe2/view?usp=sharing))       | RLE-4      | 1.317
| camvid     | unet-bilinear ([onnx](https://drive.google.com/file/d/18PquUpo61-JvsA85m0T-SNbUtYtvy6Vi/view?usp=sharing)) | RLE-4      | 0.717
| coco       | yolov8n ([onnx](https://drive.google.com/file/d/1GeJPQL5QsIuqsWpoCuaCBk7CzA80R-BT/view?usp=sharing))       | RLE-2      | 1.112
| camvid     | unet-bilinear ([onnx](https://drive.google.com/file/d/1g3F48zihC812ymH0WdNBoMPkrn84GXSi/view?usp=sharing)) | RLE-2      | 0.838
| coco       | yolov8n ([onnx](https://drive.google.com/file/d/1MBYi-0yO7iXtEnjNYI3ZoeLVaPmR91yb/view?usp=sharing))       | Huffman    | 0.824       |
| coco       | yolov8s ([onnx](https://drive.google.com/file/d/1qhdnDcahno8Hl9vnW-_ZS1milq3JRNQV/view?usp=sharing))       | Huffman    | 0.805       |
| camvid     | unet-bilinear ([onnx](https://drive.google.com/file/d/1X6Ps_qcbP7vJLgNCkHbsHtWY6aSnG8es/view?usp=sharing)) | Huffman    | 0.684       |
| cityscapes | unet ([onnx](https://drive.google.com/file/d/1d2v6VJI8B9DZY020Nq_AWQR0e8F9LH6A/view?usp=sharing))          | Huffman    | 0.692       |
| ucf101     | x3d_s ([onnx](https://drive.google.com/file/d/19c6jwuHZVcfZXPpXMaGmaK9AsRXPO5lJ/view?usp=sharing))         | Huffman    | 0.835       |
| ucf101     | x3d_m ([onnx](https://drive.google.com/file/d/1RQr0lEuROwO14F0WtObBUmuz8Na3Vci2/view?usp=sharing))         | Huffman    | 0.833       |
| brats2020  | unet3d ([onnx](https://drive.google.com/file/d/1BMwL-ilnyUYZnEW9l-m_NI25s18ak805/view?usp=sharing))        | Huffman    | 0.718       |

## Links to other repos
* Optimizer: https://github.com/AlexMontgomerie/fpgaconvnet-optimiser; https://github.com/AlexMontgomerie/samo
* Model: https://github.com/AlexMontgomerie/fpgaconvnet-model
* HLS: https://github.com/AlexMontgomerie/fpgaconvnet-hls
* Tutorial: https://github.com/AlexMontgomerie/fpgaconvnet-tutorial
