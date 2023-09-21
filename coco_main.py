import argparse
import pathlib
import random
import torch

from sparsity_utils import *
from ultralytics import YOLO # install this fork, https://github.com/Yu-Zhewen/ultralytics

parser = argparse.ArgumentParser(description='ultralytics COCO')
parser.add_argument('-a', '--arch', default='yolov8n')
parser.add_argument('--output_path', default=None, type=str,
                    help='output path')
args = parser.parse_args()
if args.output_path == None:
    args.output_path = os.getcwd() + "/output"
pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)
print(args)

random.seed(0)
torch.manual_seed(0)

# Load a model
model = YOLO("yolov8n.pt") 
replace_with_vanilla_convolution(model)
results = model.val(plots=False, batch=1, data="coco128.yaml") #data="coco128.yaml"(subset 128 images) or "coco.yaml" 
output_sparsity_to_csv("yolov8n", model, args.output_path)
print(results)