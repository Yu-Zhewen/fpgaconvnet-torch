# imagenet
python quantization_example.py --dataset_name imagenet --dataset_path ~/dataset/ILSVRC2012_img --model_name resnet18 --batch-size 512 --gpu 0
python quantization_example.py --dataset_name imagenet --dataset_path ~/dataset/ILSVRC2012_img --model_name resnet50 --batch-size 256 --gpu 0
python quantization_example.py --dataset_name imagenet --dataset_path ~/dataset/ILSVRC2012_img --model_name mobilenet_v2 --batch-size 256 --gpu 0
python quantization_example.py --dataset_name imagenet --dataset_path ~/dataset/ILSVRC2012_img --model_name repvgg_a0 --batch-size 512 --gpu 0

# coco
python quantization_example.py --dataset_name coco --dataset_path ~/dataset/ultralytics/datasets --model_name yolov8n --batch-size 128 --gpu 0

# camvid
python quantization_example.py --dataset_name camvid --dataset_path ~/dataset/CamVid --model_name unet --batch-size 1 --gpu 0

# cityscapes
python quantization_example.py --dataset_name cityscapes --dataset_path ~/dataset/cityscapes --model_name unet --batch-size 1 --gpu 0
 