IMAGENET_PATH=${HOME}/dataset/ILSVRC2012_img

base:
	pip install -r requirements.txt
	pip install nvidia-pyindex
	pip install onnx-graphsurgeon

torch-cpu: base
	pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu

torch-gpu: base
	pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

cifar: ;

imagenet:
	pip install timm==0.9.8
	
	mkdir -p ${IMAGENET_PATH}

	cd ${IMAGENET_PATH} && \
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar && \
	mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train && \
	tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar && \
	find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
	
	cd ${IMAGENET_PATH} && \
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar && \
	mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && \
	tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar	&& \
	wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

coco: ;

camvid: ;

lggmri: ;

ucf101: ;

brats2020: ;
