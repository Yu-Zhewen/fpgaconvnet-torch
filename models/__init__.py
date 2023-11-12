def initialize_wrapper(dataset_name, model_name,
                       dataset_path, batch_size, workers):
    model_wrapper = None
    if dataset_name == "imagenet":
        os.environ['IMAGENET_PATH'] = dataset_path
        if model_name in ["resnet18", "resnet50", "mobilenet_v2"]:
            from models.classification.imagenet import TorchvisionModelWrapper
            model_wrapper = TorchvisionModelWrapper(model_name)
        elif model_name in ["repvgg_a0"]:
            from models.classification.imagenet import TimmModelWrapper
            model_wrapper = TimmModelWrapper(model_name)
    elif dataset_name == "coco":
        os.environ['COCO_PATH'] = dataset_path
        if model_name in ["yolov8n"]:
            from models.detection.coco import UltralyticsModelWrapper
            model_wrapper = UltralyticsModelWrapper(model_name)
    elif dataset_name == "camvid":
        os.environ['CAMVID_PATH'] = dataset_path
        if model_name in ["unet"]:
            from models.segmentation.camvid import NncfModelWrapper
            model_wrapper = NncfModelWrapper(model_name)
    elif dataset_name == "cityscapes":
        os.environ['CITYSCAPES_PATH'] = dataset_path
        if model_name in ["unet"]:
            from models.segmentation.cityscapes import MmsegmentationModelWrapper
            model_wrapper = MmsegmentationModelWrapper(model_name)
    elif dataset_name == "lggmri":
        os.environ['LGGMRI_PATH'] = dataset_path
        if model_name in ["unet"]:
            from models.segmentation.lggmri import BrainModelWrapper
            model_wrapper = BrainModelWrapper(model_name)
    
    if model_wrapper is None:
        raise NotImplementedError("Unknown dataset/model combination")

    model_wrapper.load_data(batch_size, workers)
    model_wrapper.load_model()
    return model_wrapper