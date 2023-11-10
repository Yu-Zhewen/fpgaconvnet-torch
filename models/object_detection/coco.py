import onnx
import os
import torch
import yaml

import numpy as np
import onnx_graphsurgeon as gs

from models.base import TorchModelWrapper
# note: do NOT move ultralytic import to the top, otherwise the edit in settings will not take effect

class UltralyticsModelWrapper(TorchModelWrapper):

    def load_model(self, eval=True):
        from ultralytics import YOLO 
        self.yolo = YOLO(self.model_name)
        self.model = self.yolo.model
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # utlralytics conv bn fusion is currently not working for compressed model
        # disbale it for now
        def _fuse(verbose=True):
            return self.model
        self.model.fuse = _fuse

    def load_data(self, batch_size, workers):
        from ultralytics import settings

        COCO_PATH = os.environ.get("COCO_PATH", os.path.expanduser("~/dataset/ultralytics/datasets"))
        assert COCO_PATH.endswith("/datasets"), "dataset path should end with 'datasets'"
        # set dataset path
        settings.update({'datasets_dir': COCO_PATH})

        # note: ultralytics automatically handle the dataloaders, only need to set the path
        self.data_loaders['calibrate'] = "coco128.yaml"
        self.data_loaders['validate'] = "coco.yaml"
        
        self.batch_size = batch_size
        self.workers = workers
        
    def inference(self, mode="validate"):
        print("Inference mode: {}".format(mode))
        self.yolo.model = self.model
        return self.yolo.val(batch=self.batch_size, workers=self.workers,
            data=self.data_loaders[mode], plots=False)

    def onnx_exporter(self, onnx_path):
        path = self.yolo.export(format="onnx", simplify=True)
        os.rename(path, onnx_path)
        self.remove_detection_head_v8(onnx_path)

    def remove_detection_head_v8(self, onnx_path):
        graph = onnx.load(onnx_path)
        graph = gs.import_onnx(graph)

        max_idx = 0
        for idx, node in enumerate(graph.nodes):
            if node.name == "/model.22/Reshape":
                reshape_l_idx = idx
            if node.name == "/model.22/Reshape_1":
                reshape_m_idx = idx
            if node.name == "/model.22/Reshape_2":
                reshape_r_idx = idx

        # remove extra operations
        del graph.nodes[reshape_r_idx:-1]
        del graph.nodes[reshape_m_idx:]
        del graph.nodes[reshape_l_idx:]

        # get output layers
        concat_l = next(filter(lambda x: x.name == "/model.22/Concat", graph.nodes))
        concat_m = next(filter(lambda x: x.name == "/model.22/Concat_1", graph.nodes))
        concat_r = next(filter(lambda x: x.name == "/model.22/Concat_2", graph.nodes))

        # get the resize layers
        resize = next(filter(lambda x: x.name == "/model.10/Resize", graph.nodes))
        resize.inputs[1] = gs.Constant("roi_0", np.array([0.0,0.0,0.0,0.0]))
        resize = next(filter(lambda x: x.name == "/model.13/Resize", graph.nodes))
        resize.inputs[1] = gs.Constant("roi_1", np.array([0.0,0.0,0.0,0.0]))

        # create the output nodes
        output_l = gs.Variable("/model.22/Concat_output_0",   shape=concat_l.outputs[0].shape, dtype="float16")
        output_m = gs.Variable("/model.22/Concat_1_output_0", shape=concat_m.outputs[0].shape, dtype="float16")
        output_r = gs.Variable("/model.22/Concat_2_output_0", shape=concat_r.outputs[0].shape, dtype="float16")

        # connect the output nodes
        concat_l.outputs = [ output_l ]
        concat_m.outputs = [ output_m ]
        concat_r.outputs = [ output_r ]

        # update the graph outputs
        graph.outputs = [ concat_l.outputs[0], concat_m.outputs[0], concat_r.outputs[0] ]

        # cleanup graph
        graph.cleanup()

        # save the reduced network
        graph = gs.export_onnx(graph)
        graph.ir_version = 8 # need to downgrade the ir version
        onnx.save(graph, onnx_path)