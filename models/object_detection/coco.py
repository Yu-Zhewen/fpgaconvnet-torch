class UltralyticsModelWrapper(TorchModelWrapper):
    def load_model(self, eval=True):
        self.model = YOLO(f"{self.model_name}.pt") 