import torch
import torch.nn as nn

from utils import replace_modules
import os

class VariableReLUWrapper(nn.Module):
    def __init__(self, threshold):
        super(VariableReLUWrapper, self).__init__()

        self.threshold = threshold

    def forward(self, x):
        """
        Parameters
        ----------
        x: tensor
            Data input to layer

        Returns
        -------
        tensor
            output after applying ReLU with specified threshold
        """

        return (x >= self.threshold)*x
        
def replace_with_variable_relu(model, threshold = 0):
    replace_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            new_module = VariableReLUWrapper(threshold)
            replace_dict[module] = new_module

    replace_modules(model, replace_dict)


def output_accuracy_to_csv(model_name, relu_threshold, top1, top5):
    file_path = os.path.join(os.getcwd(), "runlog", str(model_name) + "_accuracy_var_relu.csv")
    
    if not (os.path.isfile(file_path)):
        with open(file_path, "w") as f:    
            f.write("ReLU Threshold,Top1 Accuracy,Top5 Accuracy\n")

    with open(file_path, "a") as f:
            row = ",".join([str(relu_threshold), str(top1), str(top5)]) + "\n"
            f.write(row)


if __name__ == "__main__":
    output_accuracy_to_csv("test", 0.01, 65, 90)