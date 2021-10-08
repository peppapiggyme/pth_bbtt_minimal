"""
o------o
| NOTE |
o------o

* Convert pytorch model to onnx model 

o------o
| TODO |
o------o


"""

import torch
from nn_inputs import *
from nn_models import *

model = BBTT_DNN()
model.load_state_dict(torch.load(f"output/model-8.pt"))

dummy_input = torch.randn(1, 5)

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = ["input"]
output_names = ["output"]

torch.onnx.export(
    model, 
    dummy_input, 
    "output/model.onnx", 
    verbose=True, 
    input_names=input_names, 
    output_names=output_names)
