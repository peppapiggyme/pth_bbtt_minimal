## setup environment on IHEP
```
singularity shell -e docker://atlasml/ml-base:latest
```

## files
```
.
├── __init__.py     # 
├── nn_inputs.py    # Get pytorch tensors (inputs) from ROOT file
├── nn_models.py    # Simple deep neural network model
├── nn_utils.py     # Helper functions
├── README.md       # This file
├── train.py        # Training and testing scripts
└── convert.py      # pytorch -> onnx model conversion
```

## usage
```
mkdir output
python train.py
python convert.py
```

## input data
```
/scratchfs/atlas/bowenzhang/public/ntuple
```