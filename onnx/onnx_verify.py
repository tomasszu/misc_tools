import onnxruntime as ort
import numpy as np
import torch

# Load ONNX model
onnx_model_path = "/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.onnx"
session = ort.InferenceSession(onnx_model_path)

# Get input name for ONNX model
input_name = session.get_inputs()[0].name
print("Input name:", input_name)

# Optional: Get output names
output_names = [output.name for output in session.get_outputs()]
print("Output names:", output_names)

# Prepare dummy input (same shape and type as original model input)
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust shape as per your model
input_numpy = dummy_input.numpy()

# Run inference
outputs = session.run(output_names, {input_name: input_numpy})

# Inspect output
print("ONNX model output:", outputs[0].shape)
