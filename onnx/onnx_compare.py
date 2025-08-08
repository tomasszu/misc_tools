import torch
from vehicle_reid.load_model import load_model_from_opts
import onnxruntime as ort
import numpy as np
import torch.nn as nn

# Switch model to CPU for export
device = "cpu"
model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", 
                                ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", 
                                remove_classifier=True)
model.to(device)
model.eval()

model.classifier.add_block[2] = nn.Sequential()  # remove classifier



#Load ONNX model
onnx_model_path = "/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.onnx"
session = ort.InferenceSession(onnx_model_path)

# Get input name for ONNX model
input_name = session.get_inputs()[0].name
# print("Input name:", input_name)

# Optional: Get output names
output_names = [output.name for output in session.get_outputs()]
# print("Output names:", output_names)

# Prepare dummy input (same shape and type as original model input)
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust shape as per your model
input_numpy = dummy_input.numpy()


# Assuming you have access to the original PyTorch model
model.eval()
with torch.no_grad():
    torch_output = model(dummy_input)

outputs = session.run(output_names, {input_name: input_numpy})

# Compare
onnx_output = outputs[0]
torch_output_np = torch_output.cpu().numpy()

# Check numerical closeness
np.testing.assert_allclose(torch_output_np, onnx_output, rtol=1e-03, atol=1e-05)
print("✅ ONNX output matches PyTorch output.")

abs_diff = np.abs(torch_output_np - onnx_output)
rel_diff = abs_diff / (np.abs(torch_output_np) + 1e-8)

print("Max abs diff:", abs_diff.max())
print("Max rel diff:", rel_diff.max())

# If the assertion passes, it means the outputs are close enough
# If it fails, it will raise an AssertionError with details about the mismatch
print("ONNX output shape:", onnx_output.shape)
print("PyTorch output shape:", torch_output_np.shape)
# print("ONNX output:", onnx_output)
# print("PyTorch output:", torch_output_np)
